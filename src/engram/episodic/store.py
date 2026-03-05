"""Qdrant-backed episodic memory store.

Handles embeddings manually (not via built-in embedding_function) to avoid
dimension mismatch when switching between embedding providers (e.g. 384 vs 3072).

Embedding helpers → episodic/embeddings.py
Decay scoring    → episodic/decay.py

Mixin structure:
  _EpisodicCrudMixin       → episodic_crud.py
  _EpisodicSearchMixin     → episodic_search.py
  _BatchMixin              → batch_operations.py
  _EpisodicMaintenanceMixin → episodic_maintenance.py
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

from engram.audit import AuditLogger
from engram.config import EmbeddingConfig, EpisodicConfig, ScoringConfig
from engram.episodic.batch_operations import _BatchMixin
from engram.episodic.episodic_builder import _collection_name
from engram.episodic.episodic_crud import _EpisodicCrudMixin
from engram.episodic.episodic_maintenance import _EpisodicMaintenanceMixin
from engram.episodic.episodic_search import _EpisodicSearchMixin
from engram.episodic.embeddings import _detect_embedding_dim_from_model, _get_embeddings  # noqa: F401 — re-exported for test patches
from engram.episodic.fts_index import FtsIndex

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")


class EpisodicStore(
    _EpisodicCrudMixin,
    _EpisodicSearchMixin,
    _BatchMixin,
    _EpisodicMaintenanceMixin,
):
    """Qdrant-backed episodic memory store.

    Manages embeddings externally to prevent dimension conflicts.
    Collection uses cosine similarity with no built-in embedding function.
    Supports namespaces (separate Qdrant collections) and TTL expiry.
    """

    def __init__(
        self,
        config: EpisodicConfig,
        embedding_config: EmbeddingConfig,
        namespace: str | None = None,
        on_remember_hook: str | None = None,
        audit: AuditLogger | None = None,
        scoring: ScoringConfig | None = None,
        guard_enabled: bool = False,
    ):
        from engram.episodic import make_episodic_backend

        self._embed_model = f"{embedding_config.provider}/{embedding_config.model}"
        # Namespace determines which Qdrant collection we use
        self._namespace = namespace or getattr(config, "namespace", "default") or "default"
        self.COLLECTION_NAME = _collection_name(self._namespace)
        # Build backend via factory (Qdrant)
        self._backend = make_episodic_backend(config)
        # _collection kept as None; _ensure_collection() initialises backend lazily
        self._collection = None
        self._backend_initialised = False
        self._embedding_dim: int | None = None  # Detected on first operation
        self._on_remember_hook = on_remember_hook
        self._audit = audit
        self._decay_enabled = getattr(config, "decay_enabled", True)
        self._default_decay_rate = getattr(config, "decay_rate", 0.1)
        self._scoring = scoring or ScoringConfig()
        self._guard_enabled = guard_enabled
        self._dedup_enabled = getattr(config, "dedup_enabled", True)
        self._dedup_threshold = getattr(config, "dedup_threshold", 0.85)
        # FTS5 full-text search index — path from config if available
        fts_db_path = getattr(config, "fts_db_path", "~/.engram/fts_index.db")
        fts_enabled = getattr(config, "fts_enabled", True)
        self._fts = FtsIndex(db_path=fts_db_path) if fts_enabled else None

    def _ensure_collection(self) -> Any:
        """Return backend instance. Real initialisation is async via _ensure_backend()."""
        return self._backend

    async def _ensure_backend(self) -> None:
        """Initialise backend on first async call (lazy, idempotent)."""
        if getattr(self, "_backend_initialised", False):
            return

        if not hasattr(self, "_backend"):
            raise RuntimeError(
                "EpisodicStore: no backend configured. Call __init__ properly."
            )

        await self._backend.initialize(self.COLLECTION_NAME, self._embedding_dim)
        self._backend_initialised = True

    async def _detect_embedding_dim(self) -> None:
        """Detect embedding dimension from existing collection data."""
        if self._embedding_dim is not None:
            return
        # Check known dimensions for configured model (fast path via registry)
        known_dim = _detect_embedding_dim_from_model(self._embed_model)
        if known_dim is not None:
            self._embedding_dim = known_dim
            return
        # Peek at existing data to detect dimension
        await self._ensure_backend()
        count = await self._backend.count()
        if count > 0:
            try:
                peek = await self._backend.peek(1)
                if peek and peek.get("embeddings") and peek["embeddings"][0]:
                    self._embedding_dim = len(peek["embeddings"][0])
            except Exception as exc:
                logger.warning("_detect_embedding_dim: peek failed (%s), skipping", exc)
