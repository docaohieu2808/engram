"""ChromaDB-backed episodic memory store.

Handles embeddings manually (not via ChromaDB embedding_function) to avoid
dimension mismatch when switching between providers (e.g. default 384 vs gemini 3072).

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
    """ChromaDB-backed episodic memory store.

    Manages embeddings externally to prevent dimension conflicts.
    Collection uses cosine similarity with no built-in embedding function.
    Supports namespaces (separate ChromaDB collections) and TTL expiry.
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
        # Namespace determines which ChromaDB collection we use
        self._namespace = namespace or getattr(config, "namespace", "default") or "default"
        self.COLLECTION_NAME = _collection_name(self._namespace)
        # Build backend via factory (embedded by default, http when config.mode == "http")
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
        """Return a sentinel so legacy synchronous callers get a non-None value.

        Real initialisation is async; mixins must call `await self._ensure_backend()`
        before any backend operation. This shim preserves backward compat for any
        code that still checks `self._collection is not None`.
        """
        # Return backend so legacy `collection.xyz()` patterns surface a clear error
        # rather than AttributeError on None. Mixins are updated to use self._backend directly.
        return self._backend

    async def _ensure_backend(self) -> None:
        """Initialise backend on first async call (lazy, idempotent).

        Also handles legacy test setups that inject a raw ChromaDB collection
        mock via `store._collection = mock_obj` without calling __init__.
        In that case a _LegacyCollectionBackend shim is created automatically.
        """
        if getattr(self, "_backend_initialised", False):
            return

        if not hasattr(self, "_backend"):
            # Legacy test path: _collection was injected directly (MagicMock or real collection)
            col = getattr(self, "_collection", None)
            if col is not None:
                from engram.episodic._legacy_collection_backend import _LegacyCollectionBackend
                self._backend = _LegacyCollectionBackend(col)
                self._backend_initialised = True
                return
            # No collection and no backend — something is wrong
            raise RuntimeError(
                "EpisodicStore: no backend configured. Call __init__ or set _collection."
            )

        await self._backend.initialize(self.COLLECTION_NAME, self._embedding_dim)
        self._backend_initialised = True

    async def _detect_embedding_dim(self) -> None:
        """Detect embedding dimension from existing collection data (D-H1: async to avoid blocking)."""
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
            peek = await self._backend.peek(1)
            if peek and peek.get("embeddings") and peek["embeddings"][0]:
                self._embedding_dim = len(peek["embeddings"][0])
