"""ChromaDB-backed episodic memory store.

Handles embeddings manually (not via ChromaDB embedding_function) to avoid
dimension mismatch when switching between providers (e.g. default 384 vs gemini 3072).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import litellm
litellm.suppress_debug_info = True

from engram.config import EmbeddingConfig, EpisodicConfig
from engram.models import EpisodicMemory, MemoryType

# Known embedding dimensions per model
_EMBEDDING_DIMS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
}


_default_ef = None


def _get_default_ef():
    """Lazy-load ChromaDB's default embedding function (all-MiniLM-L6-v2, 384d)."""
    global _default_ef
    if _default_ef is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _default_ef = DefaultEmbeddingFunction()
    return _default_ef


def _ensure_api_key() -> None:
    """Load GEMINI_API_KEY from ~/.bashrc if not in environment."""
    import os, re
    if os.environ.get("GEMINI_API_KEY"):
        return
    for rc in (".bashrc", ".zshrc", ".profile"):
        rc_path = os.path.expanduser(f"~/{rc}")
        try:
            with open(rc_path) as f:
                match = re.search(r'export GEMINI_API_KEY="([^"]+)"', f.read())
                if match:
                    os.environ["GEMINI_API_KEY"] = match.group(1)
                    return
        except FileNotFoundError:
            continue


def _get_embeddings(model: str, texts: list[str], expected_dim: int | None = None) -> list[list[float]]:
    """Generate embeddings via litellm, fallback to ChromaDB default on error.

    Args:
        expected_dim: If set, validates fallback embeddings match this dimension.
                      Raises RuntimeError on mismatch to prevent silent corruption.
    """
    _ensure_api_key()
    try:
        response = litellm.embedding(model=model, input=texts)
        return [item["embedding"] for item in response.data]
    except Exception:
        import warnings
        fallback = _get_default_ef()(input=texts)
        fallback_dim = len(fallback[0]) if fallback else 0
        # Prevent dimension mismatch: if collection already has higher-dim embeddings,
        # silently inserting 384d vectors would corrupt search quality
        if expected_dim and fallback_dim != expected_dim:
            raise RuntimeError(
                f"Embedding API unavailable and fallback dimension ({fallback_dim}d) "
                f"doesn't match collection ({expected_dim}d). "
                f"Set GEMINI_API_KEY or ensure API access to avoid data corruption."
            )
        warnings.warn(
            "Embedding API unavailable, using ChromaDB default (384d). "
            "Set GEMINI_API_KEY for higher-quality embeddings.",
            stacklevel=2,
        )
        return fallback


class EpisodicStore:
    """ChromaDB-backed episodic memory store.

    Manages embeddings externally to prevent dimension conflicts.
    Collection uses cosine similarity with no built-in embedding function.
    """

    COLLECTION_NAME = "engram_memories"

    def __init__(self, config: EpisodicConfig, embedding_config: EmbeddingConfig):
        import chromadb
        from pathlib import Path
        import os

        db_path = str(Path(os.path.expanduser(config.path)))
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        self._embed_model = f"{embedding_config.provider}/{embedding_config.model}"
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_dim: int | None = None  # Detected on first operation

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
        entities: list[str] | None = None,
    ) -> str:
        """Store a memory and return its ID."""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        entities = entities or []

        doc_metadata: dict[str, Any] = {
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "priority": priority,
            "timestamp": timestamp,
            "entities": ",".join(entities),
        }
        if metadata:
            doc_metadata.update(metadata)

        try:
            self._detect_embedding_dim()
            embeddings = _get_embeddings(self._embed_model, [content], self._embedding_dim)
            self._embedding_dim = self._embedding_dim or len(embeddings[0])
            self._collection.add(
                ids=[memory_id],
                documents=[content],
                embeddings=embeddings,
                metadatas=[doc_metadata],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to store memory: {e}") from e

        return memory_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicMemory]:
        """Search memories by semantic similarity with optional metadata filters."""
        try:
            self._detect_embedding_dim()
            query_embedding = _get_embeddings(self._embed_model, [query], self._embedding_dim)
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embedding,
                "n_results": min(limit, self._collection.count() or 1),
            }
            if filters:
                kwargs["where"] = filters

            results = self._collection.query(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

        memories: list[EpisodicMemory] = []
        if not results["ids"] or not results["ids"][0]:
            return memories

        for i, mem_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            memories.append(_build_memory(mem_id, doc, meta))

        return memories

    async def get(self, id: str) -> EpisodicMemory | None:
        """Retrieve a single memory by ID."""
        try:
            result = self._collection.get(ids=[id])
        except Exception:
            return None

        if not result["ids"]:
            return None

        doc = result["documents"][0] if result["documents"] else ""
        meta = result["metadatas"][0] if result["metadatas"] else {}
        return _build_memory(result["ids"][0], doc, meta)

    async def delete(self, id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        try:
            self._collection.delete(ids=[id])
            return True
        except Exception:
            return False

    async def stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "count": count,
            "collection": self.COLLECTION_NAME,
        }


    def _detect_embedding_dim(self) -> None:
        """Detect embedding dimension from existing collection data."""
        if self._embedding_dim is not None:
            return
        # Check known dimensions for configured model
        model_name = self._embed_model.split("/")[-1] if "/" in self._embed_model else self._embed_model
        if model_name in _EMBEDDING_DIMS:
            self._embedding_dim = _EMBEDDING_DIMS[model_name]
            return
        # Peek at existing data to detect dimension
        if self._collection.count() > 0:
            peek = self._collection.peek(limit=1)
            if peek and peek.get("embeddings") and peek["embeddings"][0]:
                self._embedding_dim = len(peek["embeddings"][0])


def _build_memory(mem_id: str, document: str, metadata: dict[str, Any]) -> EpisodicMemory:
    """Construct EpisodicMemory from ChromaDB result fields."""
    raw_type = metadata.get("memory_type", MemoryType.FACT.value)
    try:
        memory_type = MemoryType(raw_type)
    except ValueError:
        memory_type = MemoryType.FACT

    raw_ts = metadata.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now()
    except (ValueError, TypeError):
        timestamp = datetime.now()

    raw_entities = metadata.get("entities", "")
    entities = [e.strip() for e in raw_entities.split(",") if e.strip()] if raw_entities else []

    # Exclude internal fields from extra metadata
    extra = {k: v for k, v in metadata.items()
              if k not in ("memory_type", "priority", "timestamp", "entities")}

    return EpisodicMemory(
        id=mem_id,
        content=document,
        memory_type=memory_type,
        priority=int(metadata.get("priority", 5)),
        metadata=extra,
        entities=entities,
        timestamp=timestamp,
    )
