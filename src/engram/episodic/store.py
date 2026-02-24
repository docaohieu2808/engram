"""ChromaDB-backed episodic memory store.

Handles embeddings manually (not via ChromaDB embedding_function) to avoid
dimension mismatch when switching between providers (e.g. default 384 vs gemini 3072).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm

from engram.audit import AuditLogger
from engram.config import EmbeddingConfig, EpisodicConfig
from engram.hooks import fire_hook
from engram.models import EpisodicMemory, MemoryType
from engram.sanitize import sanitize_content

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")

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


def _get_embeddings(model: str, texts: list[str], expected_dim: int | None = None) -> list[list[float]]:
    """Generate embeddings via litellm, fallback to ChromaDB default on error.

    Args:
        expected_dim: If set, validates fallback embeddings match this dimension.
                      Raises RuntimeError on mismatch to prevent silent corruption.
    """
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


def _collection_name(namespace: str) -> str:
    """Build ChromaDB collection name from namespace."""
    return f"engram_{namespace}"


class EpisodicStore:
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
    ):
        import chromadb
        from pathlib import Path
        import os

        db_path = str(Path(os.path.expanduser(config.path)))
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        self._embed_model = f"{embedding_config.provider}/{embedding_config.model}"
        # Namespace determines which ChromaDB collection we use
        self._namespace = namespace or getattr(config, "namespace", "default") or "default"
        self.COLLECTION_NAME = _collection_name(self._namespace)
        # Defer collection creation to first operation (lazy loading)
        self._collection = None
        self._embedding_dim: int | None = None  # Detected on first operation
        self._on_remember_hook = on_remember_hook
        self._audit = audit

    def _ensure_collection(self) -> Any:
        """Create ChromaDB collection on first access (lazy initialization)."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        """Store a memory and return its ID."""
        content = sanitize_content(content)
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        entities = entities or []
        tags = tags or []

        doc_metadata: dict[str, Any] = {
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "priority": priority,
            "timestamp": timestamp,
            "entities": json.dumps(entities),
            "tags": json.dumps(tags),
        }
        if expires_at is not None:
            doc_metadata["expires_at"] = expires_at.isoformat()
        if metadata:
            doc_metadata.update(metadata)

        try:
            collection = self._ensure_collection()
            self._detect_embedding_dim()
            embeddings = _get_embeddings(self._embed_model, [content], self._embedding_dim)
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d. "
                    "Use a consistent embedding model or reset the collection."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            collection.add(
                ids=[memory_id],
                documents=[content],
                embeddings=embeddings,
                metadatas=[doc_metadata],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to store memory: {e}") from e

        fire_hook(self._on_remember_hook, {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
        })
        if self._audit:
            self._audit.log(
                tenant_id=self._namespace,
                actor="system",
                operation="episodic.remember",
                resource_id=memory_id,
                details={"memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type},
            )
        return memory_id

    async def remember_batch(self, memories: list[dict[str, Any]]) -> list[str]:
        """Store multiple memories in a single ChromaDB upsert call.

        Each dict may contain: content (required), memory_type, priority, entities, tags,
        expires_at, metadata. Returns list of memory IDs in same order as input.
        """
        if not memories:
            return []

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for mem in memories:
            content = mem["content"]
            memory_type = mem.get("memory_type", MemoryType.FACT)
            priority = mem.get("priority", 5)
            entities = mem.get("entities") or []
            tags = mem.get("tags") or []
            extra_meta = mem.get("metadata") or {}
            expires_at = mem.get("expires_at")

            mem_id = str(uuid.uuid4())
            doc_metadata: dict[str, Any] = {
                "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
                "priority": priority,
                "timestamp": timestamp,
                "entities": json.dumps(entities),
                "tags": json.dumps(tags),
            }
            if expires_at is not None:
                if isinstance(expires_at, datetime):
                    doc_metadata["expires_at"] = expires_at.isoformat()
                else:
                    doc_metadata["expires_at"] = expires_at
            doc_metadata.update(extra_meta)

            ids.append(mem_id)
            documents.append(content)
            metadatas.append(doc_metadata)

        try:
            collection = self._ensure_collection()
            self._detect_embedding_dim()
            # Single batch embedding call
            embeddings = _get_embeddings(self._embed_model, documents, self._embedding_dim)
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            # Single upsert call for all memories
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to batch store memories: {e}") from e

        return ids

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[EpisodicMemory]:
        """Search memories by semantic similarity with optional metadata filters.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            filters: ChromaDB `where` clause dict.
            tags: Optional tag list — all provided tags must be present in memory.
        """
        try:
            collection = self._ensure_collection()
            self._detect_embedding_dim()
            query_embedding = _get_embeddings(self._embed_model, [query], self._embedding_dim)
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embedding,
                "n_results": min(limit, collection.count() or 1),
            }
            if filters:
                kwargs["where"] = filters

            results = collection.query(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

        memories: list[EpisodicMemory] = []
        if not results["ids"] or not results["ids"][0]:
            return memories

        now = datetime.now()
        for i, mem_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}

            # Filter out expired memories
            raw_expires = meta.get("expires_at")
            if raw_expires:
                try:
                    expires_dt = datetime.fromisoformat(raw_expires)
                    # Strip tz info for comparison with naive now() (backward compat)
                    if expires_dt.tzinfo is not None:
                        expires_dt = expires_dt.replace(tzinfo=None)
                    if expires_dt < now:
                        continue
                except (ValueError, TypeError):
                    pass

            memory = _build_memory(mem_id, doc, meta)

            # Filter by tags (all requested tags must be present)
            if tags:
                if not all(t in memory.tags for t in tags):
                    continue

            memories.append(memory)

        return memories

    async def cleanup_expired(self) -> int:
        """Delete all expired memories. Returns count deleted."""
        try:
            collection = self._ensure_collection()
            count = collection.count()
            if count == 0:
                return 0
            # Fetch all memories (no vector query needed)
            result = collection.get(include=["metadatas"])
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}") from e

        now = datetime.now()
        expired_ids: list[str] = []

        for i, mem_id in enumerate(result["ids"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            raw_expires = meta.get("expires_at")
            if raw_expires:
                try:
                    expires_dt = datetime.fromisoformat(raw_expires)
                    if expires_dt < now:
                        expired_ids.append(mem_id)
                except (ValueError, TypeError):
                    pass

        if expired_ids:
            collection.delete(ids=expired_ids)

        return len(expired_ids)

    async def get_recent(self, n: int = 20) -> list[EpisodicMemory]:
        """Retrieve the most recent N memories sorted by timestamp descending."""
        try:
            collection = self._ensure_collection()
            count = collection.count()
            if count == 0:
                return []
            result = collection.get(
                include=["documents", "metadatas"],
                limit=min(n * 2, count),  # Fetch extra to allow sorting
            )
        except Exception as e:
            raise RuntimeError(f"get_recent failed: {e}") from e

        memories: list[EpisodicMemory] = []
        for i, mem_id in enumerate(result["ids"]):
            doc = result["documents"][i] if result["documents"] else ""
            meta = result["metadatas"][i] if result["metadatas"] else {}
            memories.append(_build_memory(mem_id, doc, meta))

        # Sort by timestamp descending, take top N
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:n]

    async def get(self, id: str) -> EpisodicMemory | None:
        """Retrieve a single memory by ID."""
        try:
            result = self._ensure_collection().get(ids=[id])
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
            self._ensure_collection().delete(ids=[id])
            return True
        except Exception:
            return False

    async def stats(self) -> dict[str, Any]:
        """Return collection statistics including embedding dimension."""
        collection = self._ensure_collection()
        count = collection.count()
        self._detect_embedding_dim()
        result: dict[str, Any] = {
            "count": count,
            "collection": self.COLLECTION_NAME,
            "namespace": self._namespace,
        }
        if self._embedding_dim is not None:
            result["embedding_dim"] = self._embedding_dim
        return result


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
        collection = self._ensure_collection()
        if collection.count() > 0:
            peek = collection.peek(limit=1)
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
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        timestamp = datetime.now(timezone.utc)

    raw_entities = metadata.get("entities", "")
    if raw_entities:
        if raw_entities.startswith("["):
            # JSON array format (new)
            try:
                entities = json.loads(raw_entities)
            except (json.JSONDecodeError, ValueError):
                entities = []
        else:
            # CSV format (backward compat for old data)
            entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
    else:
        entities = []

    # Parse tags (JSON array)
    raw_tags = metadata.get("tags", "[]")
    try:
        tags = json.loads(raw_tags) if raw_tags else []
    except (json.JSONDecodeError, ValueError):
        tags = []

    # Parse expires_at
    raw_expires = metadata.get("expires_at")
    expires_at = None
    if raw_expires:
        try:
            expires_at = datetime.fromisoformat(raw_expires)
        except (ValueError, TypeError):
            pass

    # Exclude internal fields from extra metadata
    _internal = {"memory_type", "priority", "timestamp", "entities", "tags", "expires_at"}
    extra = {k: v for k, v in metadata.items() if k not in _internal}

    return EpisodicMemory(
        id=mem_id,
        content=document,
        memory_type=memory_type,
        priority=int(metadata.get("priority", 5)),
        metadata=extra,
        entities=entities,
        tags=tags,
        timestamp=timestamp,
        expires_at=expires_at,
    )
