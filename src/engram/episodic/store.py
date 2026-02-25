"""ChromaDB-backed episodic memory store.

Handles embeddings manually (not via ChromaDB embedding_function) to avoid
dimension mismatch when switching between providers (e.g. default 384 vs gemini 3072).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm

from engram.audit import AuditLogger
from engram.config import EmbeddingConfig, EpisodicConfig, ScoringConfig
from engram.episodic.fts_index import FtsIndex
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


@functools.lru_cache(maxsize=1)
def _get_default_ef():
    """Lazy-load ChromaDB's default embedding function (all-MiniLM-L6-v2, 384d)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


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
        scoring: ScoringConfig | None = None,
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
        self._decay_enabled = getattr(config, "decay_enabled", True)
        self._default_decay_rate = getattr(config, "decay_rate", 0.1)
        self._scoring = scoring or ScoringConfig()
        # FTS5 full-text search index (always-on, no config needed)
        self._fts = FtsIndex()

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
        topic_key: str | None = None,
    ) -> str:
        """Store a memory and return its ID."""
        content = sanitize_content(content)

        # Topic key upsert: update existing memory if same topic_key exists
        if topic_key:
            existing = await self._find_by_topic_key(topic_key)
            if existing:
                return await self._update_topic(existing, content, topic_key, metadata, entities, tags, priority, memory_type)

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
            "access_count": 0,
            "decay_rate": self._default_decay_rate,
        }
        if topic_key:
            doc_metadata["topic_key"] = topic_key
            doc_metadata["revision_count"] = 0
        if expires_at is not None:
            doc_metadata["expires_at"] = expires_at.isoformat()
        if metadata:
            doc_metadata.update(metadata)

        try:
            collection = self._ensure_collection()
            self._detect_embedding_dim()
            embeddings = await asyncio.to_thread(_get_embeddings, self._embed_model, [content], self._embedding_dim)
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d. "
                    "Use a consistent embedding model or reset the collection."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            await asyncio.to_thread(
                collection.add,
                ids=[memory_id],
                documents=[content],
                embeddings=embeddings,
                metadatas=[doc_metadata],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to store memory: {e}") from e

        # Sync to FTS5 index (fire-and-forget; non-blocking on failure)
        try:
            mt_str = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
            await asyncio.to_thread(self._fts.insert, memory_id, content, mt_str)
        except Exception as e:
            logger.debug("FTS5 insert failed for %s: %s", memory_id, e)

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
            self._audit.log_modification(
                tenant_id=self._namespace, actor="system",
                mod_type="memory_create", resource_id=memory_id,
                after_value=content[:500],
                description=f"New {memory_type.value if isinstance(memory_type, MemoryType) else memory_type} memory (priority={priority})",
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
            # M6 fix: sanitize content just like single remember() does
            content = sanitize_content(mem["content"])
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
                "access_count": 0,
                "decay_rate": self._default_decay_rate,
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
            embeddings = await asyncio.to_thread(_get_embeddings, self._embed_model, documents, self._embedding_dim)
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            # Single upsert call for all memories
            await asyncio.to_thread(
                collection.upsert,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to batch store memories: {e}") from e

        # Sync to FTS5 index
        try:
            fts_entries = [
                (ids[i], documents[i], metadatas[i].get("memory_type", "fact"))
                for i in range(len(ids))
            ]
            await asyncio.to_thread(self._fts.insert_batch, fts_entries)
        except Exception as e:
            logger.debug("FTS5 batch insert failed: %s", e)

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
            query_embedding = await asyncio.to_thread(_get_embeddings, self._embed_model, [query], self._embedding_dim)
            coll_count = await asyncio.to_thread(collection.count)
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embedding,
                "n_results": min(limit, coll_count or 1),
            }
            if filters:
                kwargs["where"] = filters

            results = await asyncio.to_thread(collection.query, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

        memories: list[EpisodicMemory] = []
        if not results["ids"] or not results["ids"][0]:
            return memories

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, EpisodicMemory]] = []
        access_ids: list[str] = []
        access_metas: list[dict[str, Any]] = []

        for i, mem_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results.get("distances") else 0.0

            # Filter out expired memories
            raw_expires = meta.get("expires_at")
            if raw_expires:
                try:
                    expires_dt = datetime.fromisoformat(raw_expires)
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    if expires_dt < now:
                        continue
                except (ValueError, TypeError):
                    pass

            memory = _build_memory(mem_id, doc, meta)

            # Filter by tags (all requested tags must be present)
            if tags:
                if not all(t in memory.tags for t in tags):
                    continue

            # Compute activation score
            similarity = max(0.0, 1.0 - distance)  # ChromaDB cosine distance → similarity
            score = _compute_activation_score(
                similarity, memory.timestamp, memory.access_count,
                memory.decay_rate, now, self._scoring, self._decay_enabled,
            )
            scored.append((score, memory))

            # Track access for batch update
            new_count = memory.access_count + 1
            access_ids.append(mem_id)
            access_metas.append({
                "access_count": new_count,
                "last_accessed": now.isoformat(),
            })

        # Re-sort by activation score
        scored.sort(key=lambda x: x[0], reverse=True)
        memories = [m for _, m in scored]

        # Fire-and-forget access tracking update
        if access_ids:
            try:
                await asyncio.to_thread(collection.update, ids=access_ids, metadatas=access_metas)
            except Exception as e:
                logger.debug("Access tracking update failed: %s", e)

        return memories

    async def cleanup_expired(self) -> int:
        """Delete all expired memories. Returns count deleted.

        Processes in paginated chunks of 1000 to avoid OOM on large stores.
        """
        _PAGE = 1000
        try:
            collection = self._ensure_collection()
            total = await asyncio.to_thread(collection.count)
            if total == 0:
                return 0
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}") from e

        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []

        # Paginate through all memories in chunks
        for offset in range(0, total, _PAGE):
            try:
                result = await asyncio.to_thread(
                    collection.get, include=["metadatas"], limit=_PAGE, offset=offset,
                )
            except Exception:
                break
            for i, mem_id in enumerate(result["ids"]):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                raw_expires = meta.get("expires_at")
                if raw_expires:
                    try:
                        expires_dt = datetime.fromisoformat(raw_expires)
                        if expires_dt.tzinfo is None:
                            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                        if expires_dt < now:
                            expired_ids.append(mem_id)
                    except (ValueError, TypeError):
                        pass

        if expired_ids:
            # Delete in batches too
            for i in range(0, len(expired_ids), _PAGE):
                batch = expired_ids[i:i + _PAGE]
                await asyncio.to_thread(collection.delete, ids=batch)
            if self._audit:
                self._audit.log_modification(
                    tenant_id=self._namespace, actor="system",
                    mod_type="cleanup_expired", resource_id="",
                    after_value={"deleted_count": len(expired_ids), "ids": expired_ids[:20]},
                    reversible=False, description=f"Cleaned up {len(expired_ids)} expired memories",
                )

        return len(expired_ids)

    async def get_recent(self, n: int = 20) -> list[EpisodicMemory]:
        """Retrieve the most recent N memories sorted by timestamp descending."""
        n = min(n, 1000)  # Hard cap to prevent unbounded fetches
        try:
            collection = self._ensure_collection()
            count = await asyncio.to_thread(collection.count)
            if count == 0:
                return []
            result = await asyncio.to_thread(
                collection.get,
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
            result = await asyncio.to_thread(self._ensure_collection().get, ids=[id])
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
            # Capture before-value for audit trail
            before_content = None
            if self._audit and self._audit.enabled:
                try:
                    result = await asyncio.to_thread(self._ensure_collection().get, ids=[id], include=["documents", "metadatas"])
                    if result["ids"]:
                        before_content = result["documents"][0] if result["documents"] else ""
                except Exception:
                    pass

            await asyncio.to_thread(self._ensure_collection().delete, ids=[id])

            # Remove from FTS5 index
            try:
                await asyncio.to_thread(self._fts.delete, id)
            except Exception as e:
                logger.debug("FTS5 delete failed for %s: %s", id, e)

            if self._audit:
                self._audit.log_modification(
                    tenant_id=self._namespace, actor="system",
                    mod_type="memory_delete", resource_id=id,
                    before_value=before_content, after_value=None,
                    reversible=False, description="Memory deleted",
                )
            return True
        except Exception:
            return False

    async def search_fulltext(self, query: str, limit: int = 10) -> list[EpisodicMemory]:
        """Search memories using FTS5 exact keyword matching.

        Returns EpisodicMemory objects fetched from ChromaDB by ID.
        Skips IDs not found in ChromaDB (may have been deleted without FTS sync).
        """
        fts_results = await asyncio.to_thread(self._fts.search, query, limit)
        if not fts_results:
            return []

        ids = [r.id for r in fts_results]
        try:
            result = await asyncio.to_thread(
                self._ensure_collection().get,
                ids=ids,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.debug("FTS5 ChromaDB fetch failed: %s", e)
            return []

        memories: list[EpisodicMemory] = []
        for i, mem_id in enumerate(result.get("ids", [])):
            doc = result["documents"][i] if result.get("documents") else ""
            meta = result["metadatas"][i] if result.get("metadatas") else {}
            memories.append(_build_memory(mem_id, doc, meta))
        return memories

    async def stats(self) -> dict[str, Any]:
        """Return collection statistics including embedding dimension."""
        collection = self._ensure_collection()
        count = await asyncio.to_thread(collection.count)
        self._detect_embedding_dim()
        result: dict[str, Any] = {
            "count": count,
            "collection": self.COLLECTION_NAME,
            "namespace": self._namespace,
        }
        if self._embedding_dim is not None:
            result["embedding_dim"] = self._embedding_dim
        return result


    async def update_metadata(self, mem_id: str, metadata: dict[str, Any]) -> bool:
        """Update metadata fields on an existing memory. Returns True on success."""
        try:
            collection = self._ensure_collection()
            # Capture before-value for audit trail
            before_meta = None
            if self._audit and self._audit.enabled:
                try:
                    existing = await asyncio.to_thread(collection.get, ids=[mem_id], include=["metadatas"])
                    if existing["metadatas"]:
                        before_meta = {k: existing["metadatas"][0].get(k) for k in metadata}
                except Exception:
                    pass

            await asyncio.to_thread(collection.update, ids=[mem_id], metadatas=[metadata])

            if self._audit:
                self._audit.log_modification(
                    tenant_id=self._namespace, actor="system",
                    mod_type="metadata_update", resource_id=mem_id,
                    before_value=before_meta, after_value=metadata,
                    description=f"Updated fields: {', '.join(metadata.keys())}",
                )
            return True
        except Exception as e:
            logger.warning("Failed to update metadata for %s: %s", mem_id, e)
            return False

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

    async def _find_by_topic_key(self, topic_key: str) -> str | None:
        """Find memory ID by topic_key using ChromaDB where filter."""
        try:
            collection = self._ensure_collection()
            result = await asyncio.to_thread(collection.get, where={"topic_key": topic_key})
            if result["ids"]:
                return result["ids"][0]
        except Exception:
            pass
        return None

    async def _update_topic(
        self, mem_id: str, content: str, topic_key: str,
        metadata: dict[str, Any] | None, entities: list[str] | None,
        tags: list[str] | None, priority: int, memory_type: MemoryType,
    ) -> str:
        """Re-embed and update existing topic-keyed memory."""
        collection = self._ensure_collection()
        existing = await asyncio.to_thread(collection.get, ids=[mem_id], include=["metadatas"])
        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        revision = int(old_meta.get("revision_count", 0)) + 1

        self._detect_embedding_dim()
        embeddings = await asyncio.to_thread(_get_embeddings, self._embed_model, [content], self._embedding_dim)

        new_meta: dict[str, Any] = {
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entities": json.dumps(entities or []),
            "tags": json.dumps(tags or []),
            "access_count": int(old_meta.get("access_count", 0)),
            "decay_rate": float(old_meta.get("decay_rate", self._default_decay_rate)),
            "topic_key": topic_key,
            "revision_count": revision,
        }
        if metadata:
            new_meta.update(metadata)

        # Get old content for audit trail
        old_content = None
        if self._audit and self._audit.enabled:
            try:
                old_doc = await asyncio.to_thread(collection.get, ids=[mem_id], include=["documents"])
                if old_doc["documents"]:
                    old_content = old_doc["documents"][0]
            except Exception:
                pass

        await asyncio.to_thread(
            collection.update,
            ids=[mem_id], documents=[content],
            embeddings=embeddings, metadatas=[new_meta],
        )

        # Sync updated content to FTS5 index
        try:
            mt_str = new_meta.get("memory_type", "fact")
            await asyncio.to_thread(self._fts.insert, mem_id, content, mt_str)
        except Exception as e:
            logger.debug("FTS5 update failed for %s: %s", mem_id, e)

        if self._audit:
            self._audit.log_modification(
                tenant_id=self._namespace, actor="system",
                mod_type="memory_update", resource_id=mem_id,
                before_value=old_content, after_value=content,
                description=f"Topic-key upsert (revision={revision})",
            )
        logger.info("Updated topic-keyed memory %s (revision=%d)", mem_id[:8], revision)
        return mem_id


def _compute_activation_score(
    similarity: float,
    timestamp: datetime,
    access_count: int,
    decay_rate: float,
    now: datetime,
    scoring: ScoringConfig,
    decay_enabled: bool,
) -> float:
    """Compute composite activation score for a memory.

    Components: vector similarity, Ebbinghaus retention, recency, frequency.
    """
    if not decay_enabled:
        return similarity

    # Ensure both datetimes are tz-aware for safe subtraction
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    days_old = max(0.0, (now - ts).total_seconds() / 86400)

    # Ebbinghaus retention with access reinforcement
    retention = math.exp(-decay_rate * days_old / (1 + 0.1 * access_count))
    # Recency boost (newer = higher)
    recency = 1.0 / (1.0 + days_old * 0.1)
    # Frequency boost (myelination metaphor)
    frequency = 1.0 + min(0.3, 0.05 * math.log1p(access_count))

    return (
        similarity * scoring.similarity_weight
        + retention * scoring.retention_weight
        + recency * scoring.recency_weight
        + frequency * scoring.frequency_weight
    )


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

    # Parse decay/access fields
    access_count = int(metadata.get("access_count", 0))
    decay_rate = float(metadata.get("decay_rate", 0.1))
    raw_last_accessed = metadata.get("last_accessed")
    last_accessed = None
    if raw_last_accessed:
        try:
            last_accessed = datetime.fromisoformat(raw_last_accessed)
        except (ValueError, TypeError):
            pass

    # Parse consolidation fields
    consolidation_group = metadata.get("consolidation_group")
    consolidated_into = metadata.get("consolidated_into")

    # Parse topic key fields
    topic_key = metadata.get("topic_key")
    revision_count = int(metadata.get("revision_count", 0))

    # Parse feedback fields
    confidence = float(metadata.get("confidence", 1.0))
    negative_count = int(metadata.get("negative_count", 0))

    # Exclude internal fields from extra metadata
    _internal = {
        "memory_type", "priority", "timestamp", "entities", "tags", "expires_at",
        "access_count", "last_accessed", "decay_rate",
        "consolidation_group", "consolidated_into",
        "topic_key", "revision_count",
        "confidence", "negative_count",
    }
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
        access_count=access_count,
        last_accessed=last_accessed,
        decay_rate=decay_rate,
        consolidation_group=consolidation_group,
        consolidated_into=consolidated_into,
        topic_key=topic_key,
        revision_count=revision_count,
        confidence=confidence,
        negative_count=negative_count,
    )
