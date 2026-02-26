"""ChromaDB-backed episodic memory store.

Handles embeddings manually (not via ChromaDB embedding_function) to avoid
dimension mismatch when switching between providers (e.g. default 384 vs gemini 3072).

Embedding helpers → episodic/embeddings.py
Decay scoring    → episodic/decay.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm

from engram.audit import AuditLogger
from engram.config import EmbeddingConfig, EpisodicConfig, ScoringConfig
from engram.episodic.decay import compute_activation_score
from engram.episodic.embeddings import (
    _EMBEDDING_DIMS,
    _detect_embedding_dim_from_model,
    _get_embeddings,
)
from engram.episodic.fts_index import FtsIndex
from engram.hooks import fire_hook
from engram.models import EpisodicMemory, MemoryType
from engram.sanitize import sanitize_content

litellm.suppress_debug_info = True
logger = logging.getLogger("engram")


def _safe_json_list(val: str | None) -> list:
    """Parse a JSON string as list, returning [] on any failure."""
    if not val:
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


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
        guard_enabled: bool = False,
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
        self._guard_enabled = guard_enabled
        self._dedup_enabled = getattr(config, "dedup_enabled", True)
        self._dedup_threshold = getattr(config, "dedup_threshold", 0.85)
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

        if self._guard_enabled:
            from engram.ingestion.guard import check_content
            is_safe, reason = check_content(content)
            if not is_safe:
                logger.warning("Poisoning guard blocked: %s", reason)
                raise ValueError(f"Content rejected: {reason}")

        # Resolve temporal references (e.g. "hôm nay" → annotated ISO date)
        try:
            from engram.recall.temporal_resolver import resolve_temporal
            resolved_content, resolved_date = resolve_temporal(content)
            if resolved_date:
                content = resolved_content
                if metadata is None:
                    metadata = {}
                metadata["resolved_date"] = resolved_date
        except Exception as _te:
            logger.debug("Temporal resolution skipped: %s", _te)

        # Resolve pronouns using provided entity names (regex-based, no LLM)
        if entities:
            try:
                from engram.recall.pronoun_resolver import resolve_pronouns, has_resolvable_pronouns
                if has_resolvable_pronouns(content):
                    content = resolve_pronouns(content, entities)
            except Exception as _pe:
                logger.debug("Pronoun resolution skipped: %s", _pe)

        # Topic key upsert: update existing memory if same topic_key exists
        if topic_key:
            existing = await self._find_by_topic_key(topic_key)
            if existing:
                return await self._update_topic(existing, content, topic_key, metadata, entities, tags, priority, memory_type)

        # Semantic dedup: merge into existing memory if cosine similarity > threshold
        if self._dedup_enabled:
            dedup_result = await self._dedup_merge(
                content, entities or [], tags or [], priority, memory_type, metadata,
            )
            if dedup_result:
                return dedup_result

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
            await self._detect_embedding_dim()
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
            if self._guard_enabled:
                from engram.ingestion.guard import check_content
                is_safe, reason = check_content(content)
                if not is_safe:
                    logger.warning("Poisoning guard blocked: %s", reason)
                    raise ValueError(f"Content rejected: {reason}")
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
            await self._detect_embedding_dim()
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
            await self._detect_embedding_dim()
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

            # Skip outdated memories from search (still accessible via get_by_id)
            if meta.get("outdated") == "true":
                continue

            # Filter by tags (all requested tags must be present)
            if tags:
                if not all(t in memory.tags for t in tags):
                    continue

            # Compute activation score
            similarity = max(0.0, 1.0 - distance)  # ChromaDB cosine distance → similarity
            score = compute_activation_score(
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
                # Sync FTS5 index
                if self._fts:
                    for mid in batch:
                        self._fts.delete(mid)
            if self._audit:
                self._audit.log_modification(
                    tenant_id=self._namespace, actor="system",
                    mod_type="cleanup_expired", resource_id="",
                    after_value={"deleted_count": len(expired_ids), "ids": expired_ids[:20]},
                    reversible=False, description=f"Cleaned up {len(expired_ids)} expired memories",
                )

        return len(expired_ids)

    async def get_recent(self, n: int = 20) -> list[EpisodicMemory]:
        """Retrieve the most recent N memories sorted by timestamp descending.

        Uses ChromaDB native ordering via metadata timestamp_iso filter where possible.
        Falls back to fetching a larger window when collection is small.
        """
        n = min(n, 1000)  # Hard cap to prevent unbounded fetches
        try:
            collection = self._ensure_collection()
            count = await asyncio.to_thread(collection.count)
            if count == 0:
                return []
            # Fetch enough to cover recent items: cap at min(n*5, count, 2000)
            # This trades a slightly larger fetch for correctness at scale.
            fetch_limit = min(n * 5, count, 2000)
            result = await asyncio.to_thread(
                collection.get,
                include=["documents", "metadatas"],
                limit=fetch_limit,
            )
        except Exception as e:
            raise RuntimeError(f"get_recent failed: {e}") from e

        memories: list[EpisodicMemory] = []
        for i, mem_id in enumerate(result["ids"]):
            doc = result["documents"][i] if result["documents"] else ""
            meta = result["metadatas"][i] if result["metadatas"] else {}
            memories.append(_build_memory(mem_id, doc, meta))

        # Sort by timestamp descending, take top N
        # Use isoformat() to handle mixed offset-naive/offset-aware datetimes
        memories.sort(key=lambda m: m.timestamp.isoformat() if m.timestamp else "", reverse=True)
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
                except Exception as exc:
                    logger.debug("store: audit pre-fetch before delete failed for %s: %s", id, exc)

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
        await self._detect_embedding_dim()
        result: dict[str, Any] = {
            "count": count,
            "collection": self.COLLECTION_NAME,
            "namespace": self._namespace,
        }
        if self._embedding_dim is not None:
            result["embedding_dim"] = self._embedding_dim
        return result

    async def reconcile_stores(self) -> dict[str, int]:
        """Sync FTS5 index with ChromaDB (source of truth). Returns stats dict.

        D-C3: Periodic reconciliation to fix drift between ChromaDB and FTS5
        since writes to both stores are not atomic.
        """
        if not self._fts:
            return {"orphaned_removed": 0, "missing_added": 0}
        collection = self._ensure_collection()
        # Fetch all IDs from both stores (in threads to avoid blocking event loop)
        all_chroma = await asyncio.to_thread(collection.get, include=[])
        chroma_ids = set(all_chroma["ids"])
        fts_ids = set(await asyncio.to_thread(self._fts.get_all_ids))
        # Remove FTS entries not present in ChromaDB
        orphaned_fts = fts_ids - chroma_ids
        for oid in orphaned_fts:
            await asyncio.to_thread(self._fts.delete, oid)
        # Add FTS entries for ChromaDB IDs missing from FTS
        missing_fts = chroma_ids - fts_ids
        if missing_fts:
            batch = await asyncio.to_thread(
                collection.get, list(missing_fts), include=["documents", "metadatas"]
            )
            entries: list[tuple[str, str, str]] = []
            for mid, doc, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
                mt_str = meta.get("memory_type", "fact") if meta else "fact"
                entries.append((mid, doc or "", mt_str))
            if entries:
                await asyncio.to_thread(self._fts.insert_batch, entries)
        return {"orphaned_removed": len(orphaned_fts), "missing_added": len(missing_fts)}

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
                except Exception as exc:
                    logger.debug("store: audit pre-fetch before metadata update failed for %s: %s", mem_id, exc)

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

    async def _detect_embedding_dim(self) -> None:
        """Detect embedding dimension from existing collection data (D-H1: async to avoid blocking)."""
        if self._embedding_dim is not None:
            return
        # Check known dimensions for configured model (fast path via registry)
        known_dim = _detect_embedding_dim_from_model(self._embed_model)
        if known_dim is not None:
            self._embedding_dim = known_dim
            return
        # Peek at existing data to detect dimension (wrapped in thread to avoid blocking event loop)
        collection = self._ensure_collection()
        count = await asyncio.to_thread(collection.count)
        if count > 0:
            peek = await asyncio.to_thread(collection.peek, 1)
            if peek and peek.get("embeddings") and peek["embeddings"][0]:
                self._embedding_dim = len(peek["embeddings"][0])

    async def _find_by_topic_key(self, topic_key: str) -> str | None:
        """Find memory ID by topic_key using ChromaDB where filter."""
        try:
            collection = self._ensure_collection()
            result = await asyncio.to_thread(collection.get, where={"topic_key": topic_key})
            if result["ids"]:
                return result["ids"][0]
        except Exception as exc:
            logger.debug("store: topic lookup failed for key %s: %s", topic_key, exc)
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

        await self._detect_embedding_dim()
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
            except Exception as exc:
                logger.debug("store: audit pre-fetch before topic update failed for %s: %s", mem_id, exc)

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

    async def cleanup_dedup(
        self,
        threshold: float = 0.85,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Retroactively deduplicate existing memories by cosine similarity.

        Paginates through ALL memories, finds near-duplicates above threshold,
        merges entities/tags into the winner (higher priority), and deletes losers.

        Args:
            threshold: Cosine similarity cutoff (0.0-1.0, default 0.85).
            batch_size: Number of memories to load per page.
            dry_run: When True, report what WOULD be merged without any writes.

        Returns:
            {"merged": X, "deleted": Y, "remaining": Z, "dry_run": bool}
        """
        try:
            collection = self._ensure_collection()
            total = await asyncio.to_thread(collection.count)
            if total == 0:
                return {"merged": 0, "deleted": 0, "remaining": 0, "dry_run": dry_run}
        except Exception as e:
            raise RuntimeError(f"cleanup_dedup failed: {e}") from e

        processed: set[str] = set()
        to_delete: list[str] = []
        merged_count = 0

        # Paginate through all memories to fetch IDs + embeddings
        for offset in range(0, total, batch_size):
            try:
                page = await asyncio.to_thread(
                    collection.get,
                    include=["embeddings", "metadatas", "documents"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception as e:
                logger.warning("cleanup_dedup: failed to fetch page offset=%d: %s", offset, e)
                break

            page_ids = page.get("ids", [])
            _raw_emb = page.get("embeddings")
            page_embeddings = _raw_emb if _raw_emb is not None else []
            _raw_meta = page.get("metadatas")
            page_metadatas = _raw_meta if _raw_meta is not None else []

            for i, mem_id in enumerate(page_ids):
                if mem_id in processed:
                    continue

                embedding = page_embeddings[i] if i < len(page_embeddings) else None
                if embedding is None or (hasattr(embedding, '__len__') and len(embedding) == 0):
                    processed.add(mem_id)
                    continue

                # Query top-N similar memories for this embedding
                try:
                    n_query = min(20, total)  # Use cached total instead of per-iteration count()
                    sim_result = await asyncio.to_thread(
                        collection.query,
                        query_embeddings=[embedding],
                        n_results=n_query,
                        include=["metadatas", "distances"],
                    )
                except Exception as e:
                    logger.debug("cleanup_dedup: query failed for %s: %s", mem_id[:8], e)
                    processed.add(mem_id)
                    continue

                if not sim_result["ids"] or not sim_result["ids"][0]:
                    processed.add(mem_id)
                    continue

                winner_meta = page_metadatas[i] if i < len(page_metadatas) else {}
                winner_entities = set(_safe_json_list(winner_meta.get("entities", "[]")))
                winner_tags = set(_safe_json_list(winner_meta.get("tags", "[]")))
                winner_priority = int(winner_meta.get("priority", 5) or 5)
                dups_for_this: list[str] = []

                for j, candidate_id in enumerate(sim_result["ids"][0]):
                    if candidate_id == mem_id or candidate_id in processed:
                        continue
                    distance = sim_result["distances"][0][j]
                    similarity = 1.0 - (distance / 2.0)
                    if similarity < threshold:
                        continue

                    # This candidate is a near-duplicate of mem_id — absorb it
                    cand_meta = sim_result["metadatas"][0][j] if sim_result.get("metadatas") else {}
                    cand_entities = set(_safe_json_list(cand_meta.get("entities", "[]")))
                    cand_tags = set(_safe_json_list(cand_meta.get("tags", "[]")))
                    cand_priority = int(cand_meta.get("priority", 5) or 5)

                    winner_entities |= cand_entities
                    winner_tags |= cand_tags
                    winner_priority = max(winner_priority, cand_priority)
                    dups_for_this.append(candidate_id)
                    processed.add(candidate_id)

                if dups_for_this:
                    merged_count += 1
                    to_delete.extend(dups_for_this)
                    if not dry_run:
                        try:
                            await asyncio.to_thread(
                                collection.update,
                                ids=[mem_id],
                                metadatas=[{
                                    "entities": json.dumps(sorted(winner_entities)),
                                    "tags": json.dumps(sorted(winner_tags)),
                                    "priority": winner_priority,
                                }],
                            )
                        except Exception as e:
                            logger.warning("cleanup_dedup: update winner %s failed: %s", mem_id[:8], e)

                    logger.debug(
                        "cleanup_dedup: %s absorbs %d duplicate(s) (sim>=%.2f)",
                        mem_id[:8], len(dups_for_this), threshold,
                    )

                processed.add(mem_id)

            processed_total = len(processed)
            if processed_total % 100 == 0 or offset + batch_size >= total:
                logger.info(
                    "cleanup_dedup: processed %d/%d memories, %d duplicates found so far",
                    processed_total, total, len(to_delete),
                )

        # Delete all losers
        deleted_count = 0
        if to_delete and not dry_run:
            _BATCH = 500
            for i in range(0, len(to_delete), _BATCH):
                batch = to_delete[i:i + _BATCH]
                try:
                    await asyncio.to_thread(collection.delete, ids=batch)
                    deleted_count += len(batch)
                except Exception as e:
                    logger.warning("cleanup_dedup: delete batch failed: %s", e)
                # Remove from FTS index
                for dup_id in batch:
                    try:
                        await asyncio.to_thread(self._fts.delete, dup_id)
                    except Exception as exc:
                        logger.debug("store: FTS dedup delete failed for %s: %s", dup_id, exc)
        elif dry_run:
            deleted_count = len(to_delete)

        remaining = await asyncio.to_thread(collection.count)

        if not dry_run and self._audit and to_delete:
            self._audit.log_modification(
                tenant_id=self._namespace, actor="system",
                mod_type="cleanup_dedup", resource_id="",
                after_value={"merged_groups": merged_count, "deleted": deleted_count},
                reversible=False,
                description=f"Retroactive dedup: merged {merged_count} groups, deleted {deleted_count} duplicates",
            )

        logger.info(
            "cleanup_dedup complete: merged=%d deleted=%d remaining=%d dry_run=%s",
            merged_count, deleted_count, remaining, dry_run,
        )
        return {"merged": merged_count, "deleted": deleted_count, "remaining": remaining, "dry_run": dry_run}

    async def _dedup_merge(
        self,
        content: str,
        entities: list[str],
        tags: list[str],
        priority: int,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Check if a semantically similar memory exists. If so, merge entities/tags.

        ChromaDB cosine distance: 0.0 = identical, 2.0 = opposite.
        Similarity = 1 - (distance / 2).  Threshold of 0.85 → distance < 0.30.

        Returns existing memory ID if merged, None if no duplicate found.
        """
        try:
            collection = self._ensure_collection()
            coll_count = await asyncio.to_thread(collection.count)
            if coll_count == 0:
                return None

            await self._detect_embedding_dim()
            embedding = await asyncio.to_thread(
                _get_embeddings, self._embed_model, [content], self._embedding_dim,
            )
            result = await asyncio.to_thread(
                collection.query,
                query_embeddings=embedding,
                n_results=1,
                include=["metadatas", "documents", "distances"],
            )
            if not result["ids"] or not result["ids"][0]:
                return None

            distance = result["distances"][0][0]
            similarity = 1.0 - (distance / 2.0)

            if similarity < self._dedup_threshold:
                return None

            # Found a near-duplicate — merge entities and tags into existing
            existing_id = result["ids"][0][0]
            existing_meta = result["metadatas"][0][0]
            existing_content = result["documents"][0][0] if result["documents"] else ""

            # Merge entities
            old_entities = set(json.loads(existing_meta.get("entities", "[]")))
            merged_entities = sorted(old_entities | set(entities))

            # Merge tags
            old_tags = set(json.loads(existing_meta.get("tags", "[]")))
            merged_tags = sorted(old_tags | set(tags))

            # Keep higher priority
            old_priority = int(existing_meta.get("priority", 5))
            merged_priority = max(old_priority, priority)

            update_meta: dict[str, Any] = {
                "entities": json.dumps(merged_entities),
                "tags": json.dumps(merged_tags),
                "priority": merged_priority,
            }
            if metadata:
                update_meta.update(metadata)

            await asyncio.to_thread(collection.update, ids=[existing_id], metadatas=[update_meta])

            logger.info(
                "Dedup merged into %s (sim=%.2f): +%d entities, +%d tags",
                existing_id[:8], similarity,
                len(set(entities) - old_entities),
                len(set(tags) - old_tags),
            )

            if self._audit:
                self._audit.log(
                    tenant_id=self._namespace, actor="system",
                    operation="episodic.dedup_merge",
                    resource_id=existing_id,
                    details={
                        "similarity": round(similarity, 3),
                        "new_content": content[:200],
                        "existing_content": existing_content[:200],
                    },
                )
            return existing_id
        except Exception as e:
            # Fail-open: if dedup check fails, let remember() create a new memory
            logger.debug("Dedup check failed (proceeding with new memory): %s", e)
            return None


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
