"""CRUD mixin for EpisodicStore: remember, get, delete, update_metadata, topic upsert, dedup merge.

Depends on: episodic_builder helpers, fts_sync helpers.
All methods rely on self._* attributes resolved at runtime via MRO.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import sys as _sys

from engram.episodic.episodic_builder import (
    _build_memory,
    _canonicalize_entities,
)


def _get_embeddings(*args, **kwargs):
    """Proxy to engram.episodic.store._get_embeddings so test patches on that name work."""
    return _sys.modules["engram.episodic.store"]._get_embeddings(*args, **kwargs)
from engram.episodic.fts_sync import fts_delete, fts_insert
from engram.hooks import fire_hook
from engram.models import EpisodicMemory, MemoryType
from engram.sanitize import sanitize_content

logger = logging.getLogger("engram")


class _EpisodicCrudMixin:
    """Mixin providing remember(), get(), delete(), update_metadata(), and internal helpers."""

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
        timestamp: datetime | None = None,
        source: str = "",
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

        # Resolve pronouns via unified resolver (regex → LLM fallback if needed).
        # store.py only has entity names (no conversation history), so we build
        # a minimal synthetic context so entity_resolver can extract names.
        # Without conversation history the LLM fallback is skipped automatically.
        if entities:
            try:
                from engram.recall.entity_resolver import resolve as _entity_resolve
                # Build synthetic context messages so _extract_entity_names_from_context
                # can pick up the entity names we already have.
                synthetic_ctx = [{"role": "assistant", "content": " ".join(entities)}]
                _resolved = await _entity_resolve(
                    content,
                    context=synthetic_ctx,
                    resolve_temporal_refs=False,
                    resolve_pronoun_refs=True,
                )
                content = _resolved.resolved
            except Exception as _pe:
                logger.debug("Pronoun resolution skipped: %s", _pe)

        # Topic key upsert: update existing memory if same topic_key exists
        if topic_key:
            existing = await self._find_by_topic_key(topic_key)
            if existing:
                return await self._update_topic(
                    existing, content, topic_key, metadata, entities, tags, priority, memory_type
                )

        # Semantic dedup: merge into existing memory if cosine similarity > threshold
        if self._dedup_enabled:
            dedup_result = await self._dedup_merge(
                content, entities or [], tags or [], priority, memory_type, metadata,
            )
            if dedup_result:
                return dedup_result

        memory_id = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc)
        timestamp_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        entities = _canonicalize_entities(entities)
        tags = tags or []

        doc_metadata: dict[str, Any] = {
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "priority": priority,
            "timestamp": timestamp_str,
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
        if source:
            doc_metadata["source"] = source
        if metadata:
            doc_metadata.update(metadata)

        try:
            await self._ensure_backend()
            await self._detect_embedding_dim()
            embeddings = await asyncio.to_thread(
                _get_embeddings, self._embed_model, [content], self._embedding_dim
            )
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d. "
                    "Use a consistent embedding model or reset the collection."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            await self._backend.add(
                id=memory_id,
                embedding=embeddings[0],
                content=content,
                metadata=doc_metadata,
            )
        except ValueError:
            # Re-raise validation errors (dimension mismatch, etc.) — not retriable
            raise
        except Exception as e:
            # Embedding or backend failure — enqueue for retry instead of dropping
            logger.warning("Embedding failed, queued for retry: %s", content[:50])
            try:
                from engram.episodic.pending_queue import get_pending_queue
                get_pending_queue().enqueue(
                    content=content,
                    timestamp=doc_metadata.get("timestamp"),
                    metadata={k: v for k, v in doc_metadata.items() if k != "timestamp"},
                )
            except Exception as qe:
                logger.error("PendingQueue enqueue also failed: %s", qe)
            raise RuntimeError(f"Failed to store memory: {e}") from e

        # Sync to FTS5 index (fire-and-forget; non-blocking on failure)
        mt_str = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        await fts_insert(self._fts, memory_id, content, mt_str)

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

    async def get(self, id: str) -> EpisodicMemory | None:
        """Retrieve a single memory by ID."""
        try:
            await self._ensure_backend()
            item = await self._backend.get(id)
        except Exception:
            return None

        if item is None:
            return None

        return _build_memory(item["id"], item["document"], item["metadata"])

    async def delete(self, id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        try:
            await self._ensure_backend()
            # Capture before-value for audit trail
            before_content = None
            if self._audit and self._audit.enabled:
                try:
                    result = await self._backend.get_many(
                        ids=[id], include=["documents", "metadatas"]
                    )
                    if result["ids"]:
                        before_content = result["documents"][0] if result["documents"] else ""
                except Exception as exc:
                    logger.debug("store: audit pre-fetch before delete failed for %s: %s", id, exc)

            await self._backend.delete(ids=[id])

            # Remove from FTS5 index
            await fts_delete(self._fts, id)

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

    async def update_metadata(self, mem_id: str, metadata: dict[str, Any]) -> bool:
        """Update metadata fields on an existing memory. Returns True on success."""
        try:
            await self._ensure_backend()

            # Canonicalize entities on every write path to prevent duplicates
            # like Engram/engram from being persisted again.
            if "entities" in metadata:
                raw = metadata.get("entities")
                parsed: list[str]
                if isinstance(raw, str):
                    if raw.startswith("["):
                        try:
                            parsed = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            parsed = []
                    else:
                        parsed = [e.strip() for e in raw.split(",") if e.strip()]
                elif isinstance(raw, list):
                    parsed = raw
                else:
                    parsed = []
                metadata["entities"] = json.dumps(_canonicalize_entities(parsed))

            # Capture before-value for audit trail
            before_meta = None
            if self._audit and self._audit.enabled:
                try:
                    existing = await self._backend.get_many(
                        ids=[mem_id], include=["metadatas"]
                    )
                    if existing["metadatas"]:
                        before_meta = {k: existing["metadatas"][0].get(k) for k in metadata}
                except Exception as exc:
                    logger.debug(
                        "store: audit pre-fetch before metadata update failed for %s: %s", mem_id, exc
                    )

            await self._backend.update(ids=[mem_id], metadatas=[metadata])

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

    async def _find_by_topic_key(self, topic_key: str) -> str | None:
        """Find memory ID by topic_key using ChromaDB where filter."""
        try:
            await self._ensure_backend()
            result = await self._backend.get_many(where={"topic_key": topic_key})
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
        await self._ensure_backend()
        existing = await self._backend.get_many(ids=[mem_id], include=["metadatas"])
        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        revision = int(old_meta.get("revision_count", 0)) + 1

        await self._detect_embedding_dim()
        embeddings = await asyncio.to_thread(
            _get_embeddings, self._embed_model, [content], self._embedding_dim
        )

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
                old_doc = await self._backend.get_many(ids=[mem_id], include=["documents"])
                if old_doc["documents"]:
                    old_content = old_doc["documents"][0]
            except Exception as exc:
                logger.debug(
                    "store: audit pre-fetch before topic update failed for %s: %s", mem_id, exc
                )

        await self._backend.update_with_embeddings(
            ids=[mem_id],
            documents=[content],
            embeddings=embeddings,
            metadatas=[new_meta],
        )

        # Sync updated content to FTS5 index
        mt_str = new_meta.get("memory_type", "fact")
        await fts_insert(self._fts, mem_id, content, mt_str)

        if self._audit:
            self._audit.log_modification(
                tenant_id=self._namespace, actor="system",
                mod_type="memory_update", resource_id=mem_id,
                before_value=old_content, after_value=content,
                description=f"Topic-key upsert (revision={revision})",
            )
        logger.info("Updated topic-keyed memory %s (revision=%d)", mem_id[:8], revision)
        return mem_id

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
            await self._ensure_backend()
            coll_count = await self._backend.count()
            if coll_count == 0:
                return None

            await self._detect_embedding_dim()
            embedding = await asyncio.to_thread(
                _get_embeddings, self._embed_model, [content], self._embedding_dim,
            )
            result = await self._backend.query(
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

            # Merge entities (canonicalized, case-insensitive dedupe)
            old_entities_list = _canonicalize_entities(
                json.loads(existing_meta.get("entities", "[]"))
            )
            merged_entities = _canonicalize_entities(old_entities_list + (entities or []))
            old_entities = set(old_entities_list)

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

            await self._backend.update(ids=[existing_id], metadatas=[update_meta])

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
