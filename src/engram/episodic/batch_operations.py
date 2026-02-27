"""Batch operations mixin for EpisodicStore: remember_batch.

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

from engram.episodic.episodic_builder import _canonicalize_entities


def _get_embeddings(*args, **kwargs):
    """Proxy to engram.episodic.store._get_embeddings so test patches on that name work."""
    return _sys.modules["engram.episodic.store"]._get_embeddings(*args, **kwargs)
from engram.episodic.fts_sync import fts_insert_batch
from engram.models import MemoryType
from engram.sanitize import sanitize_content

logger = logging.getLogger("engram")


class _BatchMixin:
    """Mixin providing remember_batch() for EpisodicStore."""

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
            # Sanitize content just like single remember() does
            content = sanitize_content(mem["content"])
            if self._guard_enabled:
                from engram.ingestion.guard import check_content
                is_safe, reason = check_content(content)
                if not is_safe:
                    logger.warning("Poisoning guard blocked: %s", reason)
                    raise ValueError(f"Content rejected: {reason}")
            memory_type = mem.get("memory_type", MemoryType.FACT)
            priority = mem.get("priority", 5)
            entities = _canonicalize_entities(mem.get("entities") or [])
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
            await self._ensure_backend()
            await self._detect_embedding_dim()
            # Single batch embedding call
            embeddings = await asyncio.to_thread(
                _get_embeddings, self._embed_model, documents, self._embedding_dim
            )
            new_dim = len(embeddings[0])
            if self._embedding_dim is not None and new_dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {new_dim}d but collection uses {self._embedding_dim}d."
                )
            self._embedding_dim = self._embedding_dim or new_dim
            # Single upsert call for all memories
            await self._backend.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to batch store memories: {e}") from e

        # Sync to FTS5 index
        fts_entries = [
            (ids[i], documents[i], metadatas[i].get("memory_type", "fact"))
            for i in range(len(ids))
        ]
        await fts_insert_batch(self._fts, fts_entries)

        return ids
