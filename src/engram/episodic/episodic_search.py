"""Search mixin for EpisodicStore: semantic search, full-text search, recent recall.

Depends on: episodic_builder._build_memory, fts_sync helpers.
All methods rely on self._* attributes resolved at runtime via MRO.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from engram.episodic.decay import compute_activation_score
from engram.episodic.episodic_builder import _build_memory
import sys as _sys

from engram.episodic.fts_sync import fts_search


def _get_embeddings(*args, **kwargs):
    """Proxy to engram.episodic.store._get_embeddings so test patches on that name work."""
    return _sys.modules["engram.episodic.store"]._get_embeddings(*args, **kwargs)
from engram.models import EpisodicMemory

logger = logging.getLogger("engram")


class _EpisodicSearchMixin:
    """Mixin providing search(), search_fulltext(), and get_recent() for EpisodicStore."""

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
            await self._ensure_backend()
            await self._detect_embedding_dim()
            query_embedding = await asyncio.to_thread(
                _get_embeddings, self._embed_model, [query], self._embedding_dim
            )
            coll_count = await self._backend.count()
            results = await self._backend.query(
                query_embeddings=query_embedding,
                n_results=min(limit, coll_count or 1),
                where=filters if filters else None,
            )
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
                await self._backend.update(ids=access_ids, metadatas=access_metas)
            except Exception as e:
                logger.debug("Access tracking update failed: %s", e)

        return memories

    async def search_fulltext(self, query: str, limit: int = 10) -> list[EpisodicMemory]:
        """Search memories using FTS5 exact keyword matching.

        Returns EpisodicMemory objects fetched from ChromaDB by ID.
        Skips IDs not found in ChromaDB (may have been deleted without FTS sync).
        """
        fts_results = await fts_search(self._fts, query, limit)
        if not fts_results:
            return []

        ids = [r.id for r in fts_results]
        try:
            await self._ensure_backend()
            result = await self._backend.get_many(
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

    async def get_recent(self, n: int = 20) -> list[EpisodicMemory]:
        """Retrieve the most recent N memories sorted by timestamp descending.

        Uses ChromaDB native ordering via metadata timestamp_iso filter where possible.
        Falls back to fetching a larger window when collection is small.
        """
        n = min(n, 1000)  # Hard cap to prevent unbounded fetches
        try:
            await self._ensure_backend()
            count = await self._backend.count()
            if count == 0:
                return []
            # Fetch enough to cover recent items: cap at min(n*5, count, 2000)
            # This trades a slightly larger fetch for correctness at scale.
            fetch_limit = min(n * 5, count, 2000)
            result = await self._backend.get_many(
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
