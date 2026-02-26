"""Parallel search with fusion — multi-source memory retrieval.

Runs parallel searches across episodic (ChromaDB), entity graph (NetworkX),
and keyword (graph query) sources, then fuses results by deduplication and scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from engram.config import RecallPipelineConfig
from engram.models import ResolvedText, SearchResult

if TYPE_CHECKING:
    from engram.episodic.store import EpisodicStore
    from engram.semantic.graph import SemanticGraph

logger = logging.getLogger("engram")


class ParallelSearcher:
    """Execute parallel searches across multiple memory stores and fuse results."""

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticGraph,
        config: RecallPipelineConfig | None = None,
    ):
        self._episodic = episodic
        self._semantic = semantic
        self._config = config or RecallPipelineConfig()

    async def search(
        self,
        query: str,
        resolved: ResolvedText | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Run parallel searches and fuse results.

        Args:
            query: Original search query.
            resolved: Optional resolved text with entities/temporal refs.
            limit: Max results to return (defaults to config.fusion_top_k).
        """
        top_k = limit or self._config.fusion_top_k
        entity_names = [e.name for e in resolved.entities] if resolved else []
        search_query = resolved.resolved if resolved else query

        # Run semantic + entity graph + FTS5 searches in parallel
        tasks: list[Any] = [
            self._search_semantic(search_query, top_k),
            self._search_entity_graph(entity_names),
            self._search_fts(search_query, top_k),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten, filtering out exceptions
        all_results: list[SearchResult] = []
        for r in results:
            if isinstance(r, list):
                all_results.extend(r)
            elif isinstance(r, Exception):
                logger.debug("Search source failed: %s", r)

        # Fallback: if best score < threshold, add keyword search
        best_score = max((r.score for r in all_results), default=0.0)
        if best_score < self._config.fallback_threshold:
            try:
                keyword_results = await self._search_keyword(search_query, top_k)
                all_results.extend(keyword_results)
            except Exception as e:
                logger.debug("Keyword search fallback failed: %s", e)

        return self._fuse(all_results, top_k)

    async def _search_semantic(self, query: str, limit: int) -> list[SearchResult]:
        """ChromaDB vector similarity search."""
        memories = await self._episodic.search(query, limit=limit)
        return [
            SearchResult(
                id=m.id,
                content=m.content,
                score=_memory_score(m),
                source="semantic",
                memory_type=m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                importance=m.priority,
                timestamp=m.timestamp,
                metadata=m.metadata,
            )
            for m in memories
        ]

    async def _search_entity_graph(self, entity_names: list[str]) -> list[SearchResult]:
        """Traverse semantic graph for entity-related information."""
        if not entity_names:
            return []
        related = await self._semantic.get_related(entity_names, depth=2)
        results: list[SearchResult] = []
        for name, data in related.items():
            for node in data.get("nodes", []):
                attrs_str = f" {node.attributes}" if node.attributes else ""
                results.append(SearchResult(
                    id=node.key,
                    content=f"{node.type}: {node.name}{attrs_str}",
                    score=0.5,
                    source="entity_graph",
                ))
            for edge in data.get("edges", []):
                results.append(SearchResult(
                    id=edge.key,
                    content=f"{edge.from_node} --{edge.relation}--> {edge.to_node}",
                    score=0.4,
                    source="entity_graph",
                ))
        return results

    async def _search_fts(self, query: str, limit: int) -> list[SearchResult]:
        """FTS5 exact keyword search via EpisodicStore.search_fulltext."""
        if not hasattr(self._episodic, "search_fulltext"):
            return []
        memories = await self._episodic.search_fulltext(query, limit=limit)
        return [
            SearchResult(
                id=m.id,
                content=m.content,
                score=0.6,  # Fixed score; FTS5 confirms exact keyword match
                source="fts",
                memory_type=m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                importance=m.priority,
                timestamp=m.timestamp,
            )
            for m in memories
        ]

    async def _search_keyword(self, query: str, limit: int) -> list[SearchResult]:
        """Keyword-based search via semantic graph query."""
        nodes = await self._semantic.query(query)
        return [
            SearchResult(
                id=n.key,
                content=f"{n.type}: {n.name}",
                score=0.3,
                source="keyword",
            )
            for n in nodes[:limit]
        ]

    def _fuse(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        """Deduplicate by content hash, keep highest score, return top K."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            key = hashlib.sha256(r.content.encode()).hexdigest()
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        fused = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return fused[:limit]


def _memory_score(memory: Any) -> float:
    """Compute a normalized score from an EpisodicMemory for fusion ranking."""
    # Priority normalized to 0-1 range, weighted with recency
    priority_score = memory.priority / 10.0
    return max(0.1, priority_score)
