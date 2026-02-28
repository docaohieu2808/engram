"""Parallel search with fusion — multi-source memory retrieval.

Runs parallel searches across episodic (ChromaDB), entity graph (NetworkX),
and keyword (graph query) sources, then fuses results by deduplication and scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from engram.config import RecallConfig, RecallPipelineConfig, ScoringConfig
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
        scoring: ScoringConfig | None = None,
        recall: RecallConfig | None = None,
    ):
        self._episodic = episodic
        self._semantic = semantic
        self._config = config or RecallPipelineConfig()
        self._scoring = scoring
        self._recall = recall or RecallConfig()

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
        """ChromaDB vector similarity search. Excludes consolidated originals."""
        memories = await self._episodic.search(query, limit=limit)
        results = []
        for m in memories:
            # I-H10: skip memories that have been consolidated into a summary
            if m.consolidated_into:
                continue
            results.append(SearchResult(
                id=m.id,
                content=m.content,
                score=_memory_score(m, self._scoring),
                source="semantic",
                memory_type=m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                importance=m.priority,
                timestamp=m.timestamp,
                metadata=m.metadata,
            ))
        return results

    async def _search_entity_graph(self, entity_names: list[str]) -> list[SearchResult]:
        """Traverse semantic graph for entity-related information."""
        if not entity_names:
            return []
        related = await self._semantic.get_related(entity_names, depth=self._recall.entity_graph_depth)
        results: list[SearchResult] = []
        for name, data in related.items():
            for node in data.get("nodes", []):
                attrs_str = f" {node.attributes}" if node.attributes else ""
                results.append(SearchResult(
                    id=node.key,
                    content=f"{node.type}: {node.name}{attrs_str}",
                    score=self._recall.semantic_edge_score,
                    source="entity_graph",
                ))
            for edge in data.get("edges", []):
                results.append(SearchResult(
                    id=edge.key,
                    content=f"{edge.from_node} --{edge.relation}--> {edge.to_node}",
                    score=self._recall.entity_co_mention_score,
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
                score=self._recall.keyword_exact_match_score,
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
                score=self._recall.fuzzy_match_score,
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


def _memory_score(memory: Any, scoring: ScoringConfig | None = None) -> float:
    """Compute a normalized score from an EpisodicMemory for fusion ranking.

    When scoring is None, uses default weights (60% similarity, 40% priority).
    When scoring is provided, uses ScoringConfig.similarity_weight and
    ScoringConfig.retention_weight, normalized so they sum to 1.0.

    Similarity is proxied via confidence (0-1) which reflects user feedback
    and store-scored quality; priority is normalized to 0-1 as retention proxy.
    """
    priority_score = memory.priority / 10.0
    # confidence (0-1) reflects user feedback and store-scored similarity quality
    similarity_score = getattr(memory, "confidence", 0.5)

    if scoring is not None:
        # Normalize similarity_weight + retention_weight so they sum to 1.0
        w_sim = scoring.similarity_weight
        w_ret = scoring.retention_weight
        total = w_sim + w_ret
        if total > 0:
            w_sim = w_sim / total
            w_ret = w_ret / total
        else:
            w_sim, w_ret = 0.6, 0.4
        score = w_sim * similarity_score + w_ret * priority_score
    else:
        # Default: 60% similarity, 40% priority
        score = 0.6 * similarity_score + 0.4 * priority_score

    return max(0.1, score)
