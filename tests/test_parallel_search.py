"""Tests for parallel search with fusion."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from engram.recall.parallel_search import ParallelSearcher, _memory_score
from engram.config import RecallPipelineConfig
from engram.models import (
    EpisodicMemory,
    MemoryType,
    ResolvedText,
    Entity,
    SearchResult,
    SemanticNode,
    SemanticEdge,
)


def _make_episodic(content: str, priority: int = 5) -> EpisodicMemory:
    return EpisodicMemory(
        id=f"mem-{hash(content) % 10000}",
        content=content,
        memory_type=MemoryType.FACT,
        priority=priority,
        timestamp=datetime.now(timezone.utc),
    )


def _make_mock_stores(
    search_results: list[EpisodicMemory] | None = None,
    related_results: dict | None = None,
    query_results: list[SemanticNode] | None = None,
):
    """Create mock episodic + semantic stores."""
    episodic = AsyncMock()
    episodic.search = AsyncMock(return_value=search_results or [])

    semantic = AsyncMock()
    semantic.get_related = AsyncMock(return_value=related_results or {})
    semantic.query = AsyncMock(return_value=query_results or [])

    return episodic, semantic


class TestParallelSearcher:
    """Test parallel search and fusion."""

    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(self):
        memories = [_make_episodic("deployed API v2", priority=8)]
        episodic, semantic = _make_mock_stores(search_results=memories)
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("deployment")
        assert len(results) >= 1
        assert results[0].content == "deployed API v2"
        assert results[0].source == "semantic"

    @pytest.mark.asyncio
    async def test_entity_graph_search_with_entities(self):
        node = SemanticNode(type="Person", name="Trâm")
        edge = SemanticEdge(from_node="Person:Trâm", to_node="Job:VLTL", relation="works_as")
        related = {
            "Trâm": {
                "nodes": [node],
                "edges": [edge],
            }
        }
        episodic, semantic = _make_mock_stores(related_results=related)
        searcher = ParallelSearcher(episodic, semantic)

        resolved = ResolvedText(
            original="cô ấy",
            resolved="Trâm",
            entities=[Entity(name="Trâm", type="person")],
        )
        results = await searcher.search("job", resolved=resolved)
        # Should include entity graph results
        sources = {r.source for r in results}
        assert "entity_graph" in sources

    @pytest.mark.asyncio
    async def test_empty_entities_skips_graph_search(self):
        episodic, semantic = _make_mock_stores()
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("something")
        semantic.get_related.assert_not_called()

    @pytest.mark.asyncio
    async def test_fusion_deduplicates(self):
        # Two identical content from different sources
        memories = [
            _make_episodic("PostgreSQL is the main database", priority=7),
            _make_episodic("PostgreSQL is the main database", priority=5),
        ]
        episodic, semantic = _make_mock_stores(search_results=memories)
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("database")
        # Should deduplicate to 1 result
        contents = [r.content for r in results]
        assert contents.count("PostgreSQL is the main database") == 1

    @pytest.mark.asyncio
    async def test_fusion_keeps_highest_score(self):
        memories = [
            _make_episodic("Redis for caching", priority=9),
            _make_episodic("Redis for caching", priority=3),
        ]
        episodic, semantic = _make_mock_stores(search_results=memories)
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("caching")
        redis_result = [r for r in results if "Redis" in r.content]
        assert len(redis_result) == 1
        assert redis_result[0].importance == 9  # kept the higher priority one

    @pytest.mark.asyncio
    async def test_fusion_respects_limit(self):
        memories = [_make_episodic(f"memory {i}", priority=5) for i in range(20)]
        episodic, semantic = _make_mock_stores(search_results=memories)
        config = RecallPipelineConfig(fusion_top_k=5)
        searcher = ParallelSearcher(episodic, semantic, config)

        results = await searcher.search("test")
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_fallback_triggers_on_low_scores(self):
        # No semantic results → low score → triggers keyword fallback
        query_nodes = [SemanticNode(type="Technology", name="PostgreSQL")]
        episodic, semantic = _make_mock_stores(query_results=query_nodes)
        config = RecallPipelineConfig(fallback_threshold=0.5)
        searcher = ParallelSearcher(episodic, semantic, config)

        results = await searcher.search("PostgreSQL")
        semantic.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_one_source_doesnt_crash(self):
        episodic = AsyncMock()
        episodic.search = AsyncMock(side_effect=Exception("ChromaDB down"))
        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[])

        searcher = ParallelSearcher(episodic, semantic)
        # Should not raise
        results = await searcher.search("test")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_custom_limit(self):
        memories = [_make_episodic(f"mem {i}") for i in range(10)]
        episodic, semantic = _make_mock_stores(search_results=memories)
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("test", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(self):
        memories = [
            _make_episodic("low priority", priority=2),
            _make_episodic("high priority", priority=9),
            _make_episodic("medium priority", priority=5),
        ]
        episodic, semantic = _make_mock_stores(search_results=memories)
        searcher = ParallelSearcher(episodic, semantic)

        results = await searcher.search("test")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestMemoryScore:
    """Test memory scoring function."""

    def test_high_priority_higher_score(self):
        high = _make_episodic("important", priority=9)
        low = _make_episodic("trivial", priority=2)
        assert _memory_score(high) > _memory_score(low)

    def test_score_in_valid_range(self):
        mem = _make_episodic("test", priority=5)
        score = _memory_score(mem)
        assert 0.0 <= score <= 1.0
