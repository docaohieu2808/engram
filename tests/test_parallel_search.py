"""Tests for parallel search with fusion."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from engram.recall.parallel_search import ParallelSearcher, _memory_score
from engram.config import RecallPipelineConfig, ScoringConfig
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

    def test_score_zero_priority(self):
        """Minimum meaningful inputs → floor kicks in, score >= 0.1."""
        # EpisodicMemory validates priority 1-10; use MagicMock to bypass
        mem = MagicMock()
        mem.priority = 1
        mem.confidence = 0.0
        score = _memory_score(mem)
        # 0.6*0.0 + 0.4*(1/10) = 0.04 → floor returns 0.1
        assert score >= 0.1

    def test_score_max_values(self):
        """priority=10, confidence=1.0 → score = 1.0."""
        mem = _make_episodic("max", priority=10)
        mem.confidence = 1.0
        score = _memory_score(mem)
        assert score == pytest.approx(1.0)

    def test_custom_scoring_config_changes_score(self):
        """Custom ScoringConfig weights produce different score than defaults."""
        mem = MagicMock()
        mem.priority = 8   # priority_score = 0.8
        mem.confidence = 0.2  # similarity_score = 0.2

        # Default: 0.6*0.2 + 0.4*0.8 = 0.12 + 0.32 = 0.44
        default_score = _memory_score(mem)
        assert default_score == pytest.approx(0.44)

        # Custom: similarity_weight=0.1, retention_weight=0.9 → normalized 1/10 and 9/10
        # score = 0.1*0.2 + 0.9*0.8 = 0.02 + 0.72 = 0.74
        scoring = ScoringConfig(similarity_weight=0.1, retention_weight=0.9)
        custom_score = _memory_score(mem, scoring)
        assert custom_score == pytest.approx(0.74)
        assert custom_score != pytest.approx(default_score)

    def test_scoring_config_none_uses_defaults(self):
        """Passing scoring=None behaves identically to no scoring arg."""
        mem = MagicMock()
        mem.priority = 5
        mem.confidence = 0.5
        assert _memory_score(mem, None) == pytest.approx(_memory_score(mem))

    def test_scoring_config_equal_weights_normalized(self):
        """Equal similarity+retention weights → each normalized to 0.5."""
        mem = MagicMock()
        mem.priority = 6   # priority_score = 0.6
        mem.confidence = 0.4  # similarity_score = 0.4
        # Both weights equal → normalized 0.5 each → 0.5*0.4 + 0.5*0.6 = 0.5
        scoring = ScoringConfig(similarity_weight=0.3, retention_weight=0.3)
        score = _memory_score(mem, scoring)
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_parallel_searcher_passes_scoring_to_score(self):
        """ParallelSearcher with ScoringConfig produces scores consistent with config."""
        mem = MagicMock()
        mem.id = "test-id"
        mem.content = "test content"
        mem.memory_type = MagicMock()
        mem.memory_type.value = "fact"
        mem.priority = 8
        mem.timestamp = datetime.now(timezone.utc)
        mem.metadata = {}
        mem.confidence = 0.2
        mem.consolidated_into = None

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[mem])
        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[])

        # High retention weight → score dominated by priority (0.8)
        scoring = ScoringConfig(similarity_weight=0.1, retention_weight=0.9)
        searcher = ParallelSearcher(episodic, semantic, scoring=scoring)
        results = await searcher.search("test")
        assert len(results) >= 1
        # Expected: normalized 1/10 sim + 9/10 ret = 0.1*0.2 + 0.9*0.8 = 0.74
        assert results[0].score == pytest.approx(0.74)


class TestLexicalFallback:
    """Tests for keyword fallback path triggered when best fusion score < fallback_threshold."""

    @pytest.mark.asyncio
    async def test_fallback_triggers_below_threshold(self):
        """When best semantic score < fallback_threshold, keyword search is invoked."""
        # confidence=0.0, priority=1 → _memory_score = max(0.1, 0.6*0+0.4*0.1) = 0.1
        # 0.1 < fallback_threshold=0.3 → keyword fallback must fire
        mem = MagicMock()
        mem.id = "low-score-mem"
        mem.content = "some low quality memory"
        mem.memory_type = MagicMock()
        mem.memory_type.value = "fact"
        mem.priority = 1
        mem.timestamp = datetime.now(timezone.utc)
        mem.metadata = {}
        mem.confidence = 0.0
        mem.consolidated_into = None

        keyword_node = SemanticNode(type="Topic", name="fallback result")

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[mem])
        # Disable FTS so it doesn't add a 0.6 score result that would skip fallback
        del episodic.search_fulltext

        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[keyword_node])

        config = RecallPipelineConfig(fallback_threshold=0.3)
        searcher = ParallelSearcher(episodic, semantic, config)

        results = await searcher.search("test query")

        # Keyword fallback must have been called
        semantic.query.assert_called_once()
        # Keyword result appears in final output
        sources = {r.source for r in results}
        assert "keyword" in sources

    @pytest.mark.asyncio
    async def test_fallback_skipped_above_threshold(self):
        """When best semantic score >= fallback_threshold, keyword search is NOT called."""
        # confidence=0.5, priority=5 → _memory_score = 0.6*0.5 + 0.4*0.5 = 0.5
        # 0.5 >= fallback_threshold=0.3 → keyword fallback must NOT fire
        mem = MagicMock()
        mem.id = "good-score-mem"
        mem.content = "high quality memory above threshold"
        mem.memory_type = MagicMock()
        mem.memory_type.value = "fact"
        mem.priority = 5
        mem.timestamp = datetime.now(timezone.utc)
        mem.metadata = {}
        mem.confidence = 0.5
        mem.consolidated_into = None

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[mem])
        del episodic.search_fulltext

        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[])

        config = RecallPipelineConfig(fallback_threshold=0.3)
        searcher = ParallelSearcher(episodic, semantic, config)

        results = await searcher.search("test query")

        # Keyword fallback must NOT have been called
        semantic.query.assert_not_called()
        # Semantic result still present
        assert any(r.source == "semantic" for r in results)

    @pytest.mark.asyncio
    async def test_fallback_results_fused_correctly(self):
        """Keyword fallback results are deduped and sorted with existing results."""
        # Low-score semantic result triggers fallback
        mem = MagicMock()
        mem.id = "mem-001"
        mem.content = "unique semantic content"
        mem.memory_type = MagicMock()
        mem.memory_type.value = "fact"
        mem.priority = 1
        mem.timestamp = datetime.now(timezone.utc)
        mem.metadata = {}
        mem.confidence = 0.0
        mem.consolidated_into = None

        # Two keyword nodes — one duplicates semantic content, one is unique
        dup_node = SemanticNode(type="Topic", name="unique semantic content")  # same as mem.content
        unique_node = SemanticNode(type="Topic", name="unique semantic content")  # duplicate of dup_node
        fresh_node = SemanticNode(type="Entity", name="fresh keyword result")

        episodic = AsyncMock()
        episodic.search = AsyncMock(return_value=[mem])
        del episodic.search_fulltext

        semantic = AsyncMock()
        semantic.get_related = AsyncMock(return_value={})
        semantic.query = AsyncMock(return_value=[dup_node, fresh_node])

        config = RecallPipelineConfig(fallback_threshold=0.3)
        searcher = ParallelSearcher(episodic, semantic, config)

        results = await searcher.search("test query")

        # Results sorted descending by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Dedup: "Topic: unique semantic content" appears only once
        keyword_contents = [r.content for r in results if r.source == "keyword"]
        assert keyword_contents.count("Topic: unique semantic content") <= 1

        # Both unique keyword result and semantic result present
        all_contents = [r.content for r in results]
        assert any("fresh keyword result" in c for c in all_contents)


def _make_search_result(content: str, score: float, id: str | None = None) -> SearchResult:
    return SearchResult(
        id=id or f"sr-{hash(content) % 10000}",
        content=content,
        score=score,
        source="semantic",
    )


class TestFuse:
    """Direct unit tests for ParallelSearcher._fuse()."""

    def _fuse(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        # _fuse uses no instance state — call via None self
        return ParallelSearcher._fuse(None, results, limit)  # type: ignore[arg-type]

    def test_fuse_dedup_keeps_highest_score(self):
        """Two results with identical content → only one kept with higher score."""
        r_low = _make_search_result("same content", score=0.4, id="r1")
        r_high = _make_search_result("same content", score=0.9, id="r2")
        fused = self._fuse([r_low, r_high], limit=10)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(0.9)
        assert fused[0].id == "r2"

    def test_fuse_respects_limit(self):
        """10 distinct results with limit=3 → only top 3 returned."""
        results = [_make_search_result(f"memory {i}", score=float(i) / 10) for i in range(10)]
        fused = self._fuse(results, limit=3)
        assert len(fused) == 3

    def test_fuse_empty_input(self):
        """Empty list → empty list."""
        assert self._fuse([], limit=5) == []

    def test_fuse_single_result(self):
        """One result → that same result returned."""
        r = _make_search_result("only one", score=0.7)
        fused = self._fuse([r], limit=5)
        assert len(fused) == 1
        assert fused[0].content == "only one"

    def test_fuse_score_ordering(self):
        """5 results with known scores → output is strictly descending."""
        scores = [0.3, 0.9, 0.1, 0.7, 0.5]
        results = [_make_search_result(f"item {i}", score=s) for i, s in enumerate(scores)]
        fused = self._fuse(results, limit=10)
        returned_scores = [r.score for r in fused]
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_fuse_different_content_preserved(self):
        """Distinct content (even if similar) → both kept."""
        r1 = _make_search_result("PostgreSQL is the primary database", score=0.8)
        r2 = _make_search_result("PostgreSQL is the main database", score=0.7)
        fused = self._fuse([r1, r2], limit=10)
        assert len(fused) == 2
