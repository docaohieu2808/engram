"""Tests for memory consolidation engine (consolidation/engine.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ConsolidationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.models import EpisodicMemory, MemoryType


# --- Helpers ---

def _mem(content: str, entities: list[str] = None, tags: list[str] = None,
         consolidated_into: str | None = None, mem_id: str = "m1") -> EpisodicMemory:
    return EpisodicMemory(
        id=mem_id, content=content, memory_type=MemoryType.FACT,
        entities=entities or [], tags=tags or [],
        consolidated_into=consolidated_into,
    )


@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.remember = AsyncMock(return_value="consolidated-id-1")
    ep.update_metadata = AsyncMock()
    return ep


@pytest.fixture
def engine(mock_episodic):
    config = ConsolidationConfig(min_cluster_size=2, similarity_threshold=0.3)
    return ConsolidationEngine(mock_episodic, model="gpt-4", config=config)


# --- Tests: not enough memories ---

@pytest.mark.asyncio
async def test_consolidate_skips_when_too_few(engine, mock_episodic):
    """Skip consolidation when fewer memories than min_cluster_size."""
    mock_episodic.get_recent = AsyncMock(return_value=[_mem("only one")])
    result = await engine.consolidate()
    assert result == []
    mock_episodic.remember.assert_not_called()


@pytest.mark.asyncio
async def test_consolidate_skips_already_consolidated(engine, mock_episodic):
    """Already-consolidated memories are filtered out."""
    mems = [
        _mem("mem1", consolidated_into="old-id", mem_id="m1"),
        _mem("mem2", consolidated_into="old-id", mem_id="m2"),
    ]
    mock_episodic.get_recent = AsyncMock(return_value=mems)
    result = await engine.consolidate()
    assert result == []


# --- Tests: successful consolidation ---

@pytest.mark.asyncio
async def test_consolidate_clusters_and_stores(engine, mock_episodic):
    """Memories with overlapping entities are clustered and summarized."""
    mems = [
        _mem("User likes Python", entities=["python", "user"], tags=["preference"], mem_id="m1"),
        _mem("User uses Python daily", entities=["python", "user"], tags=["pattern"], mem_id="m2"),
        _mem("User prefers VS Code", entities=["vscode", "user"], tags=["preference"], mem_id="m3"),
    ]
    mock_episodic.get_recent = AsyncMock(return_value=mems)

    with patch("engram.consolidation.engine.litellm") as mock_llm:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "User is a daily Python developer."
        mock_llm.acompletion = AsyncMock(return_value=mock_response)

        result = await engine.consolidate()

    # Should have created at least one consolidated memory
    assert len(result) >= 1
    mock_episodic.remember.assert_called()
    # Originals should be marked as consolidated
    assert mock_episodic.update_metadata.call_count >= 2


# --- Tests: clustering logic ---

def test_cluster_by_overlap_groups_similar(engine):
    """Memories sharing entities/tags are grouped together."""
    mems = [
        _mem("A", entities=["x", "y"], mem_id="m1"),
        _mem("B", entities=["y", "z"], mem_id="m2"),  # shares y with A
        _mem("C", entities=["a", "b"], mem_id="m3"),  # disjoint
    ]
    clusters = engine._cluster_by_overlap(mems)
    # A and B should cluster, C separate
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_overlap_empty_features(engine):
    """Memories with no entities/tags stay in their own cluster."""
    mems = [
        _mem("A", mem_id="m1"),
        _mem("B", mem_id="m2"),
        _mem("C", mem_id="m3"),
    ]
    clusters = engine._cluster_by_overlap(mems)
    assert len(clusters) == 3  # each in own cluster


def test_cluster_caps_at_200(engine):
    """Clustering caps at 200 memories to prevent O(n^2) explosion."""
    mems = [_mem(f"mem-{i}", entities=["shared"], mem_id=f"m{i}") for i in range(300)]
    clusters = engine._cluster_by_overlap(mems)
    # All 200 share "shared" entity, so they form one cluster
    total = sum(len(c) for c in clusters)
    assert total == 200


def test_cluster_threshold_respected(mock_episodic):
    """High threshold means fewer clusters merge."""
    config = ConsolidationConfig(min_cluster_size=2, similarity_threshold=0.9)
    eng = ConsolidationEngine(mock_episodic, model="gpt-4", config=config)
    mems = [
        _mem("A", entities=["x", "y", "z"], tags=["t1"], mem_id="m1"),
        _mem("B", entities=["x"], tags=["t2"], mem_id="m2"),  # low overlap
    ]
    clusters = eng._cluster_by_overlap(mems)
    assert len(clusters) == 2  # threshold too high to merge


# --- Tests: LLM summarization ---

@pytest.mark.asyncio
async def test_summarize_cluster_calls_llm(engine):
    mems = [_mem("fact 1", mem_id="m1"), _mem("fact 2", mem_id="m2")]
    with patch("engram.consolidation.engine.litellm") as mock_llm:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "  Summary of facts.  "
        mock_llm.acompletion = AsyncMock(return_value=mock_response)

        summary = await engine._summarize_cluster(mems)

    assert summary == "Summary of facts."  # stripped
    mock_llm.acompletion.assert_called_once()
    call_args = mock_llm.acompletion.call_args
    assert call_args.kwargs["temperature"] == 0.3
    assert call_args.kwargs["max_tokens"] == 300


# --- Tests: error handling ---

@pytest.mark.asyncio
async def test_consolidate_handles_llm_failure(engine, mock_episodic):
    """LLM failure for one cluster doesn't stop other clusters."""
    mems = [
        _mem("A", entities=["x"], mem_id="m1"),
        _mem("B", entities=["x"], mem_id="m2"),
    ]
    mock_episodic.get_recent = AsyncMock(return_value=mems)

    with patch("engram.consolidation.engine.litellm") as mock_llm:
        mock_llm.acompletion = AsyncMock(side_effect=RuntimeError("LLM down"))
        result = await engine.consolidate()

    # Should return empty (failed) but not raise
    assert result == []


# --- Tests: metadata propagation ---

@pytest.mark.asyncio
async def test_consolidated_memory_has_correct_tags(engine, mock_episodic):
    mems = [
        _mem("A", entities=["shared", "e1"], tags=["t1"], mem_id="m1"),
        _mem("B", entities=["shared", "e2"], tags=["t1"], mem_id="m2"),
    ]
    mock_episodic.get_recent = AsyncMock(return_value=mems)

    with patch("engram.consolidation.engine.litellm") as mock_llm:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_llm.acompletion = AsyncMock(return_value=mock_response)

        await engine.consolidate()

    # Check the remember call includes merged entities/tags + "consolidated" tag
    call_kwargs = mock_episodic.remember.call_args.kwargs
    assert "consolidated" in call_kwargs["tags"]
    assert "shared" in call_kwargs["entities"]
    assert call_kwargs["memory_type"] == MemoryType.CONTEXT
    assert call_kwargs["priority"] == 6
