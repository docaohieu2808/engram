"""Tests for ReasoningEngine LLM synthesis over episodic + semantic memory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.reasoning.engine import ReasoningEngine


def _make_llm_response(text: str):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


async def test_think_with_results(episodic_store, semantic_graph):
    """think() calls LLM when episodic results are found, returns degraded=False."""
    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        await episodic_store.remember("Deployed v2 to production at noon")

    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")
    canned = _make_llm_response("Deployment happened at noon.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=canned)):
        result = await engine.think("When was the deployment?")

    assert isinstance(result, dict)
    assert result["degraded"] is False
    answer = result["answer"]
    assert "noon" in answer.lower() or "deployment" in answer.lower()


async def test_think_no_results(tmp_path):
    """think() returns fallback message when both stores are empty, degraded=False."""
    from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
    from engram.episodic.store import EpisodicStore
    from engram.semantic import create_graph

    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        store = EpisodicStore(
            config=EpisodicConfig(path=str(tmp_path / "ep"), dedup_enabled=False),
            embedding_config=EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2"),
        )
    graph = create_graph(SemanticConfig(path=str(tmp_path / "sem.db")))
    engine = ReasoningEngine(episodic=store, graph=graph, model="test/model")

    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        result = await engine.think("What is the meaning of life?")

    assert isinstance(result, dict)
    assert result["degraded"] is False
    assert "no relevant memories" in result["answer"].lower()


async def test_think_degraded_when_llm_blocked(episodic_store, semantic_graph):
    """think() returns degraded=True when ResourceMonitor blocks LLM."""
    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        await episodic_store.remember("Deployed v2 to production at noon")

    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")

    # Mock the resource monitor to block LLM usage
    mock_monitor = MagicMock()
    mock_monitor.can_use_llm.return_value = False
    mock_tier = MagicMock()
    mock_tier.value = "basic"
    mock_monitor.get_tier.return_value = mock_tier

    with patch("engram.reasoning.engine.get_resource_monitor", return_value=mock_monitor):
        result = await engine.think("When was the deployment?")

    assert isinstance(result, dict)
    assert result["degraded"] is True
    # Raw answer should contain memory content, not LLM synthesis
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


async def test_think_not_degraded_on_successful_llm(episodic_store, semantic_graph):
    """think() returns degraded=False when LLM call succeeds."""
    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        await episodic_store.remember("Database migration ran on Monday")

    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")
    canned = _make_llm_response("Migration ran Monday.")

    mock_monitor = MagicMock()
    mock_monitor.can_use_llm.return_value = True

    with patch("engram.reasoning.engine.get_resource_monitor", return_value=mock_monitor), \
         patch("litellm.acompletion", new=AsyncMock(return_value=canned)):
        result = await engine.think("When did the migration run?")

    assert isinstance(result, dict)
    assert result["degraded"] is False
    assert "migration" in result["answer"].lower() or "monday" in result["answer"].lower()
