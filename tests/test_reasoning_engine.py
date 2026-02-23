"""Tests for ReasoningEngine LLM synthesis over episodic + semantic memory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.reasoning.engine import ReasoningEngine


def _make_llm_response(text: str):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


async def test_think_with_results(episodic_store, semantic_graph):
    """think() calls LLM when episodic results are found."""
    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        await episodic_store.remember("Deployed v2 to production at noon")

    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")
    canned = _make_llm_response("Deployment happened at noon.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=canned)):
        answer = await engine.think("When was the deployment?")

    assert "noon" in answer.lower() or "deployment" in answer.lower()


async def test_think_no_results(tmp_path):
    """think() returns fallback message when both stores are empty."""
    from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
    from engram.episodic.store import EpisodicStore
    from engram.semantic.graph import SemanticGraph

    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        store = EpisodicStore(
            config=EpisodicConfig(path=str(tmp_path / "ep")),
            embedding_config=EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2"),
        )
    graph = SemanticGraph(config=SemanticConfig(path=str(tmp_path / "sem.db")))
    engine = ReasoningEngine(episodic=store, graph=graph, model="test/model")

    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)):
        answer = await engine.think("What is the meaning of life?")

    assert "no relevant memories" in answer.lower()
