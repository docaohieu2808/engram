"""Tests for Phase 3 performance additions: batch ops and entity hints caching."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from engram.models import MemoryType, SemanticEdge, SemanticNode
from engram.reasoning.engine import ReasoningEngine


# ---------------------------------------------------------------------------
# EpisodicStore.remember_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_batch_returns_correct_count(episodic_store):
    """remember_batch() returns one ID per input memory."""
    batch = [
        {"content": "First event"},
        {"content": "Second event", "memory_type": MemoryType.DECISION, "priority": 8},
        {"content": "Third event", "entities": ["api", "db"]},
    ]
    ids = await episodic_store.remember_batch(batch)
    assert len(ids) == 3
    assert all(isinstance(i, str) and len(i) > 0 for i in ids)


@pytest.mark.asyncio
async def test_remember_batch_ids_are_unique(episodic_store):
    """Each ID returned by remember_batch() is unique."""
    batch = [{"content": f"Memory {i}"} for i in range(5)]
    ids = await episodic_store.remember_batch(batch)
    assert len(set(ids)) == 5


@pytest.mark.asyncio
async def test_remember_batch_empty_returns_empty(episodic_store):
    """remember_batch([]) returns empty list without error."""
    ids = await episodic_store.remember_batch([])
    assert ids == []


@pytest.mark.asyncio
async def test_remember_batch_uses_single_embedding_call(tmp_path):
    """remember_batch() calls _get_embeddings once for all texts."""
    from engram.config import EmbeddingConfig, EpisodicConfig
    from engram.episodic.store import EpisodicStore

    cfg = EpisodicConfig(path=str(tmp_path / "ep"))
    emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    store = EpisodicStore(config=cfg, embedding_config=emb)

    batch = [{"content": f"Item {i}"} for i in range(4)]
    with patch("engram.episodic.store._get_embeddings", side_effect=lambda m, t, d=None: [[0.1] * 384] * len(t)) as mock_embed:
        await store.remember_batch(batch)

    # Should be called exactly once (single batch call, not once per item)
    assert mock_embed.call_count == 1
    # The single call should have received all 4 texts
    texts_arg = mock_embed.call_args[0][1]
    assert len(texts_arg) == 4


@pytest.mark.asyncio
async def test_remember_batch_stored_memories_are_retrievable(episodic_store):
    """Memories stored via remember_batch() can be retrieved by get()."""
    batch = [{"content": "Batch stored fact"}]
    ids = await episodic_store.remember_batch(batch)
    mem = await episodic_store.get(ids[0])
    assert mem is not None
    assert mem.content == "Batch stored fact"


@pytest.mark.asyncio
async def test_remember_batch_preserves_memory_type(episodic_store):
    """remember_batch() correctly stores memory_type metadata."""
    batch = [{"content": "A decision was made", "memory_type": MemoryType.DECISION}]
    ids = await episodic_store.remember_batch(batch)
    mem = await episodic_store.get(ids[0])
    assert mem is not None
    assert mem.memory_type == MemoryType.DECISION


# ---------------------------------------------------------------------------
# SemanticGraph.add_nodes_batch / add_edges_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_nodes_batch_all_stored(semantic_graph):
    """add_nodes_batch() stores all nodes, retrievable via get_nodes()."""
    nodes = [
        SemanticNode(type="Service", name="auth"),
        SemanticNode(type="Service", name="payments"),
        SemanticNode(type="Team", name="platform"),
    ]
    await semantic_graph.add_nodes_batch(nodes)
    all_nodes = await semantic_graph.get_nodes()
    names = {n.name for n in all_nodes}
    assert {"auth", "payments", "platform"} == names


@pytest.mark.asyncio
async def test_add_nodes_batch_empty_is_noop(semantic_graph):
    """add_nodes_batch([]) does not raise and leaves graph empty."""
    await semantic_graph.add_nodes_batch([])
    nodes = await semantic_graph.get_nodes()
    assert nodes == []


@pytest.mark.asyncio
async def test_add_edges_batch_all_stored(semantic_graph):
    """add_edges_batch() stores all edges, retrievable via get_edges()."""
    await semantic_graph.add_nodes_batch([
        SemanticNode(type="Service", name="api"),
        SemanticNode(type="Team", name="ops"),
        SemanticNode(type="Team", name="security"),
    ])
    edges = [
        SemanticEdge(from_node="Team:ops", to_node="Service:api", relation="owns"),
        SemanticEdge(from_node="Team:security", to_node="Service:api", relation="audits"),
    ]
    await semantic_graph.add_edges_batch(edges)
    stored = await semantic_graph.get_edges()
    relations = {e.relation for e in stored}
    assert {"owns", "audits"} == relations


@pytest.mark.asyncio
async def test_add_edges_batch_empty_is_noop(semantic_graph):
    """add_edges_batch([]) does not raise and leaves edges empty."""
    await semantic_graph.add_edges_batch([])
    edges = await semantic_graph.get_edges()
    assert edges == []


@pytest.mark.asyncio
async def test_add_nodes_batch_persists_to_new_instance(tmp_path):
    """Nodes stored via batch are reloaded by a fresh SemanticGraph instance."""
    from engram.config import SemanticConfig
    from engram.semantic import create_graph

    db_file = str(tmp_path / "sem.db")
    graph1 = create_graph(SemanticConfig(path=db_file))
    await graph1.add_nodes_batch([
        SemanticNode(type="Service", name="worker"),
        SemanticNode(type="Service", name="scheduler"),
    ])
    await graph1.close()

    graph2 = create_graph(SemanticConfig(path=db_file))
    nodes = await graph2.get_nodes()
    assert {n.name for n in nodes} == {"worker", "scheduler"}


# ---------------------------------------------------------------------------
# ReasoningEngine entity hints caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_hints_cache_populated_on_first_think(episodic_store, semantic_graph):
    """Cache is None before think() and populated after."""
    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")
    assert engine._node_names_cache is None

    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_llm_response("ok"))):
        await engine.think("anything")

    assert engine._node_names_cache is not None


@pytest.mark.asyncio
async def test_entity_hints_cache_avoids_repeated_get_nodes(episodic_store, semantic_graph):
    """get_nodes() is only called once across multiple think() calls."""
    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")

    with patch.object(semantic_graph, "get_nodes", wraps=semantic_graph.get_nodes) as spy, \
         patch("litellm.acompletion", new=AsyncMock(return_value=_fake_llm_response("ok"))):
        await engine.think("first question")
        await engine.think("second question")
        await engine.think("third question")

    assert spy.call_count == 1


@pytest.mark.asyncio
async def test_invalidate_cache_resets_to_none(episodic_store, semantic_graph):
    """invalidate_cache() resets _node_names_cache to None."""
    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")
    # Populate cache
    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_llm_response("ok"))):
        await engine.think("question")
    assert engine._node_names_cache is not None

    engine.invalidate_cache()
    assert engine._node_names_cache is None


@pytest.mark.asyncio
async def test_entity_hints_cache_refreshes_after_invalidate(episodic_store, semantic_graph):
    """After invalidate_cache(), next think() calls get_nodes() again."""
    engine = ReasoningEngine(episodic=episodic_store, graph=semantic_graph, model="test/model")

    with patch.object(semantic_graph, "get_nodes", wraps=semantic_graph.get_nodes) as spy, \
         patch("litellm.acompletion", new=AsyncMock(return_value=_fake_llm_response("ok"))):
        await engine.think("first")
        engine.invalidate_cache()
        await engine.think("second")

    # Called once before invalidate, once after — total 2
    assert spy.call_count == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from types import SimpleNamespace


def _fake_llm_response(text: str):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])
