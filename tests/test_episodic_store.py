"""Tests for EpisodicStore CRUD and search operations."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engram.config import EmbeddingConfig, EpisodicConfig
from engram.episodic.store import EpisodicStore
from engram.models import MemoryType

_FIXED_EMBEDDING = [0.1] * 384


def _fake_embeddings(_model, texts, _expected_dim=None):
    return [_FIXED_EMBEDDING for _ in texts]


@pytest.fixture
def store(tmp_path):
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        cfg = EpisodicConfig(path=str(tmp_path / "episodic"), dedup_enabled=False)
        emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
        yield EpisodicStore(config=cfg, embedding_config=emb)


@pytest.mark.asyncio
async def test_remember_returns_id(store):
    """remember() returns a non-empty UUID string."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await store.remember("Deploy failed on prod")
    assert isinstance(mem_id, str) and len(mem_id) > 0


@pytest.mark.asyncio
async def test_search_finds_stored(store):
    """Stored memory is returned by search query."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Database migration completed")
        results = await store.search("migration")
    assert len(results) >= 1
    assert any("migration" in r.content for r in results)


@pytest.mark.asyncio
async def test_search_with_filters(store):
    """Search with memory_type filter returns only matching type."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Decision: use PostgreSQL", memory_type=MemoryType.DECISION)
        await store.remember("Fact: server is down", memory_type=MemoryType.FACT)
        results = await store.search(
            "info", filters={"memory_type": {"$eq": "decision"}}
        )
    assert all(r.memory_type == MemoryType.DECISION for r in results)


@pytest.mark.asyncio
async def test_get_by_id(store):
    """get(id) retrieves the exact memory stored."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await store.remember("Rollback deployed at 14:00")
        mem = await store.get(mem_id)
    assert mem is not None
    assert mem.id == mem_id
    assert "Rollback" in mem.content


@pytest.mark.asyncio
async def test_delete(store):
    """delete(id) returns True; subsequent get returns None."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await store.remember("Temporary debug note")
        deleted = await store.delete(mem_id)
        mem = await store.get(mem_id)
    assert deleted is True
    assert mem is None


@pytest.mark.asyncio
async def test_stats(store):
    """stats() count increments after inserts."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("First memory")
        await store.remember("Second memory")
        s = await store.stats()
    assert s["count"] == 2
