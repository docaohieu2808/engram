"""Tests for Phase 4 new features: TTL expiry, tags, namespace isolation, summarize."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.cli.episodic import _parse_duration
from engram.config import EmbeddingConfig, EpisodicConfig
from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory, MemoryType

_FIXED_EMBEDDING = [0.1] * 384


def _fake_embeddings(_model, texts, _expected_dim=None):
    return [_FIXED_EMBEDDING for _ in texts]


@pytest.fixture
def store(tmp_path):
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        cfg = EpisodicConfig(path=str(tmp_path / "episodic"))
        emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
        yield EpisodicStore(config=cfg, embedding_config=emb)


@pytest.fixture
def ns_store(tmp_path):
    """Factory fixture returning a function that creates stores with specified namespaces."""
    def _make(namespace: str):
        cfg = EpisodicConfig(path=str(tmp_path / "episodic"))
        emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
        return EpisodicStore(config=cfg, embedding_config=emb, namespace=namespace)
    return _make


# --- TTL Expiry ---


@pytest.mark.asyncio
async def test_expired_memory_filtered_from_search(store):
    """Expired memories are not returned by search()."""
    past = datetime.now() - timedelta(hours=1)
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("This memory is expired", expires_at=past)
        results = await store.search("expired memory")
    assert all("expired" not in r.content for r in results)


@pytest.mark.asyncio
async def test_non_expired_memory_visible_in_search(store):
    """Non-expired memories with future expires_at are returned by search()."""
    future = datetime.now() + timedelta(hours=24)
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Future expiry memory", expires_at=future)
        results = await store.search("future expiry memory")
    assert any("Future expiry" in r.content for r in results)


@pytest.mark.asyncio
async def test_cleanup_expired_returns_count(store):
    """cleanup_expired() deletes expired memories and returns count."""
    past = datetime.now() - timedelta(hours=1)
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Expired one", expires_at=past)
        await store.remember("Expired two", expires_at=past)
        await store.remember("Still valid")
    deleted = await store.cleanup_expired()
    assert deleted == 2


@pytest.mark.asyncio
async def test_cleanup_expired_removes_from_store(store):
    """After cleanup_expired(), collection count decreases by expired count."""
    past = datetime.now() - timedelta(seconds=1)
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Expired", expires_at=past)
        await store.remember("Valid")
    await store.cleanup_expired()
    stats = await store.stats()
    assert stats["count"] == 1


@pytest.mark.asyncio
async def test_cleanup_no_expired_returns_zero(store):
    """cleanup_expired() returns 0 when no memories are expired."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Normal memory")
    result = await store.cleanup_expired()
    assert result == 0


# --- Tags ---


@pytest.mark.asyncio
async def test_tags_stored_and_retrieved(store):
    """Tags are stored in metadata and retrieved as list on the memory object."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await store.remember("Deploy to prod", tags=["deploy", "prod"])
        mem = await store.get(mem_id)
    assert mem is not None
    assert "deploy" in mem.tags
    assert "prod" in mem.tags


@pytest.mark.asyncio
async def test_tags_filter_in_search(store):
    """Search with tags filter returns only memories that contain all specified tags."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Deploy prod release", tags=["deploy", "prod"])
        await store.remember("Deploy staging build", tags=["deploy", "staging"])
        await store.remember("Unrelated memory")
        results = await store.search("deploy", tags=["prod"])
    assert len(results) >= 1
    assert all("prod" in r.tags for r in results)


@pytest.mark.asyncio
async def test_tags_filter_excludes_non_matching(store):
    """Memories without the required tag are excluded from search results."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Has tag alpha", tags=["alpha"])
        await store.remember("Has tag beta", tags=["beta"])
        results = await store.search("tag", tags=["alpha"])
    assert all("alpha" in r.tags for r in results)
    assert not any("beta" in r.tags and "alpha" not in r.tags for r in results)


@pytest.mark.asyncio
async def test_empty_tags_default(store):
    """Memory stored without tags has empty tags list."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await store.remember("No tags here")
        mem = await store.get(mem_id)
    assert mem is not None
    assert mem.tags == []


# --- Namespace isolation ---


@pytest.mark.asyncio
async def test_namespace_isolation(tmp_path, ns_store):
    """Memories in different namespaces are isolated from each other."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        store_a = ns_store("alpha")
        store_b = ns_store("beta")
        await store_a.remember("Memory in alpha namespace")
        results_b = await store_b.search("alpha namespace")
    assert len(results_b) == 0


@pytest.mark.asyncio
async def test_namespace_sees_own_data(tmp_path, ns_store):
    """A namespace can retrieve its own memories."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        store_a = ns_store("alpha")
        await store_a.remember("Alpha only memory")
        results = await store_a.search("alpha memory")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_namespace_collection_name(tmp_path):
    """EpisodicStore uses namespace-specific collection name."""
    cfg = EpisodicConfig(path=str(tmp_path / "ep"))
    emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    store = EpisodicStore(config=cfg, embedding_config=emb, namespace="myns")
    assert store.COLLECTION_NAME == "engram_myns"


@pytest.mark.asyncio
async def test_default_namespace_collection_name(tmp_path):
    """Default namespace uses 'engram_default' collection name."""
    cfg = EpisodicConfig(path=str(tmp_path / "ep"))
    emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    store = EpisodicStore(config=cfg, embedding_config=emb)
    assert store.COLLECTION_NAME == "engram_default"


# --- Summarize ---


@pytest.mark.asyncio
async def test_summarize_empty_store(store):
    """summarize() returns a 'no memories' message when store is empty."""
    from engram.reasoning.engine import ReasoningEngine
    graph_mock = MagicMock()
    engine = ReasoningEngine(store, graph_mock, model="test/model")
    result = await engine.summarize(n=5)
    assert "No memories" in result


@pytest.mark.asyncio
async def test_summarize_calls_llm(store):
    """summarize() calls LLM with recent memories and returns its response."""
    from engram.reasoning.engine import ReasoningEngine

    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Deployed new version")
        await store.remember("Fixed critical bug")

    graph_mock = MagicMock()
    engine = ReasoningEngine(store, graph_mock, model="gemini/gemini-2.0-flash")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Key insight: deployment and bug fix completed."

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        result = await engine.summarize(n=5)

    assert "Key insight" in result


@pytest.mark.asyncio
async def test_summarize_save_stores_memory(store):
    """summarize(save=True) stores the summary as a new memory with type=context."""
    from engram.reasoning.engine import ReasoningEngine

    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await store.remember("Original memory A")
        await store.remember("Original memory B")

    graph_mock = MagicMock()
    engine = ReasoningEngine(store, graph_mock, model="gemini/gemini-2.0-flash")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Summary: two memories recorded."

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
            await engine.summarize(n=5, save=True)

    # Verify a context-type memory was saved
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        results = await store.search("Summary")
    context_mems = [r for r in results if r.memory_type == MemoryType.CONTEXT]
    assert len(context_mems) >= 1


# --- Duration parsing ---


def test_parse_duration_days():
    before = datetime.now()
    result = _parse_duration("7d")
    assert result > before + timedelta(days=6, hours=23)
    assert result < before + timedelta(days=7, seconds=5)


def test_parse_duration_hours():
    before = datetime.now()
    result = _parse_duration("24h")
    assert result > before + timedelta(hours=23, minutes=59)


def test_parse_duration_minutes():
    before = datetime.now()
    result = _parse_duration("30m")
    assert result > before + timedelta(minutes=29, seconds=59)


def test_parse_duration_seconds():
    before = datetime.now()
    result = _parse_duration("60s")
    assert result > before + timedelta(seconds=59)


def test_parse_duration_invalid():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_duration("invalid")


def test_parse_duration_invalid_unit():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_duration("7w")
