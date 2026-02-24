"""Tests for EngramClient — auto-memory LLM wrapper.

Uses mocked litellm.acompletion and real (tmp) episodic/semantic stores.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.client import EngramClient, _extract_last_user_content, _high_overlap
from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
from engram.models import MemoryType

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FIXED_EMBEDDING = [0.1] * 384


def _fake_embeddings(_model, texts, _expected_dim=None):
    return [_FIXED_EMBEDDING for _ in texts]


def _make_litellm_response(content: str = "Test response"):
    """Build a minimal litellm-style response object."""
    msg = SimpleNamespace(content=content, role="assistant")
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def tmp_client(tmp_path):
    """EngramClient backed by tmp stores with mocked embeddings and config."""
    episodic_cfg = EpisodicConfig(path=str(tmp_path / "episodic"))
    semantic_cfg = SemanticConfig(path=str(tmp_path / "semantic.db"))
    embed_cfg = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")

    cfg = MagicMock()
    cfg.episodic = episodic_cfg
    cfg.semantic = semantic_cfg
    cfg.embedding = embed_cfg
    cfg.llm.model = "gemini/gemini-2.0-flash"

    with patch("engram.client.load_config", return_value=cfg), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        client = EngramClient(namespace="test")
        yield client


# ---------------------------------------------------------------------------
# chat() core behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_calls_litellm(tmp_client):
    """chat() forwards messages to litellm.acompletion and returns its response."""
    messages = [{"role": "user", "content": "Hello"}]
    fake_response = _make_litellm_response("Hi there")

    with patch("engram.client.litellm.acompletion", new_callable=AsyncMock, return_value=fake_response), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        response = await tmp_client.chat(messages)

    assert response.choices[0].message.content == "Hi there"


@pytest.mark.asyncio
async def test_chat_auto_recall_injects_memories(tmp_client):
    """chat() injects stored memories as system context before LLM call."""
    # Pre-store a memory
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client.remember("Python is the preferred language", priority=7)

    messages = [{"role": "user", "content": "What language should we use?"}]
    fake_response = _make_litellm_response("Use Python")
    captured_messages = []

    async def capture_call(model, messages, **kwargs):
        captured_messages.extend(messages)
        return fake_response

    with patch("engram.client.litellm.acompletion", side_effect=capture_call), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client.chat(messages)

    # System message with memories should have been injected
    system_msgs = [m for m in captured_messages if m.get("role") == "system"]
    assert len(system_msgs) >= 1
    assert "Relevant memories" in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_chat_with_auto_recall_disabled(tmp_client):
    """chat() with auto_recall=False skips memory injection."""
    tmp_client._auto_recall = False
    messages = [{"role": "user", "content": "Hello"}]
    fake_response = _make_litellm_response("Hi")
    captured_messages = []

    async def capture_call(model, messages, **kwargs):
        captured_messages.extend(messages)
        return fake_response

    with patch("engram.client.litellm.acompletion", side_effect=capture_call):
        await tmp_client.chat(messages)

    # No system message should be injected
    system_msgs = [m for m in captured_messages if m.get("role") == "system"]
    assert len(system_msgs) == 0


@pytest.mark.asyncio
async def test_chat_with_auto_extract_disabled(tmp_client):
    """chat() with auto_extract=False does not schedule extraction task."""
    tmp_client._auto_extract = False
    messages = [{"role": "user", "content": "Hello"}]
    fake_response = _make_litellm_response("Hi")

    with patch("engram.client.litellm.acompletion", new_callable=AsyncMock, return_value=fake_response), \
         patch("engram.client.asyncio.create_task") as mock_create_task, \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client.chat(messages)

    mock_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_chat_failopen_on_memory_error(tmp_client):
    """chat() still returns LLM response even if memory recall raises."""
    messages = [{"role": "user", "content": "Hello"}]
    fake_response = _make_litellm_response("Hi")

    with patch("engram.client.litellm.acompletion", new_callable=AsyncMock, return_value=fake_response), \
         patch.object(tmp_client, "_recall_for_context", side_effect=RuntimeError("DB down")):
        response = await tmp_client.chat(messages)

    assert response.choices[0].message.content == "Hi"


@pytest.mark.asyncio
async def test_chat_auto_extract_stores_facts(tmp_client):
    """After chat(), _extract_and_store is triggered and stores facts."""
    messages = [{"role": "user", "content": "We decided to use PostgreSQL"}]
    fake_response = _make_litellm_response("Great choice, PostgreSQL is robust.")

    extracted_items = [{"content": "Decision: use PostgreSQL", "type": "decision", "priority": 8}]

    with patch("engram.client.litellm.acompletion", new_callable=AsyncMock, return_value=fake_response), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        # Patch extractor to return known items
        extractor_mock = MagicMock()
        extractor_mock.extract = AsyncMock(return_value=extracted_items)
        tmp_client._extractor = extractor_mock

        await tmp_client.chat(messages)

        # Allow background task to complete
        await asyncio.sleep(0.05)

    # Verify memory was stored
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        results = await tmp_client.recall("PostgreSQL decision")
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# remember() / recall() / think()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remember_stores_memory(tmp_client):
    """remember() stores a memory and recall() finds it."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = await tmp_client.remember("Deploy on Fridays is forbidden", priority=8)
        assert isinstance(mem_id, str) and len(mem_id) > 0

        results = await tmp_client.recall("Friday deploy")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_recall_searches_memories(tmp_client):
    """recall() returns stored memories matching the query."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client.remember("Use Python 3.12", memory_type="decision", priority=7)
        results = await tmp_client.recall("Python version", limit=3)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)


@pytest.mark.asyncio
async def test_think_uses_reasoning(tmp_client):
    """think() delegates to ReasoningEngine and returns a string answer."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client.remember("The project uses FastAPI", priority=6)

    fake_answer = "The project uses FastAPI."
    engine_mock = MagicMock()
    engine_mock.think = AsyncMock(return_value=fake_answer)

    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        await tmp_client._ensure_engine()
    tmp_client._engine = engine_mock

    result = await tmp_client.think("What framework does the project use?")
    assert result == fake_answer
    engine_mock.think.assert_awaited_once()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_deduplication(tmp_client):
    """_dedup_memories skips items with high word overlap to existing memories."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        # Store a memory first
        await tmp_client.remember("User prefers dark mode in the UI", priority=6)

        # Build items where one is near-duplicate
        items = [
            {"content": "User prefers dark mode in the UI", "type": "preference", "priority": 6},
            {"content": "Server should run on port 8080", "type": "fact", "priority": 5},
        ]
        new_items = await tmp_client._dedup_memories(items)

    # The duplicate should be filtered; the new one should remain
    contents = [i["content"] for i in new_items]
    assert "Server should run on port 8080" in contents
    # Near-duplicate should be gone
    assert "User prefers dark mode in the UI" not in contents


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------

def test_sync_wrappers_work(tmp_client):
    """Sync wrappers (remember_sync, recall_sync) execute without event loop errors."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        mem_id = tmp_client.remember_sync("Sync test memory", priority=5)
        assert isinstance(mem_id, str)

        results = tmp_client.recall_sync("Sync test")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_namespace_isolation(tmp_path):
    """Two clients with different namespaces have separate memory stores."""
    episodic_cfg_a = EpisodicConfig(path=str(tmp_path / "episodic"))
    episodic_cfg_b = EpisodicConfig(path=str(tmp_path / "episodic"))
    semantic_cfg = SemanticConfig(path=str(tmp_path / "semantic.db"))
    embed_cfg = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")

    def make_cfg(episodic_cfg):
        cfg = MagicMock()
        cfg.episodic = episodic_cfg
        cfg.semantic = semantic_cfg
        cfg.embedding = embed_cfg
        cfg.llm.model = "gemini/gemini-2.0-flash"
        return cfg

    with patch("engram.client.load_config", return_value=make_cfg(episodic_cfg_a)), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        client_a = EngramClient(namespace="agent-a")
        await client_a.remember("Secret of agent A", priority=8)
        results_a = await client_a.recall("Secret")

    with patch("engram.client.load_config", return_value=make_cfg(episodic_cfg_b)), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        client_b = EngramClient(namespace="agent-b")
        results_b = await client_b.recall("Secret")

    # agent-a should find the memory; agent-b should not
    assert len(results_a) >= 1
    assert len(results_b) == 0


# ---------------------------------------------------------------------------
# Memory injection helpers
# ---------------------------------------------------------------------------

def test_inject_memories_prepends_system_message():
    """_inject_memories adds system message when none exists."""
    client = MagicMock(spec=EngramClient)
    # Call the real method
    result = EngramClient._inject_memories(client, [{"role": "user", "content": "Hi"}], "- fact 1")
    assert result[0]["role"] == "system"
    assert "Relevant memories" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_inject_memories_appends_to_existing_system():
    """_inject_memories appends to existing system message."""
    client = MagicMock(spec=EngramClient)
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}]
    result = EngramClient._inject_memories(client, messages, "- fact 1")
    assert result[0]["role"] == "system"
    assert "You are helpful." in result[0]["content"]
    assert "Relevant memories" in result[0]["content"]
    assert len(result) == 2  # no extra message added


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_extract_last_user_content_basic():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert _extract_last_user_content(msgs) == "hello"


def test_extract_last_user_content_multipart():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]}]
    assert _extract_last_user_content(msgs) == "hi"


def test_extract_last_user_content_no_user_msg():
    msgs = [{"role": "assistant", "content": "hello"}]
    assert _extract_last_user_content(msgs) == ""


def test_high_overlap_detects_duplicates():
    assert _high_overlap("user prefers dark mode", "user prefers dark mode", threshold=0.85) is True


def test_high_overlap_allows_distinct():
    assert _high_overlap("server port is 8080", "user prefers dark mode", threshold=0.85) is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_manager_closes_on_exit(tmp_path):
    """async with EngramClient closes resources on exit."""
    episodic_cfg = EpisodicConfig(path=str(tmp_path / "episodic"))
    semantic_cfg = SemanticConfig(path=str(tmp_path / "semantic.db"))
    embed_cfg = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    cfg = MagicMock()
    cfg.episodic = episodic_cfg
    cfg.semantic = semantic_cfg
    cfg.embedding = embed_cfg
    cfg.llm.model = "gemini/gemini-2.0-flash"

    with patch("engram.client.load_config", return_value=cfg), \
         patch("engram.episodic.store._get_embeddings", side_effect=_fake_embeddings):
        async with EngramClient(namespace="ctx-test") as client:
            await client.remember("context manager test", priority=5)
        # After exit, stores should be cleared
        assert client._episodic is None
        assert client._graph is None


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

def test_engram_client_importable_from_package():
    """EngramClient is exported at the engram package level."""
    from engram import EngramClient as EC  # noqa: F401
    assert EC is EngramClient
