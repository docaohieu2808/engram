"""Tests for MCP episodic tool handlers (remember, recall, cleanup)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engram.models import MemoryType


def _make_mcp_tools(register_fn, *args):
    """Register MCP tools and return dict of {name: fn}."""
    mcp = MagicMock()
    tools = {}

    def capture_tool():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator

    mcp.tool = capture_tool
    register_fn(mcp, *args)
    return tools


# ---------------------------------------------------------------------------
# engram_remember
# ---------------------------------------------------------------------------

class TestEngramRemember:
    @pytest.mark.asyncio
    async def test_remember_returns_confirmation(self, episodic_store):
        from engram.mcp.episodic_tools import register
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())

        result = await tools["engram_remember"]("Hello world", memory_type="fact", priority=5)
        assert "Remembered" in result
        assert "fact" in result
        assert "priority=5" in result

    @pytest.mark.asyncio
    async def test_remember_stores_with_tags(self, episodic_store):
        from engram.mcp.episodic_tools import register
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())

        await tools["engram_remember"](
            "Tagged content", memory_type="decision", priority=7,
            tags=["important", "arch"],
        )
        results = await episodic_store.search("Tagged content")
        assert len(results) > 0
        assert results[0].tags == ["important", "arch"]

    @pytest.mark.asyncio
    async def test_remember_default_type_is_fact(self, episodic_store):
        from engram.mcp.episodic_tools import register
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())

        result = await tools["engram_remember"]("default type content")
        assert "fact" in result


# ---------------------------------------------------------------------------
# engram_recall
# ---------------------------------------------------------------------------

class TestEngramRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_formatted_text(self, episodic_store):
        from engram.mcp.episodic_tools import register
        await episodic_store.remember("Python is a language", memory_type=MemoryType.FACT)

        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())
        result = await tools["engram_recall"]("Python")
        assert "Python" in result
        assert "fact" in result

    @pytest.mark.asyncio
    async def test_recall_no_results_returns_message(self, episodic_store):
        from engram.mcp.episodic_tools import register
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())

        result = await tools["engram_recall"]("nonexistent xyz abc 999")
        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_recall_with_memory_type_filter(self, episodic_store):
        from engram.mcp.episodic_tools import register
        await episodic_store.remember("a decision was made", memory_type=MemoryType.DECISION)
        await episodic_store.remember("a plain fact", memory_type=MemoryType.FACT)

        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())
        result = await tools["engram_recall"]("memory", memory_type="decision")
        assert "decision" in result

    @pytest.mark.asyncio
    async def test_recall_includes_entities_and_tags(self, episodic_store):
        from engram.mcp.episodic_tools import register
        await episodic_store.remember(
            "memory with metadata", entities=["Service:API"], tags=["prod"]
        )
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())
        result = await tools["engram_recall"]("metadata")
        assert "Service:API" in result
        assert "prod" in result


# ---------------------------------------------------------------------------
# engram_cleanup
# ---------------------------------------------------------------------------

class TestEngramCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_no_expired_returns_message(self, episodic_store):
        from engram.mcp.episodic_tools import register
        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())

        result = await tools["engram_cleanup"]()
        assert "No expired memories" in result

    @pytest.mark.asyncio
    async def test_cleanup_expired_returns_count(self, episodic_store):
        from datetime import datetime, timedelta, timezone
        from engram.mcp.episodic_tools import register

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        await episodic_store.remember("old memory", expires_at=past)
        await episodic_store.remember("another old one", expires_at=past)

        tools = _make_mcp_tools(register, lambda: episodic_store, lambda: None, lambda: MagicMock())
        result = await tools["engram_cleanup"]()
        assert "2" in result
        assert "memor" in result.lower()
