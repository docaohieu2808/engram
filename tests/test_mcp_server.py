"""Tests for MCP server initialization and session tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.models import MemoryType


# --- Tests: MCP server lazy initialization ---

class TestMCPServerInit:
    """Test lazy-initialized singleton getters in mcp/server.py."""

    def test_get_config_caches(self):
        """Config loaded once and cached."""
        with patch("engram.mcp.server._instances", {}):
            with patch("engram.mcp.server.load_config") as mock_load:
                mock_cfg = MagicMock()
                mock_load.return_value = mock_cfg
                from engram.mcp.server import _get_config
                cfg1 = _get_config()
                cfg2 = _get_config()
                assert cfg1 is cfg2
                mock_load.assert_called_once()

    def test_get_episodic_caches(self):
        """EpisodicStore created once per lifetime."""
        mock_cfg = MagicMock()
        mock_cfg.episodic.namespace = "test-ns"
        mock_cfg.hooks.on_remember = None
        with patch("engram.mcp.server._instances", {"config": mock_cfg}):
            with patch("engram.mcp.server.EpisodicStore") as MockStore:
                MockStore.return_value = MagicMock()
                from engram.mcp.server import _get_episodic
                ep1 = _get_episodic()
                ep2 = _get_episodic()
                assert ep1 is ep2
                MockStore.assert_called_once()

    def test_get_graph_caches(self):
        """SemanticGraph created once per lifetime."""
        mock_cfg = MagicMock()
        with patch("engram.mcp.server._instances", {"config": mock_cfg}):
            with patch("engram.mcp.server.create_graph") as MockGraph:
                MockGraph.return_value = MagicMock()
                from engram.mcp.server import _get_graph
                g1 = _get_graph()
                g2 = _get_graph()
                assert g1 is g2
                MockGraph.assert_called_once()

    def test_get_session_store_caches(self):
        """SessionStore created once per lifetime."""
        mock_cfg = MagicMock()
        mock_cfg.session.sessions_dir = "/tmp/test-sessions"
        with patch("engram.mcp.server._instances", {"config": mock_cfg}):
            with patch("engram.mcp.server.SessionStore") as MockSS:
                MockSS.return_value = MagicMock()
                from engram.mcp.server import _get_session_store
                s1 = _get_session_store()
                s2 = _get_session_store()
                assert s1 is s2
                MockSS.assert_called_once()

    def test_get_engine_caches(self):
        """ReasoningEngine created once per lifetime."""
        mock_cfg = MagicMock()
        mock_cfg.llm.model = "test-model"
        mock_cfg.hooks.on_think = None
        mock_cfg.recall_pipeline = MagicMock()
        mock_ep = MagicMock()
        mock_gr = MagicMock()
        mock_providers = MagicMock()
        mock_providers.get_active.return_value = []
        with patch("engram.mcp.server._instances", {
            "config": mock_cfg, "episodic": mock_ep,
            "graph": mock_gr, "providers": mock_providers,
        }):
            with patch("engram.mcp.server.ReasoningEngine") as MockEng:
                MockEng.return_value = MagicMock()
                from engram.mcp.server import _get_engine
                e1 = _get_engine()
                e2 = _get_engine()
                assert e1 is e2
                MockEng.assert_called_once()


# --- Tests: MCP session tools ---

class TestMCPSessionTools:
    """Test session_tools.py functions directly (bypassing MCP transport)."""

    @pytest.fixture
    def mock_session_store(self):
        store = MagicMock()
        return store

    @pytest.fixture
    def mock_episodic(self):
        ep = AsyncMock()
        ep.remember = AsyncMock(return_value="mem-session-1")
        return ep

    @pytest.mark.asyncio
    async def test_session_start(self, mock_session_store):
        """engram_session_start creates a new session."""
        mock_session = MagicMock()
        mock_session.id = "abc12345-full-uuid"
        mock_session_store.start.return_value = mock_session

        # Simulate tool registration and call
        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: AsyncMock())

        result = await tools["engram_session_start"](namespace="test")
        assert "abc12345" in result
        mock_session_store.start.assert_called_once_with(namespace="test")

    @pytest.mark.asyncio
    async def test_session_end_active(self, mock_session_store):
        mock_session = MagicMock()
        mock_session.id = "sess-id-12345678"
        mock_session.started_at = "2026-02-27T00:00:00Z"
        mock_session_store.end.return_value = mock_session

        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: AsyncMock())

        result = await tools["engram_session_end"]()
        assert "sess-id-" in result

    @pytest.mark.asyncio
    async def test_session_end_no_active(self, mock_session_store):
        mock_session_store.end.return_value = None

        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: AsyncMock())

        result = await tools["engram_session_end"]()
        assert "No active session" in result

    @pytest.mark.asyncio
    async def test_session_summary_stores_memory(self, mock_session_store, mock_episodic):
        mock_session = MagicMock()
        mock_session.id = "sess-summary-123"
        mock_session_store.end.return_value = mock_session

        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: mock_episodic)

        result = await tools["engram_session_summary"](
            goal="Fix auth bug",
            discoveries=["JWT expired too fast"],
            accomplished=["Patched expiry logic"],
            files=["auth.py"],
        )
        assert "summarized" in result
        mock_episodic.remember.assert_called_once()
        call_kwargs = mock_episodic.remember.call_args.kwargs
        assert "session-summary" in call_kwargs["tags"]
        assert call_kwargs["memory_type"] == MemoryType.DECISION

    @pytest.mark.asyncio
    async def test_session_context_returns_history(self, mock_session_store):
        sess1 = MagicMock()
        sess1.started_at = "2026-02-27T00:00:00Z"
        sess1.goal = "Implement WebSocket"
        sess1.accomplished = ["Added /ws endpoint"]
        sess1.discoveries = ["FastAPI supports WS natively"]
        mock_session_store.get_recent.return_value = [sess1]

        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: AsyncMock())

        result = await tools["engram_session_context"](limit=5)
        assert "Implement WebSocket" in result
        assert "Added /ws endpoint" in result

    @pytest.mark.asyncio
    async def test_session_context_empty(self, mock_session_store):
        mock_session_store.get_recent.return_value = []

        from engram.mcp.session_tools import register
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register(mcp, lambda: mock_session_store, lambda: AsyncMock())

        result = await tools["engram_session_context"](limit=5)
        assert "No previous sessions" in result
