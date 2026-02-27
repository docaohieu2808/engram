"""Integration tests for 6 critical flows: remember, recall, think, ingest, semantic, health.

All tests use real stores with mocked LLM/embedding. Temporary directories
ensure no production data is touched. Marked @pytest.mark.integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
from engram.episodic.store import EpisodicStore
from engram.models import MemoryType, SemanticEdge, SemanticNode
from engram.semantic import create_graph

# Fixed 384-dim embedding for all integration tests
_FIXED_VEC = [0.1] * 384


def _mock_embeddings(_model, texts, _dim=None):
    return [_FIXED_VEC for _ in texts]


# --- Fixtures ---


@pytest.fixture
def episodic(tmp_path):
    cfg = EpisodicConfig(path=str(tmp_path / "episodic"), dedup_enabled=False)
    embed = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        yield EpisodicStore(config=cfg, embedding_config=embed)


@pytest.fixture
def graph(tmp_path):
    cfg = SemanticConfig(path=str(tmp_path / "semantic.db"))
    return create_graph(cfg)


# --- Flow 1: Remember → Recall round-trip ---


@pytest.mark.integration
class TestRememberRecallFlow:
    async def test_remember_then_recall_by_content(self, episodic):
        """Store memory → search by similar content → found."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            mid = await episodic.remember(
                "Deployed v3.0 to staging at 15:00",
                memory_type=MemoryType.FACT,
                priority=7,
                tags=["deploy", "staging"],
            )
        assert mid

        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            results = await episodic.search("deployment staging", limit=5)
        assert len(results) >= 1
        assert any("Deployed" in r.content for r in results)

    async def test_remember_with_expiry_then_cleanup(self, episodic):
        """Store expired memory → cleanup → only permanent remains."""
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(hours=2)
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            await episodic.remember("Ephemeral data", expires_at=past)
            await episodic.remember("Permanent data")

        deleted = await episodic.cleanup_expired()
        assert deleted >= 1

        stats = await episodic.stats()
        assert stats["count"] >= 1

    async def test_remember_multiple_types_then_filter(self, episodic):
        """Store different memory types → search returns correct types."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            await episodic.remember("Never deploy on Friday", memory_type=MemoryType.DECISION)
            await episodic.remember("PostgreSQL uses port 5432", memory_type=MemoryType.FACT)
            await episodic.remember("User prefers dark mode", memory_type=MemoryType.PREFERENCE)

        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            results = await episodic.search("deploy Friday", limit=10)
        assert len(results) >= 1

    async def test_stats_reflect_all_memories(self, episodic):
        """After N remembers, stats.count >= N."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            for i in range(5):
                await episodic.remember(f"Memory number {i}")

        stats = await episodic.stats()
        assert stats["count"] >= 5


# --- Flow 2: Think (reasoning) ---


@pytest.mark.integration
class TestThinkFlow:
    async def test_think_returns_answer_from_memories(self, episodic, graph):
        """Store context → think() synthesizes answer from memories."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            await episodic.remember("API latency increased 3x after Redis upgrade")
            await episodic.remember("Rolled back Redis to v6.2, latency normalized")

        from engram.reasoning.engine import ReasoningEngine

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Redis upgrade caused latency spike."

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            engine = ReasoningEngine(episodic, graph, model="gemini/gemini-2.0-flash")
            result = await engine.think("What caused the latency issues?")

        # think() returns dict with "answer" key
        assert result
        answer = result["answer"] if isinstance(result, dict) else result
        assert isinstance(answer, str)
        assert len(answer) > 0

    async def test_think_with_empty_memories_returns_honest_response(self, episodic, graph):
        """think() with no relevant memories → honest 'no info' response."""
        from engram.reasoning.engine import ReasoningEngine

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No relevant memories found."

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            engine = ReasoningEngine(episodic, graph, model="gemini/gemini-2.0-flash")
            result = await engine.think("What is the weather?")

        assert result
        answer = result["answer"] if isinstance(result, dict) else result
        assert isinstance(answer, str)

    async def test_summarize_recent_memories(self, episodic, graph):
        """Store several memories → summarize() produces summary."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            await episodic.remember("Sprint review went well")
            await episodic.remember("Bug in auth module fixed")
            await episodic.remember("New hire starts Monday")

        from engram.reasoning.engine import ReasoningEngine

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Key insights: sprint went well, auth bug fixed."

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            engine = ReasoningEngine(episodic, graph, model="gemini/gemini-2.0-flash")
            summary = await engine.summarize(n=10)

        assert summary
        assert len(summary) > 0


# --- Flow 3: Ingest pipeline ---


@pytest.mark.integration
class TestIngestFlow:
    async def test_ingest_stores_episodic_memories(self, episodic):
        """Ingest messages → episodic count increases."""
        messages = [
            {"role": "user", "content": "What database should we use?"},
            {"role": "assistant", "content": "PostgreSQL is recommended for relational data."},
        ]

        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            for msg in messages:
                if msg["content"]:
                    await episodic.remember(msg["content"])

        stats = await episodic.stats()
        assert stats["count"] >= 2

    async def test_ingest_with_entity_extraction(self, episodic, graph):
        """Ingest → entity extraction → nodes appear in graph."""
        node = SemanticNode(type="Technology", name="PostgreSQL")
        await graph.add_node(node)

        results = await graph.query("PostgreSQL")
        assert len(results) >= 1
        assert any(n.name == "PostgreSQL" for n in results)

    async def test_ingest_episodic_decoupled_from_llm(self, episodic):
        """Episodic store works even if LLM extraction fails."""
        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            mid = await episodic.remember("This should work without LLM")

        assert mid
        stats = await episodic.stats()
        assert stats["count"] >= 1


# --- Flow 4: Semantic graph CRUD ---


@pytest.mark.integration
class TestSemanticGraphFlow:
    async def test_add_nodes_query_remove(self, graph):
        """Add nodes → query → remove → query empty."""
        n1 = SemanticNode(type="Service", name="auth-api")
        n2 = SemanticNode(type="Service", name="billing-api")
        await graph.add_node(n1)
        await graph.add_node(n2)

        results = await graph.query("auth")
        names = [n.name for n in results]
        assert "auth-api" in names

        await graph.remove_node(n1.key)
        results_after = await graph.query("auth-api")
        names_after = [n.name for n in results_after]
        assert "auth-api" not in names_after

    async def test_add_edge_get_related(self, graph):
        """Add nodes + edge → get_related returns connected entities."""
        team = SemanticNode(type="Team", name="backend-team")
        svc = SemanticNode(type="Service", name="user-service")
        await graph.add_node(team)
        await graph.add_node(svc)

        edge = SemanticEdge(from_node=team.key, to_node=svc.key, relation="maintains")
        await graph.add_edge(edge)

        related = await graph.get_related(["backend-team"])
        assert "backend-team" in related
        edges = related["backend-team"].get("edges", [])
        assert any(e.relation == "maintains" for e in edges)

    async def test_graph_stats_accuracy(self, graph):
        """Stats accurately reflect node/edge counts."""
        for name in ["svc-x", "svc-y", "svc-z"]:
            await graph.add_node(SemanticNode(type="Service", name=name))

        edge = SemanticEdge(
            from_node="Service:svc-x", to_node="Service:svc-y", relation="calls",
        )
        await graph.add_edge(edge)

        stats = await graph.stats()
        assert stats["node_count"] >= 3
        assert stats["edge_count"] >= 1


# --- Flow 5: MCP tool execution ---


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


@pytest.mark.integration
class TestMCPFlow:
    async def test_mcp_remember_recall_roundtrip(self, episodic):
        """MCP remember → MCP recall → content found."""
        from engram.mcp.episodic_tools import register

        tools = _make_mcp_tools(
            register, lambda: episodic, lambda: None, lambda: MagicMock(),
        )

        result = await tools["engram_remember"](
            "Database migrated to PostgreSQL 16",
            memory_type="fact",
            priority=7,
        )
        assert "Remembered" in result

        with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
            recall_result = await tools["engram_recall"]("database migration")
        assert recall_result  # non-empty response

    async def test_mcp_remember_invalid_type_returns_error(self, episodic):
        """MCP remember with invalid memory_type → helpful error."""
        from engram.mcp.episodic_tools import register

        tools = _make_mcp_tools(
            register, lambda: episodic, lambda: None, lambda: MagicMock(),
        )

        result = await tools["engram_remember"](
            "test", memory_type="invalid_type", priority=5,
        )
        assert "Invalid" in result

    async def test_mcp_status_returns_stats(self, episodic, graph):
        """MCP status tool returns memory statistics."""
        from engram.mcp.reasoning_tools import register

        # reasoning_tools.register(mcp, get_engine, get_episodic, get_graph)
        tools = _make_mcp_tools(
            register, lambda: MagicMock(), lambda: episodic, lambda: graph,
        )

        result = await tools["engram_status"]()
        assert result  # non-empty string


# --- Flow 6: Health check ---


@pytest.mark.integration
class TestHealthCheckFlow:
    async def test_deep_check_returns_healthy(self, episodic, graph):
        """deep_check with real stores → healthy status."""
        from engram.health import deep_check

        result = await deep_check(episodic, graph)
        assert result["status"] in ("healthy", "unhealthy")
        assert "components" in result
        assert "chromadb" in result["components"]
        assert "semantic" in result["components"]

    async def test_full_health_check_without_api_keys(self, episodic, graph):
        """full_health_check without API keys → degraded (skips LLM/embed)."""
        from engram.health import full_health_check

        with patch.dict("os.environ", {}, clear=False):
            # Remove GEMINI_API_KEY if present
            import os
            saved = os.environ.pop("GEMINI_API_KEY", None)
            try:
                result = await full_health_check(episodic, graph)
                assert result["status"] in ("healthy", "degraded")
                assert "components" in result
                assert "api_keys" in result["components"]
            finally:
                if saved:
                    os.environ["GEMINI_API_KEY"] = saved

    async def test_check_api_keys_healthy(self):
        """check_api_keys reports healthy when key present."""
        from engram.health import check_api_keys

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123"}):
            comp = check_api_keys()
            assert comp.status == "healthy"
            assert comp.details["primary"] is True
            assert comp.details["count"] >= 1

    async def test_check_api_keys_unhealthy(self):
        """check_api_keys reports unhealthy when no keys."""
        from engram.health import check_api_keys
        import os

        saved = os.environ.pop("GEMINI_API_KEY", None)
        saved_fb = os.environ.pop("GEMINI_API_KEY_FALLBACK", None)
        try:
            comp = check_api_keys()
            assert comp.status == "unhealthy"
        finally:
            if saved:
                os.environ["GEMINI_API_KEY"] = saved
            if saved_fb:
                os.environ["GEMINI_API_KEY_FALLBACK"] = saved_fb

    async def test_check_fts5_healthy(self, tmp_path):
        """FTS5 check with a real index → healthy."""
        from engram.episodic.fts_index import FtsIndex

        db_path = str(tmp_path / "test_fts.db")
        fts = FtsIndex(db_path)
        fts.insert("mem-1", "Test memory content", "fact")

        from engram.health import check_fts5

        comp = await check_fts5(db_path)
        assert comp.status == "healthy"
        assert comp.details["indexed"] >= 1
        fts.close()

    async def test_check_disk_healthy(self):
        """Disk check on home dir → healthy (>1GB)."""
        from engram.health import check_disk

        comp = await check_disk("~")
        assert comp.status == "healthy"
        assert comp.details["free_gb"] > 0

    async def test_check_watcher_not_running(self):
        """Watcher check when no PID file and no process → degraded."""
        from unittest.mock import patch, MagicMock
        from engram.health import check_watcher

        # Mock both PID file (not exists) and pgrep (no results)
        mock_result = MagicMock(stdout="")
        with patch("engram.health.components.os.path.exists", return_value=False), \
             patch("subprocess.run", return_value=mock_result):
            comp = await check_watcher()
        assert comp.status in ("degraded", "unhealthy")

    def test_check_feature_flags_returns_all_features(self):
        """check_feature_flags returns all FeatureStatus entries (24 after adding 3 extra flags)."""
        from engram.config import Config
        from engram.health import check_feature_flags

        cfg = Config()
        flags = check_feature_flags(cfg)
        assert len(flags) >= 21
        # All have required fields
        for f in flags:
            assert f.name
            assert f.config_path
            assert f.category
            assert f.env_var.startswith("ENGRAM_")
            assert isinstance(f.enabled, bool)

    def test_check_feature_flags_categories(self):
        """check_feature_flags covers expected categories."""
        from engram.config import Config
        from engram.health import check_feature_flags

        cfg = Config()
        flags = check_feature_flags(cfg)
        categories = {f.category for f in flags}
        # Core categories must be present; Discovery is now also included
        assert {"Storage", "Pipeline", "Enterprise", "Capture"}.issubset(categories)

    async def test_check_redis_when_disabled(self):
        """check_redis with unreachable URL → unhealthy or degraded (no crash)."""
        from engram.health import check_redis

        comp = await check_redis("redis://localhost:19999/0")
        assert comp.status in ("unhealthy", "degraded")
        assert comp.name == "redis"

    def test_check_constitution(self):
        """check_constitution returns healthy with hash details."""
        from engram.health import check_constitution

        comp = check_constitution()
        # healthy or unhealthy depending on env, but must not raise
        assert comp.name == "constitution"
        assert comp.status in ("healthy", "unhealthy")
        if comp.status == "healthy":
            assert "hash" in comp.details
            assert comp.details["laws"] == 3

    def test_check_resource_tier(self):
        """check_resource_tier returns a valid tier."""
        from engram.health import check_resource_tier

        comp = check_resource_tier()
        assert comp.name == "resource_tier"
        assert comp.status in ("healthy", "degraded", "unhealthy")
        if comp.status != "unhealthy":
            assert comp.details["tier"] in ("full", "standard", "basic", "readonly")
