"""Tests for MCP semantic graph and reasoning tool handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models import SemanticEdge, SemanticNode


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
# engram_add_entity
# ---------------------------------------------------------------------------

class TestEngramAddEntity:
    @pytest.mark.asyncio
    async def test_add_entity_returns_added_message(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_add_entity"]("PostgreSQL", type="Technology")
        assert "Added" in result
        assert "Technology:PostgreSQL" in result

    @pytest.mark.asyncio
    async def test_add_entity_update_existing(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        await tools["engram_add_entity"]("Redis", type="Technology")
        result = await tools["engram_add_entity"]("Redis", type="Technology", attributes={"version": "7"})
        assert "Updated" in result

    @pytest.mark.asyncio
    async def test_add_entity_with_attributes(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        await tools["engram_add_entity"]("AWS", type="Provider", attributes={"region": "us-east-1"})
        nodes = await semantic_graph.query("AWS")
        assert len(nodes) == 1
        assert nodes[0].attributes["region"] == "us-east-1"


# ---------------------------------------------------------------------------
# engram_add_relation
# ---------------------------------------------------------------------------

class TestEngramAddRelation:
    @pytest.mark.asyncio
    async def test_add_relation_returns_message(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_add_relation"](
            from_entity="Team:Backend",
            to_entity="Service:API",
            relation="owns",
        )
        assert "Added" in result or "Updated" in result
        assert "owns" in result

    @pytest.mark.asyncio
    async def test_add_relation_duplicate_returns_updated(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        await tools["engram_add_relation"]("Team:A", "Service:B", relation="uses")
        result = await tools["engram_add_relation"]("Team:A", "Service:B", relation="uses")
        assert "Updated" in result


# ---------------------------------------------------------------------------
# engram_query_graph
# ---------------------------------------------------------------------------

class TestEngramQueryGraph:
    @pytest.mark.asyncio
    async def test_query_by_keyword(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        await semantic_graph.add_node(SemanticNode(type="Technology", name="PostgreSQL"))
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_query_graph"](keyword="Postgres")
        assert "PostgreSQL" in result

    @pytest.mark.asyncio
    async def test_query_no_results_returns_message(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_query_graph"](keyword="nonexistent_zzz")
        assert "No entities found" in result

    @pytest.mark.asyncio
    async def test_query_related_to(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        await semantic_graph.add_node(SemanticNode(type="Team", name="Backend"))
        await semantic_graph.add_node(SemanticNode(type="Service", name="API"))
        await semantic_graph.add_edge(
            SemanticEdge(from_node="Team:Backend", to_node="Service:API", relation="owns")
        )
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_query_graph"](related_to="Backend")
        assert "Backend" in result

    @pytest.mark.asyncio
    async def test_query_by_type_filter(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        await semantic_graph.add_node(SemanticNode(type="Technology", name="Redis"))
        await semantic_graph.add_node(SemanticNode(type="Person", name="Alice"))
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_query_graph"](type="Technology")
        assert "Technology:Redis" in result
        assert "Person:Alice" not in result

    @pytest.mark.asyncio
    async def test_query_related_returns_string(self, semantic_graph):
        from engram.mcp.semantic_tools import register
        tools = _make_mcp_tools(register, lambda: semantic_graph)

        result = await tools["engram_query_graph"](related_to="NonExistent")
        # Returns either "No entities related" or a traversal result string
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# engram_think
# ---------------------------------------------------------------------------

class TestEngramThink:
    @pytest.mark.asyncio
    async def test_think_calls_engine_and_returns_result(self, episodic_store, semantic_graph):
        from engram.mcp.reasoning_tools import register
        mock_engine = AsyncMock()
        mock_engine.think = AsyncMock(return_value="LLM answer about the question")
        tools = _make_mcp_tools(
            register, lambda: mock_engine, lambda: episodic_store, lambda: semantic_graph
        )

        result = await tools["engram_think"]("What happened last week?")
        assert result == "LLM answer about the question"
        mock_engine.think.assert_awaited_once_with("What happened last week?")


# ---------------------------------------------------------------------------
# engram_summarize
# ---------------------------------------------------------------------------

class TestEngramSummarize:
    @pytest.mark.asyncio
    async def test_summarize_delegates_to_engine(self, episodic_store, semantic_graph):
        from engram.mcp.reasoning_tools import register
        mock_engine = AsyncMock()
        mock_engine.summarize = AsyncMock(return_value="Key insights bullet list")
        tools = _make_mcp_tools(
            register, lambda: mock_engine, lambda: episodic_store, lambda: semantic_graph
        )

        result = await tools["engram_summarize"](count=10, save=False)
        assert result == "Key insights bullet list"
        mock_engine.summarize.assert_awaited_once_with(n=10, save=False)

    @pytest.mark.asyncio
    async def test_summarize_with_save_flag(self, episodic_store, semantic_graph):
        from engram.mcp.reasoning_tools import register
        mock_engine = AsyncMock()
        mock_engine.summarize = AsyncMock(return_value="saved summary")
        tools = _make_mcp_tools(
            register, lambda: mock_engine, lambda: episodic_store, lambda: semantic_graph
        )

        result = await tools["engram_summarize"](count=5, save=True)
        assert result == "saved summary"
        mock_engine.summarize.assert_awaited_once_with(n=5, save=True)


# ---------------------------------------------------------------------------
# engram_status
# ---------------------------------------------------------------------------

class TestEngramStatus:
    @pytest.mark.asyncio
    async def test_status_returns_counts(self, episodic_store, semantic_graph):
        from engram.mcp.reasoning_tools import register
        tools = _make_mcp_tools(
            register, lambda: MagicMock(), lambda: episodic_store, lambda: semantic_graph
        )

        result = await tools["engram_status"]()
        assert "Engram Memory Status" in result
        assert "Episodic" in result
        assert "Semantic" in result

    @pytest.mark.asyncio
    async def test_status_shows_node_types_when_present(self, episodic_store, semantic_graph):
        from engram.mcp.reasoning_tools import register
        await semantic_graph.add_node(SemanticNode(type="Technology", name="Postgres"))
        await semantic_graph.add_node(SemanticNode(type="Team", name="Backend"))
        tools = _make_mcp_tools(
            register, lambda: MagicMock(), lambda: episodic_store, lambda: semantic_graph
        )

        result = await tools["engram_status"]()
        assert "Technology" in result
        assert "Team" in result
