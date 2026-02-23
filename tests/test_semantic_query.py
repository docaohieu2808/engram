"""Tests for semantic query helpers: keyword search, BFS traversal, shortest path."""

from __future__ import annotations

import pytest

from engram.models import SemanticEdge, SemanticNode
from engram.semantic.query import query_by_keyword, query_path, query_related


async def test_query_by_keyword(semantic_graph):
    """query_by_keyword delegates to graph.query and returns matching nodes."""
    await semantic_graph.add_node(SemanticNode(type="Service", name="billing-api"))
    await semantic_graph.add_node(SemanticNode(type="Service", name="auth-service"))
    results = await query_by_keyword(semantic_graph, "billing")
    assert len(results) == 1
    assert results[0].name == "billing-api"


async def test_query_related(semantic_graph):
    """query_related BFS returns nodes and edges within depth."""
    svc = SemanticNode(type="Service", name="payment")
    team = SemanticNode(type="Team", name="infra")
    await semantic_graph.add_node(svc)
    await semantic_graph.add_node(team)
    await semantic_graph.add_edge(
        SemanticEdge(from_node="Team:infra", to_node="Service:payment", relation="owns")
    )
    result = await query_related(semantic_graph, "Team:infra", depth=1)
    node_names = [n.name for n in result["nodes"]]
    assert "infra" in node_names
    assert len(result["edges"]) >= 1


async def test_query_path(semantic_graph):
    """query_path returns shortest path between connected nodes."""
    await semantic_graph.add_node(SemanticNode(type="Service", name="frontend"))
    await semantic_graph.add_node(SemanticNode(type="Service", name="backend"))
    await semantic_graph.add_edge(
        SemanticEdge(from_node="Service:frontend", to_node="Service:backend", relation="calls")
    )
    path = await query_path(semantic_graph, "Service:frontend", "Service:backend")
    assert path == ["Service:frontend", "Service:backend"]


async def test_query_path_no_connection(semantic_graph):
    """query_path returns None when nodes are not connected."""
    await semantic_graph.add_node(SemanticNode(type="Service", name="isolated-a"))
    await semantic_graph.add_node(SemanticNode(type="Service", name="isolated-b"))
    path = await query_path(semantic_graph, "Service:isolated-a", "Service:isolated-b")
    assert path is None
