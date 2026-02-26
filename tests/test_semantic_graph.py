"""Tests for SemanticGraph node/edge CRUD, traversal, and persistence."""

from __future__ import annotations

import pytest

from engram.models import SemanticEdge, SemanticNode


async def _add_service_team(graph):
    """Helper: add two nodes and one edge."""
    svc = SemanticNode(type="Service", name="api")
    team = SemanticNode(type="Team", name="platform")
    edge = SemanticEdge(from_node="Team:platform", to_node="Service:api", relation="owns")
    await graph.add_node(svc)
    await graph.add_node(team)
    await graph.add_edge(edge)
    return svc, team, edge


async def test_add_node_new(semantic_graph):
    """add_node returns True for a brand-new node."""
    node = SemanticNode(type="Service", name="auth")
    result = await semantic_graph.add_node(node)
    assert result is True


async def test_add_node_update(semantic_graph):
    """add_node returns False when updating an existing node."""
    node = SemanticNode(type="Service", name="auth")
    await semantic_graph.add_node(node)
    result = await semantic_graph.add_node(SemanticNode(type="Service", name="auth", attributes={"version": "2"}))
    assert result is False


async def test_add_edge(semantic_graph):
    """add_edge returns True for a new edge."""
    await semantic_graph.add_node(SemanticNode(type="Service", name="api"))
    await semantic_graph.add_node(SemanticNode(type="Team", name="ops"))
    edge = SemanticEdge(from_node="Team:ops", to_node="Service:api", relation="owns")
    result = await semantic_graph.add_edge(edge)
    assert result is True


async def test_remove_node_cascades_edges(semantic_graph):
    """Removing a node also removes its connected edges."""
    await _add_service_team(semantic_graph)
    await semantic_graph.remove_node("Service:api")
    edges = await semantic_graph.get_edges()
    assert not any(e.to_node == "Service:api" or e.from_node == "Service:api" for e in edges)


async def test_query_by_keyword(semantic_graph):
    """query() finds nodes matching keyword substring in name."""
    await semantic_graph.add_node(SemanticNode(type="Service", name="payment-gateway"))
    await semantic_graph.add_node(SemanticNode(type="Service", name="auth-service"))
    results = await semantic_graph.query("payment")
    assert len(results) == 1
    assert results[0].name.lower() == "payment-gateway"


async def test_get_related(semantic_graph):
    """get_related returns connected nodes via BFS."""
    await _add_service_team(semantic_graph)
    result = await semantic_graph.get_related(["platform"], depth=1)
    nodes = result["platform"]["nodes"]
    node_names = [n.name.lower() for n in nodes]
    assert "api" in node_names or "platform" in node_names


async def test_stats(semantic_graph):
    """stats() reports correct node_count, edge_count, node_types."""
    await _add_service_team(semantic_graph)
    s = await semantic_graph.stats()
    assert s["node_count"] == 2
    assert s["edge_count"] == 1
    assert "Service" in s["node_types"]
    assert "Team" in s["node_types"]


async def test_dump(semantic_graph):
    """dump() returns dicts with 'nodes' and 'edges' keys."""
    await _add_service_team(semantic_graph)
    data = await semantic_graph.dump()
    assert "nodes" in data and "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
