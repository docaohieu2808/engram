"""Tests for Graph UI: /api/v1/graph/data endpoint and /graph HTML page."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from engram.capture.server import create_app
from engram.models import SemanticEdge, SemanticNode


# --- Helpers ---

def _make_node(name: str, node_type: str = "Technology", attrs: dict | None = None) -> SemanticNode:
    return SemanticNode(type=node_type, name=name, attributes=attrs or {})


def _make_edge(from_node: str, to_node: str, relation: str = "uses") -> SemanticEdge:
    return SemanticEdge(from_node=from_node, to_node=to_node, relation=relation)


# --- Fixtures ---

@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.stats = AsyncMock(return_value={"count": 0})
    return ep


@pytest.fixture
def mock_engine():
    eng = AsyncMock()
    return eng


@pytest.fixture
def mock_graph_with_data():
    """Graph with 3 nodes and 2 edges."""
    g = AsyncMock()
    nodes = [
        _make_node("Alice", "Person", {"role": "engineer"}),
        _make_node("Python", "Technology"),
        _make_node("Engram", "Project", {"status": "active"}),
    ]
    edges = [
        _make_edge(nodes[0].key, nodes[1].key, "knows"),
        _make_edge(nodes[0].key, nodes[2].key, "maintains"),
    ]
    g.get_nodes = AsyncMock(return_value=nodes)
    g.get_edges = AsyncMock(return_value=edges)
    g.stats = AsyncMock(return_value={"node_count": 3, "edge_count": 2})
    return g


@pytest.fixture
def mock_graph_empty():
    """Graph with no nodes or edges."""
    g = AsyncMock()
    g.get_nodes = AsyncMock(return_value=[])
    g.get_edges = AsyncMock(return_value=[])
    g.stats = AsyncMock(return_value={"node_count": 0, "edge_count": 0})
    return g


@pytest.fixture
def mock_graph_duplicate_edges():
    """Graph where duplicate edges should be deduplicated."""
    g = AsyncMock()
    node_a = _make_node("A", "Technology")
    node_b = _make_node("B", "Technology")
    # Two edges with same (from, to, relation) — should deduplicate to one
    dup_edge_1 = _make_edge(node_a.key, node_b.key, "connects")
    dup_edge_2 = _make_edge(node_a.key, node_b.key, "connects")
    dup_edge_2._key = dup_edge_1.key  # same relation
    different_edge = _make_edge(node_a.key, node_b.key, "depends_on")
    g.get_nodes = AsyncMock(return_value=[node_a, node_b])
    g.get_edges = AsyncMock(return_value=[dup_edge_1, dup_edge_2, different_edge])
    g.stats = AsyncMock(return_value={"node_count": 2, "edge_count": 3})
    return g


@pytest.fixture
def client_with_data(mock_episodic, mock_graph_with_data, mock_engine):
    app = create_app(mock_episodic, mock_graph_with_data, mock_engine)
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def client_empty(mock_episodic, mock_graph_empty, mock_engine):
    app = create_app(mock_episodic, mock_graph_empty, mock_engine)
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def client_dup_edges(mock_episodic, mock_graph_duplicate_edges, mock_engine):
    app = create_app(mock_episodic, mock_graph_duplicate_edges, mock_engine)
    return TestClient(app, follow_redirects=False)


# --- Tests ---

class TestGraphDataEndpoint:
    def test_graph_data_returns_nodes_and_edges(self, client_with_data):
        resp = client_with_data.get("/api/v1/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

    def test_graph_data_node_has_required_fields(self, client_with_data):
        resp = client_with_data.get("/api/v1/graph/data")
        nodes = resp.json()["nodes"]
        for node in nodes:
            assert "id" in node
            assert "label" in node
            assert "group" in node
            assert "color" in node
            assert "title" in node

    def test_graph_data_edge_has_required_fields(self, client_with_data):
        resp = client_with_data.get("/api/v1/graph/data")
        edges = resp.json()["edges"]
        for edge in edges:
            assert "from" in edge
            assert "to" in edge
            assert "label" in edge
            assert edge["arrows"] == "to"

    def test_graph_data_empty_graph(self, client_empty):
        resp = client_empty.get("/api/v1/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_graph_data_deduplicates_edges(self, client_dup_edges):
        resp = client_dup_edges.get("/api/v1/graph/data")
        assert resp.status_code == 200
        edges = resp.json()["edges"]
        # Two distinct relations: "connects" (deduped from 2) + "depends_on" = 2
        assert len(edges) == 2
        relations = {e["label"] for e in edges}
        assert "connects" in relations
        assert "depends_on" in relations

    def test_graph_data_color_assigned_by_type(self, client_with_data):
        resp = client_with_data.get("/api/v1/graph/data")
        nodes = resp.json()["nodes"]
        color_by_label = {n["label"]: n["color"] for n in nodes}
        assert color_by_label["Alice"] == "#4CAF50"      # Person
        assert color_by_label["Python"] == "#2196F3"     # Technology
        assert color_by_label["Engram"] == "#FF9800"     # Project

    def test_graph_data_unknown_type_gets_default_color(self, mock_episodic, mock_engine):
        g = AsyncMock()
        g.get_nodes = AsyncMock(return_value=[_make_node("X", "UnknownType")])
        g.get_edges = AsyncMock(return_value=[])
        g.stats = AsyncMock(return_value={"node_count": 1, "edge_count": 0})
        app = create_app(mock_episodic, g, mock_engine)
        client = TestClient(app, follow_redirects=False)
        resp = client.get("/api/v1/graph/data")
        nodes = resp.json()["nodes"]
        assert nodes[0]["color"] == "#607D8B"


class TestGraphUIEndpoint:
    def test_graph_html_file_exists(self):
        html_path = Path(__file__).parent.parent / "src" / "engram" / "static" / "graph.html"
        assert html_path.exists(), f"graph.html not found at {html_path}"

    def test_graph_ui_endpoint_returns_html(self, client_with_data):
        resp = client_with_data.get("/graph")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "vis-network" in resp.text
        assert "Engram" in resp.text

    def test_graph_ui_has_fetch_to_api(self, client_with_data):
        resp = client_with_data.get("/graph")
        assert "/api/v1/graph/data" in resp.text
