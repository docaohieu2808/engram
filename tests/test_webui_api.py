"""Tests for WebUI API endpoints (memory CRUD, graph mutations, audit, scheduler, UI routes)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from engram.capture.server import create_app
from engram.models import EpisodicMemory, MemoryType, SemanticEdge, SemanticNode


# --- Helpers ---

def _make_memory(content: str = "test memory", id: str = "mem-001", memory_type=MemoryType.FACT, priority=5) -> EpisodicMemory:
    return EpisodicMemory(id=id, content=content, memory_type=memory_type, priority=priority, confidence=0.8, tags=["test"], entities=["entity1"])


def _make_node(name: str = "TestNode", type_: str = "Technology") -> SemanticNode:
    return SemanticNode(type=type_, name=name, attributes={"lang": "python"})


def _make_edge() -> SemanticEdge:
    return SemanticEdge(from_node="Technology:Python", to_node="Project:Engram", relation="uses", weight=1.0)


# --- Fixtures ---

@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.remember = AsyncMock(return_value="mem-new-id")
    ep.search = AsyncMock(return_value=[_make_memory()])
    ep.get_recent = AsyncMock(return_value=[_make_memory(), _make_memory("second", id="mem-002")])
    ep.get = AsyncMock(return_value=_make_memory())
    ep.delete = AsyncMock(return_value=True)
    ep.update_metadata = AsyncMock(return_value=True)
    ep.cleanup_expired = AsyncMock(return_value=3)
    ep.stats = AsyncMock(return_value={"count": 10})
    return ep


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    g.query = AsyncMock(return_value=[_make_node()])
    g.get_related = AsyncMock(return_value={})
    g.get_nodes = AsyncMock(return_value=[_make_node(), _make_node("Engram", "Project")])
    g.get_edges = AsyncMock(return_value=[_make_edge()])
    g.add_node = AsyncMock(return_value=True)
    g.add_edge = AsyncMock(return_value=True)
    g.remove_node = AsyncMock(return_value=True)
    g.remove_edge = AsyncMock(return_value=True)
    g.stats = AsyncMock(return_value={"node_count": 2, "edge_count": 1, "node_types": {"Technology": 1, "Project": 1}})
    return g


@pytest.fixture
def mock_engine():
    eng = AsyncMock()
    eng.think = AsyncMock(return_value="LLM answer")
    eng.summarize = AsyncMock(return_value="Summary")
    return eng


@pytest.fixture
def client(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    return TestClient(app, follow_redirects=False)


# --- Memory CRUD tests ---

def test_list_memories_pagination(client, mock_episodic):
    resp = client.get("/api/v1/memories?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "memories" in data
    assert "total" in data


def test_list_memories_with_search(client, mock_episodic):
    resp = client.get("/api/v1/memories?search=test&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    mock_episodic.search.assert_called()


def test_list_memories_with_type_filter(client, mock_episodic):
    resp = client.get("/api/v1/memories?memory_type=fact")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_get_memory_by_id(client, mock_episodic):
    resp = client.get("/api/v1/memories/mem-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["memory"]["id"] == "mem-001"


def test_get_memory_not_found(client, mock_episodic):
    mock_episodic.get = AsyncMock(return_value=None)
    resp = client.get("/api/v1/memories/nonexistent")
    assert resp.status_code == 404


def test_update_memory(client, mock_episodic):
    resp = client.put("/api/v1/memories/mem-001", json={"memory_type": "preference", "priority": 8, "tags": ["new"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    mock_episodic.update_metadata.assert_called_once()


def test_update_memory_invalid_priority(client, mock_episodic):
    resp = client.put("/api/v1/memories/mem-001", json={"priority": 15})
    assert resp.status_code == 422  # Validation error


def test_delete_memory(client, mock_episodic):
    resp = client.delete("/api/v1/memories/mem-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["deleted"] == "mem-001"


def test_delete_memory_not_found(client, mock_episodic):
    mock_episodic.delete = AsyncMock(return_value=False)
    resp = client.delete("/api/v1/memories/nonexistent")
    assert resp.status_code == 404


def test_bulk_delete_memories(client, mock_episodic):
    resp = client.post("/api/v1/memories/bulk-delete", json={"ids": ["mem-001", "mem-002"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["count"] == 2


def test_bulk_delete_empty_ids(client):
    resp = client.post("/api/v1/memories/bulk-delete", json={"ids": []})
    assert resp.status_code == 422


def test_export_memories(client, mock_episodic):
    resp = client.get("/api/v1/memories/export")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "memories" in data
    assert "count" in data


# --- Graph CRUD tests ---

def test_create_node(client, mock_graph):
    resp = client.post("/api/v1/graph/nodes", json={"type": "Person", "name": "Alice", "attributes": {"role": "dev"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["key"] == "Person:Alice"


def test_create_node_missing_fields(client):
    resp = client.post("/api/v1/graph/nodes", json={"type": "Person"})
    assert resp.status_code == 422


def test_update_node(client, mock_graph):
    resp = client.put("/api/v1/graph/nodes/Technology:TestNode", json={"attributes": {"version": "3.11"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_delete_node(client, mock_graph):
    resp = client.delete("/api/v1/graph/nodes/Technology:TestNode")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_delete_node_not_found(client, mock_graph):
    mock_graph.remove_node = AsyncMock(return_value=False)
    resp = client.delete("/api/v1/graph/nodes/Nonexistent:Node")
    assert resp.status_code == 404


def test_create_edge(client, mock_graph):
    resp = client.post("/api/v1/graph/edges", json={"from_node": "Technology:Python", "to_node": "Project:Engram", "relation": "powers"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_create_edge_missing_fields(client):
    resp = client.post("/api/v1/graph/edges", json={"from_node": "A"})
    assert resp.status_code == 422


def test_delete_edge(client, mock_graph):
    resp = client.request("DELETE", "/api/v1/graph/edges", json={"key": "A--uses-->B"})
    assert resp.status_code == 200


def test_delete_edge_not_found(client, mock_graph):
    mock_graph.remove_edge = AsyncMock(return_value=False)
    resp = client.request("DELETE", "/api/v1/graph/edges", json={"key": "nonexistent"})
    assert resp.status_code == 404


# --- Graph data endpoint ---

def test_graph_data_returns_nodes_and_edges(client, mock_graph):
    resp = client.get("/api/v1/graph/data")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2


# --- Feedback & Audit tests ---

def test_feedback_history(client):
    with patch("engram.audit.get_audit") as mock_audit:
        mock_audit.return_value.read_recent = MagicMock(return_value=[])
        resp = client.get("/api/v1/feedback/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "entries" in data


def test_audit_log(client):
    with patch("engram.audit.get_audit") as mock_audit:
        mock_audit.return_value.read_recent = MagicMock(return_value=[
            {"timestamp": "2026-01-01T00:00:00Z", "operation": "remember", "resource_id": "x", "details": {}}
        ])
        resp = client.get("/api/v1/audit/log?last=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["entries"]) == 1


# --- Scheduler tests ---

def test_list_scheduler_tasks(client):
    resp = client.get("/api/v1/scheduler/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "tasks" in data


def test_force_run_task(client):
    resp = client.post("/api/v1/scheduler/tasks/cleanup_expired/run")
    assert resp.status_code == 200


# --- Benchmark ---

def test_benchmark_run(client, mock_episodic):
    questions = [{"question": "What is test?", "expected": "test memory"}]
    resp = client.post("/api/v1/benchmark/run", json={"questions": questions})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "accuracy" in data
    assert "results" in data
    assert len(data["results"]) == 1


def test_benchmark_empty_questions(client):
    resp = client.post("/api/v1/benchmark/run", json={"questions": []})
    assert resp.status_code == 422


# --- UI routes ---

def test_ui_returns_html(client):
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Engram" in resp.text


def test_ui_catchall_returns_html(client):
    resp = client.get("/ui/memories")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_graph_page_returns_html(client):
    resp = client.get("/graph")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
