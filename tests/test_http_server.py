"""Tests for HTTP webhook server (capture/server.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from engram.capture.server import create_app
from engram.models import EpisodicMemory, MemoryType, SemanticNode


# --- Helpers ---

def _make_memory(content: str = "test memory") -> EpisodicMemory:
    return EpisodicMemory(id="abc123", content=content, memory_type=MemoryType.FACT)


def _make_node(name: str = "TestNode") -> SemanticNode:
    return SemanticNode(type="Technology", name=name)


# --- Fixtures ---

@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.remember = AsyncMock(return_value="mem-id-123")
    ep.search = AsyncMock(return_value=[_make_memory()])
    ep.cleanup_expired = AsyncMock(return_value=3)
    ep.stats = AsyncMock(return_value={"count": 5})
    return ep


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    g.query = AsyncMock(return_value=[_make_node()])
    g.get_related = AsyncMock(return_value={})
    g.stats = AsyncMock(return_value={"node_count": 2, "edge_count": 1})
    return g


@pytest.fixture
def mock_engine():
    eng = AsyncMock()
    eng.think = AsyncMock(return_value="LLM answer")
    eng.summarize = AsyncMock(return_value="Summary text")
    return eng


@pytest.fixture
def client(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    # follow_redirects=False so we can assert 301 behaviour directly
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def client_follow(mock_episodic, mock_graph, mock_engine):
    """Client that follows redirects — for testing v1 endpoints via legacy paths."""
    app = create_app(mock_episodic, mock_graph, mock_engine)
    return TestClient(app, follow_redirects=True)


# --- Tests: /health (root, no prefix) ---

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- Tests: /api/v1/status ---

def test_status_returns_stats(client, mock_episodic, mock_graph):
    resp = client.get("/api/v1/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "episodic" in data
    assert "semantic" in data
    assert data["episodic"]["count"] == 5
    assert data["semantic"]["node_count"] == 2


# --- Tests: POST /api/v1/remember ---

def test_remember_returns_200_and_id(client, mock_episodic):
    payload = {"content": "new memory", "memory_type": "fact", "priority": 5}
    resp = client.post("/api/v1/remember", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["id"] == "mem-id-123"


def test_remember_with_tags_and_entities(client, mock_episodic):
    payload = {
        "content": "tagged memory",
        "memory_type": "decision",
        "priority": 7,
        "tags": ["important"],
        "entities": ["Service:API"],
    }
    resp = client.post("/api/v1/remember", json=payload)
    assert resp.status_code == 200
    mock_episodic.remember.assert_awaited_once()
    call_kwargs = mock_episodic.remember.call_args.kwargs
    assert call_kwargs["tags"] == ["important"]
    assert call_kwargs["entities"] == ["Service:API"]


def test_remember_invalid_memory_type_returns_422(client):
    payload = {"content": "test", "memory_type": "invalid_type"}
    resp = client.post("/api/v1/remember", json=payload)
    assert resp.status_code == 422
    # Should be structured error response
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"


# --- Tests: GET /api/v1/recall ---

def test_recall_returns_results(client, mock_episodic):
    resp = client.get("/api/v1/recall", params={"query": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1
    assert data["offset"] == 0
    assert data["limit"] == 5


def test_recall_pagination_offset(client, mock_episodic):
    mock_episodic.search = AsyncMock(return_value=[
        _make_memory(f"mem {i}") for i in range(3)
    ])
    resp = client.get("/api/v1/recall", params={"query": "test", "offset": 2, "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["offset"] == 2


def test_recall_with_memory_type_filter(client, mock_episodic):
    resp = client.get("/api/v1/recall", params={"query": "test", "memory_type": "fact"})
    assert resp.status_code == 200
    mock_episodic.search.assert_awaited_once()
    call_kwargs = mock_episodic.search.call_args.kwargs
    assert call_kwargs["filters"] == {"memory_type": "fact"}


def test_recall_with_tags_filter(client, mock_episodic):
    resp = client.get("/api/v1/recall", params={"query": "test", "tags": "foo,bar"})
    assert resp.status_code == 200
    call_kwargs = mock_episodic.search.call_args.kwargs
    assert call_kwargs["tags"] == ["foo", "bar"]


def test_recall_missing_query_returns_422(client):
    resp = client.get("/api/v1/recall")
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"


# --- Tests: GET /api/v1/query ---

def test_query_by_keyword(client, mock_graph):
    resp = client.get("/api/v1/query", params={"keyword": "postgres"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["results"], list)
    assert data["total"] == 1


def test_query_with_related_to(client, mock_graph):
    mock_graph.get_related = AsyncMock(return_value={
        "Service:API": {"nodes": [_make_node()], "edges": []}
    })
    resp = client.get("/api/v1/query", params={"keyword": "api", "related_to": "Service:API"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_query_pagination(client, mock_graph):
    mock_graph.query = AsyncMock(return_value=[_make_node(f"Node{i}") for i in range(10)])
    resp = client.get("/api/v1/query", params={"keyword": "node", "offset": 5, "limit": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert data["total"] == 10


# --- Tests: POST /api/v1/think ---

def test_think_returns_answer(client, mock_engine):
    resp = client.post("/api/v1/think", json={"question": "What happened?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["answer"] == "LLM answer"


# --- Tests: POST /api/v1/cleanup ---

def test_cleanup_returns_deleted_count(client, mock_episodic):
    resp = client.post("/api/v1/cleanup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["deleted"] == 3


# --- Tests: POST /api/v1/summarize ---

def test_summarize_returns_summary(client, mock_engine):
    resp = client.post("/api/v1/summarize", json={"count": 10, "save": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["summary"] == "Summary text"


def test_summarize_with_save(client, mock_engine):
    resp = client.post("/api/v1/summarize", json={"count": 5, "save": True})
    assert resp.status_code == 200
    mock_engine.summarize.assert_awaited_once_with(n=5, save=True)


# --- Tests: POST /api/v1/ingest ---

def test_ingest_without_fn_returns_error(mock_episodic, mock_graph, mock_engine):
    """When no ingest_fn is provided, /api/v1/ingest returns error status."""
    app = create_app(mock_episodic, mock_graph, mock_engine, ingest_fn=None)
    c = TestClient(app, follow_redirects=False)
    resp = c.post("/api/v1/ingest", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_ingest_with_fn_calls_it(mock_episodic, mock_graph, mock_engine):
    from engram.models import IngestResult
    ingest_fn = AsyncMock(return_value=IngestResult(episodic_count=1))
    app = create_app(mock_episodic, mock_graph, mock_engine, ingest_fn=ingest_fn)
    c = TestClient(app, follow_redirects=False)
    msgs = [{"role": "user", "content": "hello"}]
    resp = c.post("/api/v1/ingest", json={"messages": msgs})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    ingest_fn.assert_awaited_once_with(msgs)


# --- Tests: Legacy 301 redirects ---

def test_legacy_recall_redirects_to_v1(client):
    resp = client.get("/recall", params={"query": "test"})
    assert resp.status_code == 301
    assert "/api/v1/recall" in resp.headers["location"]


def test_legacy_query_redirects_to_v1(client):
    resp = client.get("/query", params={"keyword": "x"})
    assert resp.status_code == 301
    assert "/api/v1/query" in resp.headers["location"]


def test_legacy_status_redirects_to_v1(client):
    resp = client.get("/status")
    assert resp.status_code == 301
    assert "/api/v1/status" in resp.headers["location"]


def test_legacy_remember_redirects_to_v1(client):
    resp = client.post("/remember", json={"content": "x"})
    assert resp.status_code == 301
    assert "/api/v1/remember" in resp.headers["location"]


def test_legacy_think_redirects_to_v1(client):
    resp = client.post("/think", json={"question": "x"})
    assert resp.status_code == 301
    assert "/api/v1/think" in resp.headers["location"]


def test_legacy_redirect_preserves_query_string(client):
    resp = client.get("/recall", params={"query": "hello", "limit": "3"})
    assert resp.status_code == 301
    loc = resp.headers["location"]
    assert "query=hello" in loc
    assert "limit=3" in loc


# --- Tests: Error response format ---

def test_validation_error_has_structured_format(client):
    """Missing required 'content' field → structured VALIDATION_ERROR."""
    resp = client.post("/api/v1/remember", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "message" in body["error"]
    assert "details" in body["error"]


def test_recall_missing_query_has_structured_error(client):
    resp = client.get("/api/v1/recall")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["details"], dict)
