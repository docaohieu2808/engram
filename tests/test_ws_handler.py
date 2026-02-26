"""Tests for WebSocket handler endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from engram.capture.server import create_app
from engram.models import EpisodicMemory, MemoryType, SemanticNode
from engram.ws.event_bus import event_bus


def _make_memory(content: str = "test memory") -> EpisodicMemory:
    return EpisodicMemory(id="abc123", content=content, memory_type=MemoryType.FACT)


def _make_node(name: str = "TestNode") -> SemanticNode:
    return SemanticNode(type="Technology", name=name)


@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.remember = AsyncMock(return_value="mem-id-123")
    ep.search = AsyncMock(return_value=[_make_memory()])
    ep.stats = AsyncMock(return_value={"count": 5})
    return ep


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    g.query = AsyncMock(return_value=[_make_node()])
    g.stats = AsyncMock(return_value={"node_count": 2, "edge_count": 1})
    return g


@pytest.fixture
def mock_engine():
    eng = AsyncMock()
    eng.think = AsyncMock(return_value={"answer": "LLM answer", "degraded": False})
    return eng


@pytest.fixture
def ws_client(mock_episodic, mock_graph, mock_engine):
    event_bus.clear()
    app = create_app(mock_episodic, mock_graph, mock_engine)
    return TestClient(app)


# --- Connection ---

def test_ws_connect_no_auth(ws_client):
    """Connect without token when auth is disabled (default) — should accept."""
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "status", "id": "1"})
        resp = ws.receive_json()
        assert resp["type"] == "response"
        assert resp["id"] == "1"
        assert resp["status"] == "ok"


# --- Commands ---

def test_ws_remember(ws_client, mock_episodic):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "remember", "id": "r1",
            "payload": {"content": "test memory", "priority": 7},
        })
        resp = ws.receive_json()
        assert resp["status"] == "ok"
        assert resp["data"]["id"] == "mem-id-123"
        mock_episodic.remember.assert_called_once()


def test_ws_recall(ws_client, mock_episodic):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "recall", "id": "q1",
            "payload": {"query": "test", "limit": 3},
        })
        resp = ws.receive_json()
        assert resp["status"] == "ok"
        assert len(resp["data"]["results"]) == 1


def test_ws_think(ws_client, mock_engine):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "think", "id": "t1",
            "payload": {"question": "What do you know?"},
        })
        resp = ws.receive_json()
        assert resp["data"]["answer"] == "LLM answer"
        assert resp["data"]["degraded"] is False


def test_ws_query_graph(ws_client, mock_graph):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "query", "id": "g1",
            "payload": {"keyword": "Test"},
        })
        resp = ws.receive_json()
        assert resp["status"] == "ok"
        assert len(resp["data"]["results"]) == 1


def test_ws_status(ws_client, mock_episodic):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "status", "id": "s1"})
        resp = ws.receive_json()
        assert resp["data"]["episodic"] == {"count": 5}


# --- Error handling ---

def test_ws_unknown_command(ws_client):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "nonexistent", "id": "e1"})
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert resp["code"] == "UNKNOWN_COMMAND"
        # Connection should stay open — send another command
        ws.send_json({"type": "status", "id": "e2"})
        resp2 = ws.receive_json()
        assert resp2["type"] == "response"


def test_ws_malformed_command(ws_client):
    with ws_client.websocket_connect("/ws") as ws:
        # Missing required 'type' field
        ws.send_json({"id": "bad", "payload": {}})
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert "Malformed" in resp["message"]


def test_ws_command_execution_error(ws_client, mock_episodic):
    mock_episodic.remember.side_effect = RuntimeError("DB error")
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "remember", "id": "err1",
            "payload": {"content": "will fail"},
        })
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert resp["code"] == "INTERNAL_ERROR"
        assert "DB error" in resp["message"]


# --- Feedback ---

def test_ws_feedback(ws_client, mock_episodic):
    with ws_client.websocket_connect("/ws") as ws:
        # Mock the feedback module
        import engram.ws.handler as handler_mod
        from unittest.mock import patch
        with patch("engram.feedback.auto_adjust.adjust_memory", new_callable=AsyncMock) as mock_adj:
            mock_adj.return_value = {"memory_id": "m1", "confidence": 0.8}
            ws.send_json({
                "type": "feedback", "id": "f1",
                "payload": {"memory_id": "m1", "feedback": "positive"},
            })
            resp = ws.receive_json()
            assert resp["status"] == "ok"
            assert resp["data"]["confidence"] == 0.8
