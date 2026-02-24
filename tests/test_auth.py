"""Tests for auth module: JWT, API keys, middleware, RBAC."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from engram.auth import create_api_key, create_jwt, revoke_api_key, verify_api_key, verify_jwt
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.capture.server import create_app
from engram.models import EpisodicMemory, MemoryType, SemanticNode


# --- Fixtures ---

@pytest.fixture
def mock_episodic():
    ep = AsyncMock()
    ep.remember = AsyncMock(return_value="mem-id-123")
    ep.search = AsyncMock(return_value=[EpisodicMemory(id="x", content="c", memory_type=MemoryType.FACT)])
    ep.cleanup_expired = AsyncMock(return_value=3)
    ep.stats = AsyncMock(return_value={"count": 5})
    return ep


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    g.query = AsyncMock(return_value=[SemanticNode(type="Technology", name="Test")])
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
def client_no_auth(mock_episodic, mock_graph, mock_engine):
    """Client with auth disabled (default)."""
    app = create_app(mock_episodic, mock_graph, mock_engine)
    return TestClient(app, follow_redirects=True)


SECRET = "super-secret-key-for-testing-only-32chars"


def _make_config(enabled: bool = True, secret: str = SECRET):
    from engram.config import AuthConfig, Config
    cfg = Config()
    cfg.auth = AuthConfig(enabled=enabled, jwt_secret=secret, jwt_expiry_hours=1)
    return cfg


def _make_token(role: Role = Role.AGENT, exp_offset: int = 3600) -> str:
    payload = TokenPayload(
        sub="test-agent",
        role=role,
        tenant_id="default",
        exp=int(time.time()) + exp_offset,
    )
    return create_jwt(payload, SECRET)


# --- JWT tests ---

def test_create_and_verify_jwt():
    payload = TokenPayload(sub="agent1", role=Role.AGENT, tenant_id="t1", exp=int(time.time()) + 3600)
    token = create_jwt(payload, SECRET)
    result = verify_jwt(token, SECRET)
    assert result is not None
    assert result.sub == "agent1"
    assert result.role == Role.AGENT
    assert result.tenant_id == "t1"


def test_verify_jwt_expired():
    payload = TokenPayload(sub="agent1", role=Role.AGENT, tenant_id="t1", exp=int(time.time()) - 10)
    token = create_jwt(payload, SECRET)
    assert verify_jwt(token, SECRET) is None


def test_verify_jwt_wrong_secret():
    payload = TokenPayload(sub="x", role=Role.READER, tenant_id="d", exp=int(time.time()) + 3600)
    token = create_jwt(payload, SECRET)
    assert verify_jwt(token, "wrong-secret") is None


def test_verify_jwt_malformed():
    assert verify_jwt("not.a.jwt", SECRET) is None


# --- API key tests ---

def test_create_and_verify_api_key(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    monkeypatch.setattr("engram.auth._api_keys_path", lambda: keys_file)

    key, record = create_api_key("svc-bot", Role.AGENT, "default")
    assert record.name == "svc-bot"
    assert record.role == Role.AGENT
    assert record.active is True

    found = verify_api_key(key)
    assert found is not None
    assert found.name == "svc-bot"


def test_verify_api_key_wrong_key(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    monkeypatch.setattr("engram.auth._api_keys_path", lambda: keys_file)
    create_api_key("bot", Role.READER)
    assert verify_api_key("wrong-key") is None


def test_revoke_api_key(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    monkeypatch.setattr("engram.auth._api_keys_path", lambda: keys_file)

    key, _ = create_api_key("revokable", Role.AGENT)
    assert revoke_api_key("revokable") is True
    assert verify_api_key(key) is None


# --- HTTP auth middleware tests ---

def test_no_auth_health_always_public(client_no_auth):
    resp = client_no_auth.get("/health")
    assert resp.status_code == 200


def test_no_auth_all_routes_pass(client_no_auth):
    """With auth disabled, all routes work without credentials."""
    assert client_no_auth.get("/api/v1/status").status_code == 200
    assert client_no_auth.post("/api/v1/remember", json={"content": "x"}).status_code == 200
    assert client_no_auth.get("/api/v1/recall", params={"query": "x"}).status_code == 200


def test_auth_enabled_no_credentials_returns_401(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.get("/api/v1/status")
    assert resp.status_code == 401


def test_auth_enabled_valid_jwt_grants_access(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(Role.AGENT)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.get("/api/v1/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_auth_enabled_expired_jwt_returns_401(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(Role.AGENT, exp_offset=-10)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.get("/api/v1/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_reader_role_cannot_post(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(Role.READER)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.post("/api/v1/remember", json={"content": "x"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_agent_role_cannot_access_admin_endpoints(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(Role.AGENT)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.post("/api/v1/cleanup", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_admin_role_can_access_all(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(Role.ADMIN)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.post("/api/v1/cleanup", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_api_key_auth_grants_access(tmp_path, monkeypatch, mock_episodic, mock_graph, mock_engine):
    keys_file = tmp_path / "api_keys.json"
    monkeypatch.setattr("engram.auth._api_keys_path", lambda: keys_file)

    key, _ = create_api_key("svc", Role.AGENT)
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.get("/api/v1/status", headers={"X-API-Key": key})
    assert resp.status_code == 200


def test_invalid_api_key_returns_401(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, raise_server_exceptions=False)
    with patch("engram.auth.load_config", return_value=_make_config(enabled=True)):
        resp = client.get("/api/v1/status", headers={"X-API-Key": "invalid-key"})
    assert resp.status_code == 401


# --- /auth/token endpoint ---

def test_auth_token_endpoint_disabled_returns_404(client_no_auth):
    with patch("engram.capture.server.load_config", return_value=_make_config(enabled=False)):
        resp = client_no_auth.post("/api/v1/auth/token", json={
            "sub": "agent", "role": "agent", "jwt_secret": SECRET
        })
    assert resp.status_code == 404


def test_auth_token_endpoint_issues_token(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, follow_redirects=True)
    with patch("engram.capture.server.load_config", return_value=_make_config(enabled=True)):
        resp = client.post("/api/v1/auth/token", json={
            "sub": "agent1", "role": "agent", "tenant_id": "default", "jwt_secret": SECRET
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_auth_token_wrong_secret_returns_401(mock_episodic, mock_graph, mock_engine):
    app = create_app(mock_episodic, mock_graph, mock_engine)
    client = TestClient(app, follow_redirects=True)
    with patch("engram.capture.server.load_config", return_value=_make_config(enabled=True)):
        resp = client.post("/api/v1/auth/token", json={
            "sub": "x", "role": "agent", "jwt_secret": "wrong-secret"
        })
    assert resp.status_code == 401
