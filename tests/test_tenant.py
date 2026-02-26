"""Tests for tenant isolation: TenantContext, StoreFactory, and HTTP tenant routing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.tenant import TenantContext, StoreFactory, validate_tenant_id


# --- TenantContext ---

class TestTenantContext:
    def test_default_is_default(self):
        assert TenantContext.get() == "default"

    def test_set_and_get(self):
        TenantContext.set("acme")
        assert TenantContext.get() == "acme"
        # Reset for other tests
        TenantContext.set("default")

    def test_contextvars_isolation(self):
        """Each asyncio Task gets its own contextvar copy."""
        results = {}

        async def task_a():
            TenantContext.set("tenant-a")
            await asyncio.sleep(0)
            results["a"] = TenantContext.get()

        async def task_b():
            TenantContext.set("tenant-b")
            await asyncio.sleep(0)
            results["b"] = TenantContext.get()

        async def run():
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        assert results["a"] == "tenant-a"
        assert results["b"] == "tenant-b"


# --- validate_tenant_id ---

class TestValidateTenantId:
    def test_valid_ids(self):
        assert validate_tenant_id("default") == "default"
        assert validate_tenant_id("acme-corp") == "acme-corp"
        assert validate_tenant_id("tenant_123") == "tenant_123"
        assert validate_tenant_id("A") == "A"

    def test_invalid_ids(self):
        with pytest.raises(ValueError):
            validate_tenant_id("has spaces")
        with pytest.raises(ValueError):
            validate_tenant_id("has.dots")
        with pytest.raises(ValueError):
            validate_tenant_id("")
        with pytest.raises(ValueError):
            validate_tenant_id("x" * 65)  # too long

    def test_max_length_valid(self):
        assert validate_tenant_id("x" * 64) == "x" * 64


# --- StoreFactory ---

@pytest.fixture
def mock_config():
    """Minimal Config mock for StoreFactory."""
    from engram.config import (
        Config, EpisodicConfig, EmbeddingConfig, SemanticConfig,
        HooksConfig, LLMConfig,
    )
    cfg = Config(
        episodic=EpisodicConfig(path="/tmp/engram-test-episodic", namespace="default", dedup_enabled=False),
        embedding=EmbeddingConfig(provider="gemini", model="gemini-embedding-001"),
        semantic=SemanticConfig(provider="sqlite", path="/tmp/engram-test.db"),
    )
    return cfg


class TestStoreFactory:
    def test_get_episodic_returns_store(self, mock_config):
        factory = StoreFactory(mock_config)
        with patch("engram.tenant.EpisodicStore") as MockEpisodic:
            mock_store = MagicMock()
            MockEpisodic.return_value = mock_store
            store = factory.get_episodic("acme")
            assert store is mock_store
            MockEpisodic.assert_called_once()

    def test_get_episodic_caches_same_tenant(self, mock_config):
        factory = StoreFactory(mock_config)
        with patch("engram.tenant.EpisodicStore") as MockEpisodic:
            mock_store = MagicMock()
            MockEpisodic.return_value = mock_store
            s1 = factory.get_episodic("acme")
            s2 = factory.get_episodic("acme")
            # Same object returned from cache
            assert s1 is s2
            # Constructor called only once
            assert MockEpisodic.call_count == 1

    def test_get_episodic_different_tenants_different_stores(self, mock_config):
        factory = StoreFactory(mock_config)
        stores = {}
        with patch("engram.tenant.EpisodicStore") as MockEpisodic:
            MockEpisodic.side_effect = lambda *a, **kw: MagicMock()
            stores["a"] = factory.get_episodic("tenant-a")
            stores["b"] = factory.get_episodic("tenant-b")
            assert stores["a"] is not stores["b"]
            assert MockEpisodic.call_count == 2

    def test_get_episodic_uses_tenant_context_by_default(self, mock_config):
        factory = StoreFactory(mock_config)
        TenantContext.set("ctx-tenant")
        with patch("engram.tenant.EpisodicStore") as MockEpisodic:
            mock_store = MagicMock()
            MockEpisodic.return_value = mock_store
            store = factory.get_episodic()  # no explicit tenant_id
            # Should be scoped to ctx-tenant
            assert "ctx-tenant" in factory._episodic_stores
        TenantContext.set("default")

    @pytest.mark.asyncio
    async def test_get_graph_creates_sqlite_per_tenant(self, mock_config):
        factory = StoreFactory(mock_config)
        with patch("engram.tenant.create_graph") as mock_create:
            mock_graph = AsyncMock()
            mock_create.return_value = mock_graph
            g = await factory.get_graph("tenant-x")
            assert g is mock_graph
            # Verify a tenant-scoped SemanticConfig was used
            call_arg = mock_create.call_args[0][0]
            assert "tenant-x" in call_arg.path

    @pytest.mark.asyncio
    async def test_get_graph_caches_same_tenant(self, mock_config):
        factory = StoreFactory(mock_config)
        with patch("engram.tenant.create_graph") as mock_create:
            mock_create.return_value = AsyncMock()
            g1 = await factory.get_graph("cached-tenant")
            g2 = await factory.get_graph("cached-tenant")
            assert g1 is g2
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_closes_oldest(self, mock_config):
        factory = StoreFactory(mock_config)
        factory._MAX_GRAPH_CACHE = 2  # Lower limit for test

        closed = []

        def make_graph():
            g = AsyncMock()
            g.close = AsyncMock(side_effect=lambda: closed.append(True))
            return g

        with patch("engram.tenant.create_graph") as mock_create:
            mock_create.side_effect = lambda cfg: make_graph()
            await factory.get_graph("t1")
            await factory.get_graph("t2")
            # Adding a third should evict the oldest (t1)
            await factory.get_graph("t3")

        # t1 should have been closed (evicted)
        assert len(closed) == 1

    @pytest.mark.asyncio
    async def test_close_all(self, mock_config):
        factory = StoreFactory(mock_config)
        closed = []

        def make_graph():
            g = AsyncMock()
            g.close = AsyncMock(side_effect=lambda: closed.append(True))
            return g

        with patch("engram.tenant.create_graph") as mock_create:
            mock_create.side_effect = lambda cfg: make_graph()
            await factory.get_graph("a")
            await factory.get_graph("b")

        await factory.close_all()
        assert len(closed) == 2
        assert len(factory._graphs) == 0


# --- HTTP server tenant routing ---

class TestHttpTenantRouting:
    """Verify HTTP routes resolve stores via StoreFactory based on auth.tenant_id."""

    @pytest.fixture
    def mock_ep(self):
        ep = AsyncMock()
        ep.remember = AsyncMock(return_value="mem-id-456")
        ep.search = AsyncMock(return_value=[])
        ep.stats = AsyncMock(return_value={"count": 0})
        ep.cleanup_expired = AsyncMock(return_value=0)
        return ep

    @pytest.fixture
    def mock_gr(self):
        gr = AsyncMock()
        gr.query = AsyncMock(return_value=[])
        gr.get_related = AsyncMock(return_value={})
        gr.stats = AsyncMock(return_value={"node_count": 0, "edge_count": 0})
        return gr

    @pytest.fixture
    def mock_store_factory(self, mock_ep, mock_gr):
        factory = MagicMock(spec=StoreFactory)
        factory.get_episodic = MagicMock(return_value=mock_ep)
        factory.get_graph = AsyncMock(return_value=mock_gr)
        return factory

    def _remember_path(self):
        """Detect current remember route path (versioned or legacy)."""
        from engram.capture.server import _LEGACY_POST_ROUTES
        # Routes are under /api/v1 when versioning is active
        return "/api/v1/remember"

    def test_remember_uses_tenant_store(self, mock_store_factory):
        from fastapi.testclient import TestClient
        from engram.capture.server import create_app
        from engram.auth import get_auth_context
        from engram.auth_models import AuthContext, Role

        auth = AuthContext(tenant_id="corp-a", role=Role.ADMIN)
        app = create_app(store_factory=mock_store_factory)
        app.dependency_overrides[get_auth_context] = lambda: auth

        client = TestClient(app, follow_redirects=True)
        resp = client.post("/api/v1/remember", json={
            "content": "Corp A memory",
            "memory_type": "fact",
            "priority": 5,
            "entities": [],
            "tags": [],
        })

        assert resp.status_code == 200
        # StoreFactory.get_episodic was called with corp-a tenant
        mock_store_factory.get_episodic.assert_called_with("corp-a")

    def test_tenant_isolation_different_auth(self, mock_store_factory):
        """Two requests with different tenant_ids call get_episodic with different IDs."""
        from fastapi.testclient import TestClient
        from engram.capture.server import create_app
        from engram.auth import get_auth_context
        from engram.auth_models import AuthContext, Role

        app = create_app(store_factory=mock_store_factory)

        auth_a = AuthContext(tenant_id="tenant-alpha", role=Role.ADMIN)
        auth_b = AuthContext(tenant_id="tenant-beta", role=Role.ADMIN)
        payload = {"content": "memory", "memory_type": "fact", "priority": 5, "entities": [], "tags": []}

        # Request from tenant-alpha
        app.dependency_overrides[get_auth_context] = lambda: auth_a
        client = TestClient(app, follow_redirects=True)
        client.post("/api/v1/remember", json=payload)

        # Request from tenant-beta
        app.dependency_overrides[get_auth_context] = lambda: auth_b
        client.post("/api/v1/remember", json=payload)

        calls = [c[0][0] for c in mock_store_factory.get_episodic.call_args_list]
        assert "tenant-alpha" in calls
        assert "tenant-beta" in calls
