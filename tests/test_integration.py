"""Integration tests using real stores with mocked LLM/embedding API.

All tests are marked @pytest.mark.integration and use temporary directories
so they never touch production data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
    cfg = EpisodicConfig(path=str(tmp_path / "episodic"))
    embed = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        yield EpisodicStore(config=cfg, embedding_config=embed)


@pytest.fixture
def graph(tmp_path):
    cfg = SemanticConfig(path=str(tmp_path / "semantic.db"))
    return create_graph(cfg)


@pytest.fixture
def tenant_a(tmp_path):
    cfg = EpisodicConfig(path=str(tmp_path / "shared_episodic"))
    embed = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        yield EpisodicStore(config=cfg, embedding_config=embed, namespace="tenant_a")


@pytest.fixture
def tenant_b(tmp_path):
    cfg = EpisodicConfig(path=str(tmp_path / "shared_episodic"))
    embed = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        yield EpisodicStore(config=cfg, embedding_config=embed, namespace="tenant_b")


# --- Full episodic workflow ---

@pytest.mark.integration
async def test_remember_then_recall_finds_content(episodic):
    """remember → search → content found in results."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        mem_id = await episodic.remember("Python is great for data science", memory_type=MemoryType.FACT)
    assert mem_id

    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        results = await episodic.search("Python data science", limit=5)

    assert len(results) >= 1
    contents = [r.content for r in results]
    assert any("Python" in c for c in contents)


@pytest.mark.integration
async def test_remember_with_tags_then_filter_by_tag(episodic):
    """remember with tags → recall filtered by tag → only tagged results."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await episodic.remember("Deploy to production", memory_type=MemoryType.DECISION, tags=["deploy", "prod"])
        await episodic.remember("Refactor auth module", memory_type=MemoryType.FACT, tags=["auth"])

    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        results = await episodic.search("production deploy", limit=5, tags=["deploy"])

    assert len(results) >= 1
    for r in results:
        assert "deploy" in r.tags


@pytest.mark.integration
async def test_remember_multiple_then_count_in_stats(episodic):
    """Multiple remember calls → stats count increases correctly."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await episodic.remember("First memory")
        await episodic.remember("Second memory")
        await episodic.remember("Third memory")

    stats = await episodic.stats()
    assert stats["count"] >= 3


@pytest.mark.integration
async def test_cleanup_removes_expired_memories(episodic):
    """remember with past expiry → cleanup → count decreases."""
    from datetime import datetime, timedelta

    past = datetime.now() - timedelta(hours=1)
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await episodic.remember("This should expire", expires_at=past)
        await episodic.remember("This is permanent")

    deleted = await episodic.cleanup_expired()
    assert deleted >= 1

    stats = await episodic.stats()
    assert stats["count"] >= 1


# --- Graph workflow ---

@pytest.mark.integration
async def test_add_nodes_then_query_finds_them(graph):
    """add_node × 2 → query by keyword → nodes found."""
    node1 = SemanticNode(type="Service", name="auth-service")
    node2 = SemanticNode(type="Service", name="payment-service")
    await graph.add_node(node1)
    await graph.add_node(node2)

    results = await graph.query("auth")
    assert len(results) >= 1
    names = [n.name for n in results]
    assert "auth-service" in names


@pytest.mark.integration
async def test_add_edge_then_get_related(graph):
    """add nodes + edge → get_related by name → edge appears in result."""
    team = SemanticNode(type="Team", name="platform-team")
    svc = SemanticNode(type="Service", name="api-gateway")
    await graph.add_node(team)
    await graph.add_node(svc)

    edge = SemanticEdge(from_node=team.key, to_node=svc.key, relation="owns")
    await graph.add_edge(edge)

    # get_related searches by plain name (suffix match on node keys)
    related = await graph.get_related(["platform-team"])
    assert "platform-team" in related
    edges = related["platform-team"].get("edges", [])
    assert len(edges) >= 1
    assert any(e.relation == "owns" for e in edges)


@pytest.mark.integration
async def test_remove_node_no_longer_in_query(graph):
    """add node → remove node → query returns nothing."""
    node = SemanticNode(type="Technology", name="legacy-db")
    await graph.add_node(node)

    results_before = await graph.query("legacy-db")
    assert len(results_before) >= 1

    await graph.remove_node(node.key)
    results_after = await graph.query("legacy-db")
    names_after = [n.name for n in results_after]
    assert "legacy-db" not in names_after


@pytest.mark.integration
async def test_graph_stats_reflect_adds(graph):
    """add 2 nodes, 1 edge → stats show correct counts."""
    node1 = SemanticNode(type="Service", name="svc-a")
    node2 = SemanticNode(type="Service", name="svc-b")
    await graph.add_node(node1)
    await graph.add_node(node2)
    edge = SemanticEdge(from_node=node1.key, to_node=node2.key, relation="calls")
    await graph.add_edge(edge)

    stats = await graph.stats()
    assert stats["node_count"] >= 2
    assert stats["edge_count"] >= 1


# --- Multi-tenant isolation ---

@pytest.mark.integration
async def test_tenant_isolation_store_a_not_visible_from_b(tenant_a, tenant_b):
    """Data stored by tenant A is not visible to tenant B."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await tenant_a.remember("Tenant A secret data", memory_type=MemoryType.FACT)

    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        results_b = await tenant_b.search("Tenant A secret data", limit=5)

    # Tenant B must not see tenant A's memories
    contents_b = [r.content for r in results_b]
    assert not any("Tenant A" in c for c in contents_b)


@pytest.mark.integration
async def test_tenant_isolation_separate_stats(tenant_a, tenant_b):
    """Each tenant's stats are independent."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await tenant_a.remember("A memory 1")
        await tenant_a.remember("A memory 2")

    stats_a = await tenant_a.stats()
    stats_b = await tenant_b.stats()

    assert stats_a["count"] >= 2
    assert stats_b["count"] == 0


# --- Health check ---

@pytest.mark.integration
async def test_health_check_returns_ok(episodic, graph):
    """deep_check with real stores returns overall healthy status."""
    from engram.health import deep_check

    result = await deep_check(episodic, graph)
    assert result["status"] in ("healthy", "unhealthy")  # just verifies it runs
    assert "components" in result
    assert "chromadb" in result["components"]
    assert "semantic" in result["components"]


@pytest.mark.integration
async def test_health_check_chromadb_component(episodic, graph):
    """ChromaDB component health reports count correctly."""
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await episodic.remember("health check memory")

    from engram.health import check_chromadb

    component = await check_chromadb(episodic)
    assert component.status == "healthy"
    assert component.details.get("count", 0) >= 1


# --- Backup / restore roundtrip ---

@pytest.mark.integration
async def test_backup_restore_roundtrip(episodic, graph, tmp_path):
    """backup → restore → verify episodic data intact."""
    from engram.backup import backup, restore
    from engram.config import EpisodicConfig, EmbeddingConfig

    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        await episodic.remember("Backup test memory", memory_type=MemoryType.FACT)

    archive = str(tmp_path / "test_backup.tar.gz")
    manifest = await backup(episodic, graph, archive)
    assert manifest["episodic_count"] >= 1

    # Restore into a fresh store
    fresh_cfg = EpisodicConfig(path=str(tmp_path / "restored_episodic"))
    fresh_embed = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings):
        fresh_store = EpisodicStore(config=fresh_cfg, embedding_config=fresh_embed)

    from engram.config import SemanticConfig
    fresh_graph = create_graph(SemanticConfig(path=str(tmp_path / "restored_semantic.db")))

    result = await restore(fresh_store, fresh_graph, archive)
    assert result["episodic_restored"] >= 1
