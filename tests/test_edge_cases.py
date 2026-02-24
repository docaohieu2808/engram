"""Edge case tests for engram v0.2 — unicode, special chars, large batches, TTL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engram.models import EpisodicMemory, MemoryType, SemanticEdge, SemanticNode


# ---------------------------------------------------------------------------
# Unicode / Vietnamese content
# ---------------------------------------------------------------------------

class TestUnicodeContent:
    @pytest.mark.asyncio
    async def test_remember_vietnamese_content(self, episodic_store):
        content = "Đây là một bài test bằng tiếng Việt"
        mem_id = await episodic_store.remember(content)
        assert mem_id

        results = await episodic_store.search("bài test tiếng Việt")
        assert any(r.content == content for r in results)

    @pytest.mark.asyncio
    async def test_remember_mixed_unicode(self, episodic_store):
        content = "System 日本語 тест Ελλάδα 한국어"
        mem_id = await episodic_store.remember(content)
        assert mem_id

        results = await episodic_store.search(content[:10])
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_semantic_node_with_unicode_name(self, semantic_graph):
        node = SemanticNode(type="Person", name="Nguyễn Văn An")
        await semantic_graph.add_node(node)

        results = await semantic_graph.query("Nguyễn")
        assert any(n.name == "Nguyễn Văn An" for n in results)

    @pytest.mark.asyncio
    async def test_content_with_special_chars(self, episodic_store):
        content = 'Config key: "api_url" = https://example.com/path?q=1&r=2'
        mem_id = await episodic_store.remember(content)
        assert mem_id

        results = await episodic_store.search("api_url config")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Entity names with commas (CSV fix verification)
# ---------------------------------------------------------------------------

class TestEntityNamesWithCommas:
    @pytest.mark.asyncio
    async def test_entities_with_commas_stored_correctly(self, episodic_store):
        """Entity name containing a comma should be stored and retrieved intact."""
        entities = ["Service:foo,bar", "Team:baz"]
        mem_id = await episodic_store.remember(
            "service with comma in name",
            entities=entities,
        )
        assert mem_id

        results = await episodic_store.search("service comma")
        assert len(results) > 0
        stored_entities = results[0].entities
        assert "Service:foo,bar" in stored_entities
        assert "Team:baz" in stored_entities

    @pytest.mark.asyncio
    async def test_tags_with_commas_stored_correctly(self, episodic_store):
        tags = ["tag-one", "tag,two", "tag:three"]
        await episodic_store.remember("memory with special tags", tags=tags)
        results = await episodic_store.search("special tags")
        assert len(results) > 0
        assert set(results[0].tags) == set(tags)


# ---------------------------------------------------------------------------
# Empty search results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    @pytest.mark.asyncio
    async def test_search_nonexistent_returns_empty_list(self, episodic_store):
        results = await episodic_store.search("xyzzy_totally_nonexistent_query_12345")
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_semantic_query_nonexistent_returns_empty(self, semantic_graph):
        nodes = await semantic_graph.query("nonexistent_entity_xyz_987")
        assert isinstance(nodes, list)
        assert len(nodes) == 0

    @pytest.mark.asyncio
    async def test_get_related_nonexistent_returns_empty(self, semantic_graph):
        result = await semantic_graph.get_related(["NonExistent:Node"], depth=2)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Large batch operations
# ---------------------------------------------------------------------------

class TestLargeBatch:
    @pytest.mark.asyncio
    async def test_remember_batch_50_items(self, episodic_store):
        """Batch of 50 memories should all be stored."""
        batch = [
            {"content": f"Memory item number {i}", "memory_type": "fact", "priority": 5}
            for i in range(50)
        ]
        ids = await episodic_store.remember_batch(batch)
        assert len(ids) == 50
        assert all(isinstance(i, str) for i in ids)

    @pytest.mark.asyncio
    async def test_add_nodes_batch_large(self, semantic_graph):
        """Batch of 30 nodes should all be stored."""
        nodes = [SemanticNode(type="Technology", name=f"Tool{i}") for i in range(30)]
        await semantic_graph.add_nodes_batch(nodes)

        stats = await semantic_graph.stats()
        assert stats["node_count"] >= 30

    @pytest.mark.asyncio
    async def test_large_search_returns_limited_results(self, episodic_store):
        """Search with limit=5 returns at most 5 results even with many stored."""
        batch = [
            {"content": f"Python programming tip number {i}", "memory_type": "fact"}
            for i in range(20)
        ]
        await episodic_store.remember_batch(batch)

        results = await episodic_store.search("Python programming", limit=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    @pytest.mark.asyncio
    async def test_expired_memory_not_in_search_results(self, episodic_store):
        """Memory with past expires_at should not appear in search results."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        await episodic_store.remember(
            "This memory has expired",
            expires_at=past,
        )

        results = await episodic_store.search("memory expired")
        # Expired memories should be filtered from results
        assert all(
            r.expires_at is None or r.expires_at > datetime.now(timezone.utc)
            for r in results
        )

    @pytest.mark.asyncio
    async def test_future_expiry_memory_is_in_results(self, episodic_store):
        """Memory with future expires_at should appear in search."""
        future = datetime.now(timezone.utc) + timedelta(hours=24)
        content = "This memory will expire tomorrow"
        await episodic_store.remember(content, expires_at=future)

        results = await episodic_store.search("expire tomorrow")
        assert any(r.content == content for r in results)

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_past_memories(self, episodic_store):
        """cleanup_expired() should delete memories past their expires_at."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        future = datetime.now(timezone.utc) + timedelta(hours=2)

        await episodic_store.remember("expired one", expires_at=past)
        await episodic_store.remember("expired two", expires_at=past)
        await episodic_store.remember("valid memory", expires_at=future)

        deleted = await episodic_store.cleanup_expired()
        assert deleted == 2

        # Valid memory should remain searchable
        results = await episodic_store.search("valid memory")
        assert any(r.content == "valid memory" for r in results)


# ---------------------------------------------------------------------------
# Multi-value tag filtering
# ---------------------------------------------------------------------------

class TestTagFiltering:
    @pytest.mark.asyncio
    async def test_search_by_single_tag(self, episodic_store):
        await episodic_store.remember("alpha content", tags=["alpha"])
        await episodic_store.remember("beta content", tags=["beta"])

        results = await episodic_store.search("content", tags=["alpha"])
        assert all("alpha" in r.tags for r in results)

    @pytest.mark.asyncio
    async def test_search_by_multiple_tags(self, episodic_store):
        await episodic_store.remember("both tags", tags=["x", "y"])
        await episodic_store.remember("only x tag", tags=["x"])

        results = await episodic_store.search("tags", tags=["x", "y"])
        # All returned memories must have both tags
        for r in results:
            assert "x" in r.tags
            assert "y" in r.tags

    @pytest.mark.asyncio
    async def test_search_no_tag_match_returns_empty(self, episodic_store):
        await episodic_store.remember("tagged memory", tags=["real_tag"])

        results = await episodic_store.search("tagged memory", tags=["nonexistent_tag"])
        assert len(results) == 0
