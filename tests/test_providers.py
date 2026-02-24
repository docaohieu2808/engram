"""Tests for the federated memory provider system."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import Config, DiscoveryConfig, ProviderEntry
from engram.providers.base import MemoryProvider, ProviderResult, ProviderStats
from engram.providers.file_adapter import FileAdapter
from engram.providers.registry import ProviderRegistry
from engram.providers.router import classify_query, federated_search


# --- Base / ProviderResult ---


class MockProvider(MemoryProvider):
    """Concrete provider for testing the base class."""

    def __init__(self, name="mock", results=None, healthy=True, **kwargs):
        super().__init__(name=name, provider_type="mock", **kwargs)
        self._results = results or []
        self._healthy = healthy

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        return self._results[:limit]

    async def health(self) -> bool:
        return self._healthy


class FailingProvider(MemoryProvider):
    """Provider that always raises on search."""

    def __init__(self, name="failing", **kwargs):
        super().__init__(name=name, provider_type="mock", **kwargs)

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        raise ConnectionError("Service unavailable")

    async def health(self) -> bool:
        return False


class TestProviderResult:
    def test_defaults(self):
        r = ProviderResult(content="hello")
        assert r.content == "hello"
        assert r.score == 0.0
        assert r.source == ""

    def test_with_metadata(self):
        r = ProviderResult(content="x", score=0.9, source="cognee", metadata={"file": "a.md"})
        assert r.source == "cognee"
        assert r.metadata["file"] == "a.md"


class TestProviderStats:
    def test_avg_latency_zero(self):
        s = ProviderStats()
        assert s.avg_latency_ms == 0.0

    def test_avg_latency(self):
        s = ProviderStats(query_count=10, total_latency_ms=500.0)
        assert s.avg_latency_ms == 50.0


class TestMemoryProviderBase:
    @pytest.mark.asyncio
    async def test_tracked_search_success(self):
        results = [ProviderResult(content="result1", score=0.8)]
        p = MockProvider(results=results)
        got = await p.tracked_search("test query")
        assert len(got) == 1
        assert got[0].source == "mock"
        assert p.stats.query_count == 1
        assert p.stats.hit_count == 1
        assert p.stats.error_count == 0

    @pytest.mark.asyncio
    async def test_tracked_search_empty(self):
        p = MockProvider(results=[])
        got = await p.tracked_search("test")
        assert len(got) == 0
        assert p.stats.query_count == 1
        assert p.stats.hit_count == 0

    @pytest.mark.asyncio
    async def test_tracked_search_error_counted(self):
        p = FailingProvider()
        got = await p.tracked_search("test")
        assert got == []
        assert p.stats.error_count == 1
        assert p.stats.consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_auto_disable_after_consecutive_errors(self):
        p = FailingProvider(max_consecutive_errors=3)
        for _ in range(3):
            await p.tracked_search("test")
        assert p._auto_disabled is True
        assert p.is_active is False
        # Should return empty without attempting
        got = await p.tracked_search("test")
        assert got == []
        assert p.stats.query_count == 3  # no new query counted

    @pytest.mark.asyncio
    async def test_re_enable(self):
        p = FailingProvider(max_consecutive_errors=1)
        await p.tracked_search("test")
        assert p._auto_disabled is True
        p.re_enable()
        assert p._auto_disabled is False
        assert p.is_active is True

    def test_status_label(self):
        p = MockProvider()
        assert p.status_label == "active"
        p.enabled = False
        assert p.status_label == "disabled"

    @pytest.mark.asyncio
    async def test_disabled_provider_skips_search(self):
        p = MockProvider(enabled=False, results=[ProviderResult(content="x")])
        got = await p.tracked_search("test")
        assert got == []


# --- File Adapter ---


class TestFileAdapter:
    @pytest.mark.asyncio
    async def test_search_markdown_files(self, tmp_path):
        # Create test markdown files
        (tmp_path / "notes.md").write_text("Wellington painting tips and tricks for house exteriors")
        (tmp_path / "other.md").write_text("Unrelated content about cooking")

        adapter = FileAdapter(name="test-files", path=str(tmp_path))
        results = await adapter.search("Wellington painting")
        assert len(results) == 1
        assert "Wellington" in results[0].content
        assert results[0].source == "test-files"

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_path):
        (tmp_path / "notes.md").write_text("Something completely different")
        adapter = FileAdapter(name="test", path=str(tmp_path))
        results = await adapter.search("quantum physics")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_health_existing_dir(self, tmp_path):
        adapter = FileAdapter(name="test", path=str(tmp_path))
        assert await adapter.health() is True

    @pytest.mark.asyncio
    async def test_health_missing_dir(self):
        adapter = FileAdapter(name="test", path="/nonexistent/path")
        assert await adapter.health() is False

    @pytest.mark.asyncio
    async def test_search_empty_dir(self, tmp_path):
        adapter = FileAdapter(name="test", path=str(tmp_path))
        results = await adapter.search("anything")
        assert results == []


# --- Router ---


class TestClassifyQuery:
    def test_internal_keywords(self):
        assert classify_query("what decision did I make?") == "internal"
        assert classify_query("show my todo list") == "internal"
        assert classify_query("quyết định trước đó") == "internal"

    def test_domain_keywords(self):
        assert classify_query("how to deploy nginx") == "domain"
        assert classify_query("what is kubernetes ingress") == "domain"
        assert classify_query("hướng dẫn cài đặt docker") == "domain"

    def test_long_query_defaults_domain(self):
        assert classify_query("explain the architecture of microservice deployment") == "domain"

    def test_short_query_defaults_internal(self):
        assert classify_query("server status") == "internal"


class TestFederatedSearch:
    @pytest.mark.asyncio
    async def test_fan_out_to_providers(self):
        p1 = MockProvider(name="p1", results=[ProviderResult(content="r1", score=0.9)])
        p2 = MockProvider(name="p2", results=[ProviderResult(content="r2", score=0.8)])
        results = await federated_search("how to test", [p1, p2], force_federation=True)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_skips_internal_queries(self):
        p = MockProvider(results=[ProviderResult(content="x")])
        results = await federated_search("my preference", [p])
        assert len(results) == 0  # internal query, providers skipped

    @pytest.mark.asyncio
    async def test_handles_provider_failure(self):
        good = MockProvider(name="good", results=[ProviderResult(content="ok", score=0.9)])
        bad = FailingProvider(name="bad")
        results = await federated_search("how to test", [good, bad], force_federation=True)
        assert len(results) == 1
        assert results[0].content == "ok"

    @pytest.mark.asyncio
    async def test_empty_providers(self):
        results = await federated_search("test", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_deduplication(self):
        p1 = MockProvider(name="p1", results=[ProviderResult(content="same content", score=0.9)])
        p2 = MockProvider(name="p2", results=[ProviderResult(content="same content", score=0.8)])
        results = await federated_search("test", [p1, p2], force_federation=True)
        assert len(results) == 1


# --- Registry ---


class TestProviderRegistry:
    def test_load_from_config(self):
        config = Config(providers=[
            ProviderEntry(name="test-files", type="file", path="/tmp", enabled=True),
        ])
        registry = ProviderRegistry()
        registry.load_from_config(config)
        assert len(registry.get_all()) == 1
        assert registry.get("test-files") is not None

    def test_unknown_type_skipped(self):
        config = Config(providers=[
            ProviderEntry(name="bad", type="unknown_type_xyz"),
        ])
        registry = ProviderRegistry()
        registry.load_from_config(config)
        assert len(registry.get_all()) == 0

    def test_register_manual(self):
        registry = ProviderRegistry()
        p = MockProvider(name="manual")
        registry.register(p)
        assert registry.get("manual") is p

    def test_get_active_filters_disabled(self):
        registry = ProviderRegistry()
        registry.register(MockProvider(name="on", enabled=True))
        registry.register(MockProvider(name="off", enabled=False))
        assert len(registry.get_active()) == 1
        assert registry.get_active()[0].name == "on"

    @pytest.mark.asyncio
    async def test_remove(self):
        registry = ProviderRegistry()
        registry.register(MockProvider(name="temp"))
        assert await registry.remove("temp") is True
        assert registry.get("temp") is None
        assert await registry.remove("nonexistent") is False


# --- Discovery ---


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_discover_local_paths(self, tmp_path):
        """Discovery finds file-based providers from local paths."""
        # Create a fake openclaw workspace
        openclaw_dir = tmp_path / ".openclaw" / "workspace" / "memory"
        openclaw_dir.mkdir(parents=True)
        (openclaw_dir / "test.md").write_text("test memory")

        with patch("engram.providers.discovery.KNOWN_SERVICES", [
            {
                "name": "test-openclaw",
                "ports": [],
                "paths": [str(openclaw_dir)],
                "health": None,
                "fingerprint": None,
                "default_config": {"type": "file", "pattern": "*.md"},
            },
        ]), patch("engram.providers.discovery._parse_mcp_config", return_value=[]):
            from engram.providers.discovery import discover
            found = await discover(DiscoveryConfig(local=True))
            assert len(found) == 1
            assert found[0].name == "test-openclaw"
            assert found[0].type == "file"
