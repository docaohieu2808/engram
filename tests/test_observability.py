"""Tests for Phase 7: Observability — telemetry and audit modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engram.audit import AuditLogger, get_audit, setup_audit
from engram.config import AuditConfig, Config, TelemetryConfig, load_config
from engram.telemetry import get_meter, get_tracer, setup_telemetry


# ---------------------------------------------------------------------------
# AuditLogger unit tests
# ---------------------------------------------------------------------------


class TestAuditLoggerDisabled:
    def test_disabled_by_default(self):
        audit = AuditLogger(enabled=False)
        assert not audit.enabled

    def test_log_noop_when_disabled(self, tmp_path):
        audit = AuditLogger(enabled=False, path=str(tmp_path / "audit.jsonl"))
        audit.log(tenant_id="t1", actor="agent", operation="remember")
        # No file should be created
        assert not (tmp_path / "audit.jsonl").exists()


class TestAuditLoggerEnabled:
    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))
        assert path.parent.exists()

    def test_log_writes_jsonl_entry(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))
        audit.log(
            tenant_id="tenant1",
            actor="system",
            operation="episodic.remember",
            resource_id="mem-123",
            details={"memory_type": "fact"},
        )
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tenant_id"] == "tenant1"
        assert entry["actor"] == "system"
        assert entry["operation"] == "episodic.remember"
        assert entry["resource_id"] == "mem-123"
        assert entry["details"]["memory_type"] == "fact"
        assert "timestamp" in entry

    def test_log_appends_multiple_entries(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))
        for i in range(3):
            audit.log(tenant_id="t", actor="a", operation=f"op{i}")
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_log_defaults_empty_details(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))
        audit.log(tenant_id="t", actor="a", operation="op")
        entry = json.loads(path.read_text().strip())
        assert entry["details"] == {}
        assert entry["resource_id"] == ""


class TestAuditGlobalSingleton:
    def test_get_audit_returns_disabled_by_default(self):
        audit = get_audit()
        assert not audit.enabled

    def test_setup_audit_replaces_singleton(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        result = setup_audit(enabled=True, path=path)
        assert result.enabled
        assert get_audit().enabled
        # Restore disabled state so other tests are unaffected
        setup_audit(enabled=False, path=path)


# ---------------------------------------------------------------------------
# Config: TelemetryConfig and AuditConfig defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_telemetry_disabled_by_default(self):
        config = Config()
        assert not config.telemetry.enabled
        assert config.telemetry.otlp_endpoint == ""
        assert config.telemetry.service_name == "engram"
        assert config.telemetry.sample_rate == 0.1

    def test_audit_disabled_by_default(self):
        config = Config()
        assert not config.audit.enabled
        assert config.audit.backend == "file"
        assert "audit.jsonl" in config.audit.path

    def test_telemetry_env_var_override(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("ENGRAM_TELEMETRY_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("ENGRAM_TELEMETRY_SERVICE_NAME", "myapp")
        config = load_config(path=Path("/nonexistent/config.yaml"))
        assert config.telemetry.enabled
        assert config.telemetry.otlp_endpoint == "http://localhost:4317"
        assert config.telemetry.service_name == "myapp"

    def test_audit_env_var_override(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_AUDIT_ENABLED", "true")
        monkeypatch.setenv("ENGRAM_AUDIT_PATH", "/tmp/test-audit.jsonl")
        config = load_config(path=Path("/nonexistent/config.yaml"))
        assert config.audit.enabled
        assert config.audit.path == "/tmp/test-audit.jsonl"


# ---------------------------------------------------------------------------
# Telemetry setup: noop when disabled or OTel not installed
# ---------------------------------------------------------------------------


class TestTelemetrySetup:
    def test_noop_when_disabled(self):
        config = Config()
        assert not config.telemetry.enabled
        setup_telemetry(config)
        # Tracer/meter remain whatever they were (None if not previously set)
        # Key assertion: no exception raised

    def test_get_tracer_returns_none_when_disabled(self):
        # Import fresh state: tracer is None unless setup was called with enabled=True
        import engram.telemetry as tmod
        original = tmod._tracer
        tmod._tracer = None
        assert get_tracer() is None
        tmod._tracer = original

    def test_get_meter_returns_none_when_disabled(self):
        import engram.telemetry as tmod
        original = tmod._meter
        tmod._meter = None
        assert get_meter() is None
        tmod._meter = original

    def test_logs_warning_when_otel_not_installed(self):
        config = Config()
        config.telemetry.enabled = True  # type: ignore[assignment]
        with patch.dict("sys.modules", {"opentelemetry": None,
                                         "opentelemetry.trace": None,
                                         "opentelemetry.metrics": None,
                                         "opentelemetry.sdk": None,
                                         "opentelemetry.sdk.trace": None,
                                         "opentelemetry.sdk.metrics": None,
                                         "opentelemetry.sdk.trace.export": None}):
            import logging
            with patch.object(logging.getLogger("engram"), "warning") as mock_warn:
                setup_telemetry(config)
                mock_warn.assert_called_once()


# ---------------------------------------------------------------------------
# EpisodicStore audit integration
# ---------------------------------------------------------------------------


class TestEpisodicStoreAudit:
    @pytest.mark.asyncio
    async def test_remember_calls_audit(self, tmp_path, mock_embeddings):
        from engram.config import EmbeddingConfig, EpisodicConfig
        from engram.episodic.store import EpisodicStore

        audit = MagicMock(spec=AuditLogger)
        audit.enabled = True

        store = EpisodicStore(
            config=EpisodicConfig(path=str(tmp_path / "episodic")),
            embedding_config=EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2"),
            audit=audit,
        )
        mem_id = await store.remember("test memory")
        audit.log.assert_called_once()
        call_kwargs = audit.log.call_args
        assert call_kwargs.kwargs["operation"] == "episodic.remember"
        assert call_kwargs.kwargs["resource_id"] == mem_id

    @pytest.mark.asyncio
    async def test_remember_no_audit_no_error(self, tmp_path, mock_embeddings):
        from engram.config import EmbeddingConfig, EpisodicConfig
        from engram.episodic.store import EpisodicStore

        store = EpisodicStore(
            config=EpisodicConfig(path=str(tmp_path / "episodic")),
            embedding_config=EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2"),
            audit=None,
        )
        mem_id = await store.remember("test memory without audit")
        assert mem_id  # Should succeed without audit


# ---------------------------------------------------------------------------
# SemanticGraph audit integration
# ---------------------------------------------------------------------------


class TestSemanticGraphAudit:
    @pytest.mark.asyncio
    async def test_add_node_calls_audit(self, tmp_path):
        from engram.config import SemanticConfig
        from engram.models import SemanticNode
        from engram.semantic import create_graph

        audit = MagicMock(spec=AuditLogger)
        config = SemanticConfig(path=str(tmp_path / "semantic.db"))
        graph = create_graph(config, audit=audit)
        node = SemanticNode(type="Service", name="api-gateway")
        await graph.add_node(node)
        audit.log.assert_called_once()
        assert "semantic.add_node" in audit.log.call_args.kwargs["operation"]

    @pytest.mark.asyncio
    async def test_add_edge_calls_audit(self, tmp_path):
        from engram.config import SemanticConfig
        from engram.models import SemanticEdge, SemanticNode
        from engram.semantic import create_graph

        audit = MagicMock(spec=AuditLogger)
        config = SemanticConfig(path=str(tmp_path / "semantic.db"))
        graph = create_graph(config, audit=audit)
        n1 = SemanticNode(type="Team", name="platform")
        n2 = SemanticNode(type="Service", name="api")
        await graph.add_node(n1)
        await graph.add_node(n2)
        audit.reset_mock()
        edge = SemanticEdge(from_node=n1.key, to_node=n2.key, relation="owns")
        await graph.add_edge(edge)
        audit.log.assert_called_once()
        assert audit.log.call_args.kwargs["operation"] == "semantic.add_edge"

    @pytest.mark.asyncio
    async def test_remove_node_calls_audit(self, tmp_path):
        from engram.config import SemanticConfig
        from engram.models import SemanticNode
        from engram.semantic import create_graph

        audit = MagicMock(spec=AuditLogger)
        config = SemanticConfig(path=str(tmp_path / "semantic.db"))
        graph = create_graph(config, audit=audit)
        node = SemanticNode(type="Service", name="old-svc")
        await graph.add_node(node)
        audit.reset_mock()
        await graph.remove_node(node.key)
        audit.log.assert_called_once()
        assert audit.log.call_args.kwargs["operation"] == "semantic.remove_node"

    @pytest.mark.asyncio
    async def test_no_audit_no_error(self, tmp_path):
        from engram.config import SemanticConfig
        from engram.models import SemanticNode
        from engram.semantic import create_graph

        config = SemanticConfig(path=str(tmp_path / "semantic.db"))
        graph = create_graph(config)  # no audit
        node = SemanticNode(type="Service", name="svc")
        is_new = await graph.add_node(node)
        assert is_new
