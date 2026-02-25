"""Tests for brain features: audit trail, resource tier, constitution, scheduler."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# === Feature 1: Memory Audit Trail ===


class TestAuditTrailModifications:
    """Test structured modification logging in AuditLogger."""

    def test_log_modification_when_enabled(self, tmp_path):
        from engram.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))

        audit.log_modification(
            tenant_id="ns1", actor="system",
            mod_type="memory_create", resource_id="abc123",
            before_value=None, after_value="New memory content",
            description="New fact memory",
        )

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["operation"] == "modification"
        assert entry["mod_type"] == "memory_create"
        assert entry["resource_id"] == "abc123"
        assert entry["before_value"] is None
        assert entry["after_value"] == "New memory content"
        assert entry["reversible"] is True

    def test_log_modification_noop_when_disabled(self):
        from engram.audit import AuditLogger

        audit = AuditLogger(enabled=False)
        # Should not raise
        audit.log_modification("ns1", "system", "memory_delete", "id1")

    def test_log_modification_with_before_after(self, tmp_path):
        from engram.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))

        audit.log_modification(
            tenant_id="ns1", actor="admin",
            mod_type="memory_update", resource_id="xyz",
            before_value="old content", after_value="new content",
            reversible=True, description="Topic-key upsert",
        )

        entry = json.loads(path.read_text().strip())
        assert entry["before_value"] == "old content"
        assert entry["after_value"] == "new content"
        assert entry["description"] == "Topic-key upsert"

    def test_log_modification_irreversible_delete(self, tmp_path):
        from engram.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))

        audit.log_modification(
            tenant_id="ns1", actor="system",
            mod_type="memory_delete", resource_id="del1",
            before_value="deleted content", after_value=None,
            reversible=False, description="Memory deleted",
        )

        entry = json.loads(path.read_text().strip())
        assert entry["reversible"] is False
        assert entry["mod_type"] == "memory_delete"

    def test_read_recent_entries(self, tmp_path):
        from engram.audit import AuditLogger

        path = tmp_path / "audit.jsonl"
        audit = AuditLogger(enabled=True, path=str(path))

        for i in range(5):
            audit.log("ns1", "system", f"op_{i}", f"id_{i}")

        entries = audit.read_recent(3)
        assert len(entries) == 3
        # Newest first
        assert entries[0]["operation"] == "op_4"
        assert entries[2]["operation"] == "op_2"

    def test_read_recent_empty_log(self, tmp_path):
        from engram.audit import AuditLogger

        audit = AuditLogger(enabled=True, path=str(tmp_path / "empty.jsonl"))
        entries = audit.read_recent(10)
        assert entries == []

    def test_safe_serialize_truncates_long_strings(self):
        from engram.audit import _safe_serialize

        long_str = "x" * 10000
        result = _safe_serialize(long_str, max_len=100)
        assert len(result) == 100

    def test_safe_serialize_nested_dict(self):
        from engram.audit import _safe_serialize

        data = {"key": "x" * 10000, "nested": {"inner": "y" * 10000}}
        result = _safe_serialize(data, max_len=50)
        assert len(result["key"]) == 50
        assert len(result["nested"]["inner"]) == 50

    def test_modification_types_defined(self):
        from engram.audit import MODIFICATION_TYPES

        assert "memory_create" in MODIFICATION_TYPES
        assert "memory_delete" in MODIFICATION_TYPES
        assert "memory_update" in MODIFICATION_TYPES
        assert "metadata_update" in MODIFICATION_TYPES
        assert "cleanup_expired" in MODIFICATION_TYPES


# === Feature 2: Resource-aware Retrieval ===


class TestResourceTier:
    """Test resource tier evaluation and degraded mode."""

    def test_default_tier_is_full(self):
        from engram.resource_tier import ResourceMonitor, ResourceTier

        monitor = ResourceMonitor()
        assert monitor.get_tier() == ResourceTier.FULL

    def test_degrades_to_basic_after_failures(self):
        from engram.resource_tier import ResourceMonitor, ResourceTier

        monitor = ResourceMonitor(failure_threshold=3)
        monitor.record_failure("rate_limit")
        monitor.record_failure("rate_limit")
        monitor.record_failure("rate_limit")
        assert monitor.get_tier() == ResourceTier.BASIC

    def test_can_use_llm_on_full_tier(self):
        from engram.resource_tier import ResourceMonitor

        monitor = ResourceMonitor()
        assert monitor.can_use_llm() is True

    def test_cannot_use_llm_on_basic_tier(self):
        from engram.resource_tier import ResourceMonitor

        monitor = ResourceMonitor(failure_threshold=2)
        monitor.record_failure("rate_limit")
        monitor.record_failure("rate_limit")
        assert monitor.can_use_llm() is False

    def test_success_resets_partially(self):
        from engram.resource_tier import ResourceMonitor, ResourceTier

        monitor = ResourceMonitor(failure_threshold=3)
        monitor.record_failure("transient")
        monitor.record_success()
        assert monitor.get_tier() in (ResourceTier.FULL, ResourceTier.STANDARD)

    def test_force_tier_override(self):
        from engram.resource_tier import ResourceMonitor, ResourceTier

        monitor = ResourceMonitor()
        monitor.force_tier(ResourceTier.READONLY)
        assert monitor.get_tier() == ResourceTier.READONLY
        assert monitor.can_write() is False

        monitor.force_tier(None)
        assert monitor.get_tier() == ResourceTier.FULL

    def test_status_returns_diagnostics(self):
        from engram.resource_tier import ResourceMonitor

        monitor = ResourceMonitor()
        status = monitor.status()
        assert "tier" in status
        assert "recent_failures" in status
        assert "failure_threshold" in status

    def test_global_singleton(self):
        from engram.resource_tier import get_resource_monitor, setup_resource_monitor

        monitor = setup_resource_monitor(failure_threshold=5)
        assert get_resource_monitor() is monitor
        assert monitor._failure_threshold == 5

    def test_transient_failures_degrade_to_standard(self):
        from engram.resource_tier import ResourceMonitor, ResourceTier

        monitor = ResourceMonitor(failure_threshold=3)
        monitor.record_failure("transient")
        tier = monitor.get_tier()
        assert tier in (ResourceTier.STANDARD, ResourceTier.FULL)

    def test_can_write_on_non_readonly(self):
        from engram.resource_tier import ResourceMonitor

        monitor = ResourceMonitor()
        assert monitor.can_write() is True
        # Even on BASIC, writes are allowed
        monitor.record_failure("rate_limit")
        monitor.record_failure("rate_limit")
        monitor.record_failure("rate_limit")
        assert monitor.can_write() is True


# === Feature 3: Data Constitution ===


class TestConstitution:
    """Test constitution loading, hashing, and prompt injection."""

    def test_default_constitution_content(self):
        from engram.constitution import DEFAULT_CONSTITUTION

        assert "Law I" in DEFAULT_CONSTITUTION
        assert "Law II" in DEFAULT_CONSTITUTION
        assert "Law III" in DEFAULT_CONSTITUTION
        assert "namespace" in DEFAULT_CONSTITUTION.lower()
        assert "fabricat" in DEFAULT_CONSTITUTION.lower()
        assert "audit" in DEFAULT_CONSTITUTION.lower()

    def test_constitution_prompt_prefix_contains_rules(self):
        from engram.constitution import get_constitution_prompt_prefix

        prefix = get_constitution_prompt_prefix()
        assert "CONSTITUTION" in prefix
        assert "namespace" in prefix.lower()
        assert "fabricat" in prefix.lower() or "hallucinate" in prefix.lower()
        assert "logged" in prefix.lower()

    def test_compute_hash_deterministic(self):
        from engram.constitution import compute_constitution_hash

        h1 = compute_constitution_hash("test content")
        h2 = compute_constitution_hash("test content")
        assert h1 == h2
        assert len(h1) == 16  # truncated SHA-256

    def test_compute_hash_changes_with_content(self):
        from engram.constitution import compute_constitution_hash

        h1 = compute_constitution_hash("content A")
        h2 = compute_constitution_hash("content B")
        assert h1 != h2

    def test_verify_constitution_no_stored_hash(self):
        from engram.constitution import verify_constitution

        is_valid, current_hash = verify_constitution(stored_hash=None)
        assert is_valid is True
        assert len(current_hash) == 16

    def test_verify_constitution_matching_hash(self):
        from engram.constitution import compute_constitution_hash, load_constitution, verify_constitution

        content = load_constitution()
        correct_hash = compute_constitution_hash(content)
        is_valid, _ = verify_constitution(stored_hash=correct_hash)
        assert is_valid is True

    def test_verify_constitution_mismatched_hash(self):
        from engram.constitution import verify_constitution

        is_valid, _ = verify_constitution(stored_hash="wrong_hash_value!")
        assert is_valid is False

    def test_load_constitution_creates_default_file(self, tmp_path):
        from engram.constitution import DEFAULT_CONSTITUTION

        with patch("engram.constitution.get_constitution_path", return_value=tmp_path / "constitution.md"):
            from engram.constitution import load_constitution
            content = load_constitution()
            assert content == DEFAULT_CONSTITUTION
            assert (tmp_path / "constitution.md").exists()

    def test_load_constitution_reads_existing_file(self, tmp_path):
        custom = "# Custom Constitution\nMy rules here."
        (tmp_path / "constitution.md").write_text(custom)

        with patch("engram.constitution.get_constitution_path", return_value=tmp_path / "constitution.md"):
            from engram.constitution import load_constitution
            content = load_constitution()
            assert content == custom


# === Feature 4: Memory Consolidation Scheduler ===


class TestMemoryScheduler:
    """Test background scheduler for periodic maintenance."""

    def test_register_and_status(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler()

        async def dummy_task():
            return {"ok": True}

        scheduler.register("test_task", dummy_task, interval_seconds=3600)
        tasks = scheduler.status()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "test_task"
        assert tasks[0]["interval_seconds"] == 3600
        assert tasks[0]["run_count"] == 0

    def test_register_multiple_tasks(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler()

        async def task_a():
            return {}

        async def task_b():
            return {}

        scheduler.register("a", task_a, 100)
        scheduler.register("b", task_b, 200, requires_llm=True)
        tasks = scheduler.status()
        assert len(tasks) == 2
        names = {t["name"] for t in tasks}
        assert names == {"a", "b"}

    @pytest.mark.asyncio
    async def test_tick_executes_due_task(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler(tick_interval=0.1)
        call_count = 0

        async def counting_task():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        # interval=0 means always due
        scheduler.register("counter", counting_task, interval_seconds=0)
        await scheduler._tick()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_tick_skips_not_due_task(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler()
        called = False

        async def should_not_run():
            nonlocal called
            called = True
            return {}

        scheduler.register("future_task", should_not_run, interval_seconds=999999)
        # Set last_run to now so it's not due
        scheduler._tasks["future_task"].last_run = time.monotonic()
        await scheduler._tick()
        assert called is False

    @pytest.mark.asyncio
    async def test_tick_skips_llm_task_on_basic_tier(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler()
        called = False

        async def llm_task():
            nonlocal called
            called = True
            return {}

        scheduler.register("llm_thing", llm_task, interval_seconds=0, requires_llm=True)

        with patch("engram.resource_tier.get_resource_monitor") as mock_get:
            mock_mon = MagicMock()
            mock_mon.can_use_llm.return_value = False
            mock_get.return_value = mock_mon
            await scheduler._tick()

        assert called is False

    @pytest.mark.asyncio
    async def test_tick_handles_task_error_gracefully(self):
        from engram.scheduler import MemoryScheduler

        scheduler = MemoryScheduler()

        async def failing_task():
            raise RuntimeError("boom")

        scheduler.register("failing", failing_task, interval_seconds=0)
        # Should not raise
        await scheduler._tick()
        assert scheduler._tasks["failing"].last_error == "boom"

    def test_state_persistence(self, tmp_path):
        from engram.scheduler import MemoryScheduler

        state_file = tmp_path / "state.json"
        scheduler = MemoryScheduler(state_path=str(state_file))

        async def noop():
            return {}

        scheduler.register("persist_test", noop, interval_seconds=100)
        scheduler._tasks["persist_test"].last_run = time.monotonic()
        scheduler._tasks["persist_test"].run_count = 5
        scheduler._save_state()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["persist_test"]["run_count"] == 5

    def test_create_default_scheduler(self):
        from engram.scheduler import create_default_scheduler

        mock_store = MagicMock()
        scheduler = create_default_scheduler(mock_store, consolidation_engine=None)
        tasks = scheduler.status()
        task_names = {t["name"] for t in tasks}
        assert "cleanup_expired" in task_names
        assert "decay_report" in task_names
        # No consolidation without engine
        assert "consolidate_memories" not in task_names

    def test_create_default_scheduler_with_consolidation(self):
        from engram.scheduler import create_default_scheduler

        mock_store = MagicMock()
        mock_engine = MagicMock()
        scheduler = create_default_scheduler(mock_store, consolidation_engine=mock_engine)
        task_names = {t["name"] for t in scheduler.status()}
        assert "consolidate_memories" in task_names


# === Integration: Constitution injected into reasoning engine ===


class TestConstitutionIntegration:
    """Verify constitution prefix is injected into LLM prompts."""

    @pytest.mark.asyncio
    async def test_synthesize_includes_constitution(self):
        from engram.reasoning.engine import ReasoningEngine

        mock_episodic = MagicMock()
        mock_graph = MagicMock()
        engine = ReasoningEngine(mock_episodic, mock_graph, model="test-model")

        with patch("engram.reasoning.engine.litellm") as mock_litellm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="test answer"))]
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            from engram.models import EpisodicMemory, MemoryType
            mem = EpisodicMemory(
                id="test", content="test content",
                memory_type=MemoryType.FACT, priority=5,
                timestamp=datetime.now(timezone.utc),
            )
            result = await engine._synthesize("test question", [mem], {})

            # Verify the prompt sent to LLM contains constitution
            call_args = mock_litellm.acompletion.call_args
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
            prompt_content = messages[0]["content"]
            assert "CONSTITUTION" in prompt_content
            assert "namespace" in prompt_content.lower()


# === Integration: Resource-aware reasoning ===


class TestResourceAwareReasoning:
    """Verify reasoning engine respects resource tier."""

    @pytest.mark.asyncio
    async def test_think_skips_llm_on_basic_tier(self):
        from engram.reasoning.engine import ReasoningEngine

        mock_episodic = MagicMock()
        mock_episodic.search = AsyncMock(return_value=[])
        mock_graph = MagicMock()
        mock_graph.get_related = AsyncMock(return_value={})
        mock_graph.get_nodes = AsyncMock(return_value=[])

        engine = ReasoningEngine(mock_episodic, mock_graph, model="test-model")

        with patch("engram.reasoning.engine.get_resource_monitor") as mock_get_monitor:
            monitor = MagicMock()
            monitor.can_use_llm.return_value = False
            monitor.get_tier.return_value = MagicMock(value="basic")
            mock_get_monitor.return_value = monitor

            with patch("engram.reasoning.engine.federated_search", new_callable=AsyncMock, return_value=[]):
                result = await engine.think("test question")
                assert "No relevant memories" in result
