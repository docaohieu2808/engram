"""Regression tests for cache degradation logging and watcher error visibility.

Covers:
- test_cache_logs_on_redis_failure: EngramCache logs debug when Redis unreachable
- test_watcher_logs_on_move_failure: watcher.py logs warning when failed-dir move errors
"""

from __future__ import annotations

import logging
import shutil

import pytest


# ---------------------------------------------------------------------------
# Cache degradation logging
# ---------------------------------------------------------------------------

class TestCacheDegradationLogging:
    @pytest.mark.asyncio
    async def test_cache_logs_on_redis_failure(self, caplog):
        """Cache.connect() should log debug when Redis is unreachable."""
        from engram.cache import EngramCache

        # Port 9999 is not running Redis — connection will fail immediately
        cache = EngramCache("redis://localhost:9999")
        with caplog.at_level(logging.DEBUG, logger="engram.cache"):
            await cache.connect()

        # Should have degraded gracefully and logged a debug message
        assert cache._redis is None, "Redis handle should be None after failed connect"
        assert any(
            "redis unavailable" in record.message.lower()
            or "degrading" in record.message.lower()
            for record in caplog.records
            if record.name == "engram.cache"
        ), f"Expected degradation log, got: {[r.message for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_cache_get_returns_none_when_degraded(self):
        """Cache.get() should return None (not raise) when Redis is down."""
        from engram.cache import EngramCache

        cache = EngramCache("redis://localhost:9999")
        await cache.connect()
        result = await cache.get("tenant", "recall", {"q": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_is_noop_when_degraded(self):
        """Cache.set() should silently no-op (not raise) when Redis is down."""
        from engram.cache import EngramCache

        cache = EngramCache("redis://localhost:9999")
        await cache.connect()
        # Should not raise
        await cache.set("tenant", "recall", {"q": "test"}, {"result": "ok"})


# ---------------------------------------------------------------------------
# Watcher error visibility
# ---------------------------------------------------------------------------

class TestWatcherErrorVisibility:
    @pytest.mark.asyncio
    async def test_watcher_logs_on_move_failure(self, tmp_path, caplog, monkeypatch):
        """Watcher should log warning when moving to failed-dir raises OSError."""
        from engram.capture.watcher import InboxWatcher
        from engram.capture import watcher as watcher_module

        inbox = tmp_path / "inbox"
        inbox.mkdir()

        # Create a dummy JSON file (watcher scans *.json / *.jsonl)
        test_file = inbox / "test.json"
        test_file.write_text('[{"role": "user", "content": "hello"}]')

        async def always_fail(messages: list) -> None:
            raise RuntimeError("Simulated ingest failure")

        # Override _failed_dir and _processed_dir to use tmp_path
        watcher = InboxWatcher.__new__(InboxWatcher)
        watcher._inbox = inbox
        watcher._processed_dir = tmp_path / "processed"
        watcher._processed_dir.mkdir()
        watcher._failed_dir = tmp_path / "failed"
        watcher._failed_dir.mkdir()
        watcher._ingest_fn = always_fail
        watcher._poll_interval = 0.05
        watcher._running = False
        watcher._retry_counts = {}
        watcher._auto_memory = False
        watcher._episodic_store = None
        watcher._consolidation_trigger = None

        # Set _MAX_RETRIES to 1 so first failure immediately moves to failed/
        monkeypatch.setattr(watcher_module, "_MAX_RETRIES", 1)

        # Make shutil.move always raise OSError (the failed-dir move)
        # Note: path.rename() (not shutil.move) is used for the .processing claim step,
        # so this only intercepts the final move-to-failed-dir call.
        def failing_move(src, dst):
            raise OSError("no space left on device")

        monkeypatch.setattr(shutil, "move", failing_move)

        with caplog.at_level(logging.WARNING, logger="engram"):
            # First pass: ingest fails, retry count set to 1
            # Since _MAX_RETRIES=1, this will immediately try to move to failed/
            file_path = test_file
            try:
                await watcher._process_file(file_path)
            except Exception:
                pass

        # Check that "could not move" warning was logged
        # watcher.py:196 logs: "watcher: could not move %s to failed dir: %s"
        all_messages = [r.getMessage() for r in caplog.records]
        assert any(
            "could not move" in msg.lower()
            for msg in all_messages
        ), f"Expected 'could not move' warning, got: {all_messages}"
