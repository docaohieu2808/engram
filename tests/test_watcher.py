"""Tests for InboxWatcher file processing logic (capture/watcher.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from engram.capture.watcher import InboxWatcher, _MAX_RETRIES


# --- Fixtures ---

@pytest.fixture
def inbox_dir(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    return d


@pytest.fixture
def watcher(inbox_dir):
    ingest_fn = AsyncMock()
    w = InboxWatcher(
        inbox_dir=str(inbox_dir),
        ingest_fn=ingest_fn,
        poll_interval=1,
    )
    return w


# --- _load_chat_file tests ---

class TestLoadChatFile:
    def test_load_json_array(self, tmp_path):
        msgs = [{"role": "user", "content": "hello"}]
        f = tmp_path / "chat.json"
        f.write_text(json.dumps(msgs))
        result = InboxWatcher._load_chat_file(f)
        assert result == msgs

    def test_load_json_with_messages_key(self, tmp_path):
        msgs = [{"role": "assistant", "content": "hi"}]
        f = tmp_path / "chat.json"
        f.write_text(json.dumps({"messages": msgs, "model": "gpt-4"}))
        result = InboxWatcher._load_chat_file(f)
        assert result == msgs

    def test_load_jsonl_format(self, tmp_path):
        lines = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        f = tmp_path / "chat.jsonl"
        f.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        result = InboxWatcher._load_chat_file(f)
        assert result == lines

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        f = tmp_path / "chat.jsonl"
        f.write_text('{"role":"user","content":"a"}\n\n{"role":"user","content":"b"}\n')
        result = InboxWatcher._load_chat_file(f)
        assert len(result) == 2

    def test_load_jsonl_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "chat.jsonl"
        f.write_text('{"role":"user","content":"ok"}\nnot-json\n')
        result = InboxWatcher._load_chat_file(f)
        assert len(result) == 1
        assert result[0]["content"] == "ok"

    def test_load_invalid_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json at all {{")
        with pytest.raises(Exception):
            InboxWatcher._load_chat_file(f)

    def test_load_json_unknown_structure_returns_empty(self, tmp_path):
        """JSON dict without 'messages' key returns empty list."""
        f = tmp_path / "chat.json"
        f.write_text(json.dumps({"other_key": "value"}))
        result = InboxWatcher._load_chat_file(f)
        assert result == []

    def test_explicit_jsonl_flag_overrides_extension(self, tmp_path):
        """is_jsonl=True forces JSONL parsing even for .json extension."""
        f = tmp_path / "file.json"
        f.write_text('{"role":"user","content":"x"}\n')
        result = InboxWatcher._load_chat_file(f, is_jsonl=True)
        assert len(result) == 1


# --- Retry and failure tests ---

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_counter_increments_on_failure(self, watcher, inbox_dir):
        """Retry count increases each time processing fails."""
        bad_file = inbox_dir / "bad.json"
        bad_file.write_text("not json {{{")

        await watcher._process_file(bad_file)
        assert watcher._retry_counts.get("bad.json", 0) == 1

        # Restore for second attempt
        bad_file.write_text("not json {{{")
        await watcher._process_file(bad_file)
        assert watcher._retry_counts.get("bad.json", 0) == 2

    @pytest.mark.asyncio
    async def test_file_moves_to_failed_after_max_retries(self, watcher, inbox_dir):
        """After _MAX_RETRIES failures, file is moved to failed/ directory."""
        bad_file = inbox_dir / "broken.json"

        for attempt in range(_MAX_RETRIES):
            bad_file.write_text("not json {{{")
            await watcher._process_file(bad_file)

        # File should be in failed dir, not inbox
        assert not bad_file.exists()
        assert (watcher._failed_dir / "broken.json").exists()
        # Retry count reset after moving to failed
        assert "broken.json" not in watcher._retry_counts

    @pytest.mark.asyncio
    async def test_successful_processing_resets_retry_count(self, watcher, inbox_dir):
        """On success, retry counter is cleared."""
        # Simulate one prior failure
        watcher._retry_counts["chat.json"] = 1

        good_file = inbox_dir / "chat.json"
        good_file.write_text(json.dumps([{"role": "user", "content": "hi"}]))

        await watcher._process_file(good_file)
        assert "chat.json" not in watcher._retry_counts

    @pytest.mark.asyncio
    async def test_successful_file_moves_to_processed(self, watcher, inbox_dir):
        """Successfully processed file lands in processed/ directory."""
        import uuid
        unique_name = f"ok_{uuid.uuid4().hex}.json"
        f = inbox_dir / unique_name
        f.write_text(json.dumps([{"role": "user", "content": "test"}]))

        before = set(watcher._processed_dir.iterdir())
        await watcher._process_file(f)
        after = set(watcher._processed_dir.iterdir())

        assert not f.exists()
        new_files = after - before
        assert len(new_files) == 1
        assert unique_name in new_files.pop().name

    @pytest.mark.asyncio
    async def test_already_claimed_file_skipped(self, watcher, inbox_dir):
        """If file disappears before rename (race), processing is skipped silently."""
        missing = inbox_dir / "gone.json"
        # Don't create the file — simulates another process claiming it
        await watcher._process_file(missing)
        # No errors, no retries recorded
        assert "gone.json" not in watcher._retry_counts


# --- Orphan recovery tests ---

class TestOrphanRecovery:
    def test_old_processing_file_renamed_back(self, watcher, inbox_dir):
        """Processing file older than 1 hour is renamed back to original."""
        proc_file = inbox_dir / "old_chat.json.processing"
        proc_file.write_text(json.dumps([]))

        # Set mtime to 2 hours ago
        old_mtime = time.time() - 7200
        import os
        os.utime(proc_file, (old_mtime, old_mtime))

        watcher._recover_orphaned_files()

        assert not proc_file.exists()
        # Should recover to a .json file (original extension)
        recovered = inbox_dir / "old_chat.json"
        assert recovered.exists()

    def test_recent_processing_file_not_touched(self, watcher, inbox_dir):
        """Processing file under 1 hour old is left alone."""
        proc_file = inbox_dir / "recent.json.processing"
        proc_file.write_text(json.dumps([]))
        # mtime is "now" by default

        watcher._recover_orphaned_files()

        # Still exists as .processing
        assert proc_file.exists()
        assert not (inbox_dir / "recent.json").exists()


# --- _process_inbox integration ---

class TestProcessInbox:
    @pytest.mark.asyncio
    async def test_processes_all_json_files(self, watcher, inbox_dir):
        for i in range(3):
            (inbox_dir / f"chat{i}.json").write_text(
                json.dumps([{"role": "user", "content": f"msg{i}"}])
            )

        await watcher._process_inbox()

        assert watcher._ingest_fn.await_count == 3

    @pytest.mark.asyncio
    async def test_skips_dotfiles(self, watcher, inbox_dir):
        (inbox_dir / ".hidden.json").write_text(json.dumps([]))
        (inbox_dir / "visible.json").write_text(
            json.dumps([{"role": "user", "content": "hi"}])
        )

        await watcher._process_inbox()

        assert watcher._ingest_fn.await_count == 1

    @pytest.mark.asyncio
    async def test_processes_jsonl_files(self, watcher, inbox_dir):
        f = inbox_dir / "log.jsonl"
        f.write_text('{"role":"user","content":"test"}\n')

        await watcher._process_inbox()

        assert watcher._ingest_fn.await_count == 1
