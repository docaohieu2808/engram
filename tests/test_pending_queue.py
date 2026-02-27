"""Tests for PendingQueue fail-open embedding retry queue."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.episodic.pending_queue import PendingQueue


@pytest.fixture
def queue(tmp_path):
    return PendingQueue(path=str(tmp_path / "pending_queue.jsonl"))


# ---------------------------------------------------------------------------
# enqueue / count
# ---------------------------------------------------------------------------

def test_enqueue_creates_file(queue):
    assert queue.count() == 0
    queue.enqueue("hello world")
    assert queue.path.exists()
    assert queue.count() == 1


def test_enqueue_multiple(queue):
    for i in range(5):
        queue.enqueue(f"message {i}")
    assert queue.count() == 5


def test_enqueue_stores_all_fields(queue):
    queue.enqueue("test content", timestamp="2026-01-01T00:00:00+00:00", metadata={"key": "val"})
    line = queue.path.read_text().strip()
    data = json.loads(line)
    assert data["content"] == "test content"
    assert data["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert data["metadata"] == {"key": "val"}
    assert "enqueued_at" in data


def test_count_empty_when_file_missing(queue):
    assert queue.count() == 0


# ---------------------------------------------------------------------------
# drain — working store (items removed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_with_working_store(queue):
    queue.enqueue("first")
    queue.enqueue("second")

    store = MagicMock()
    store.remember = AsyncMock(return_value="fake-id")

    success, failed = await queue.drain(store)

    assert success == 2
    assert failed == 0
    assert queue.count() == 0
    assert not queue.path.exists()


@pytest.mark.asyncio
async def test_drain_passes_content_and_timestamp(queue):
    queue.enqueue("my content", timestamp="2026-02-01T12:00:00+00:00")

    calls = []

    async def capture_remember(**kwargs):
        calls.append(kwargs)
        return "id"

    store = MagicMock()
    store.remember = capture_remember

    await queue.drain(store)

    assert len(calls) == 1
    assert calls[0]["content"] == "my content"
    assert calls[0]["timestamp"].isoformat().startswith("2026-02-01")


# ---------------------------------------------------------------------------
# drain — failing store (items stay in queue)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_with_failing_store_keeps_items(queue):
    queue.enqueue("message A")
    queue.enqueue("message B")

    store = MagicMock()
    store.remember = AsyncMock(side_effect=RuntimeError("embedding down"))

    success, failed = await queue.drain(store)

    assert success == 0
    assert failed == 2
    assert queue.count() == 2


@pytest.mark.asyncio
async def test_drain_partial_success(queue):
    queue.enqueue("good")
    queue.enqueue("bad")

    call_count = 0

    async def flaky_remember(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("fail")
        return "id"

    store = MagicMock()
    store.remember = flaky_remember

    success, failed = await queue.drain(store)

    assert success == 1
    assert failed == 1
    assert queue.count() == 1


# ---------------------------------------------------------------------------
# drain — empty queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_empty_queue(queue):
    store = MagicMock()
    store.remember = AsyncMock(return_value="id")
    success, failed = await queue.drain(store)
    assert success == 0
    assert failed == 0


# ---------------------------------------------------------------------------
# concurrent enqueue safety
# ---------------------------------------------------------------------------

def test_concurrent_enqueue(queue):
    errors = []

    def enqueue_many():
        for i in range(50):
            try:
                queue.enqueue(f"concurrent message {i}")
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=enqueue_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent enqueue raised: {errors}"
    assert queue.count() == 200  # 4 threads × 50 messages
