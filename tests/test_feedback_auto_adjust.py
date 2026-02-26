"""Tests for feedback-driven confidence and importance adjustment.

Tests verify the thin-adapter behavior of adjust_memory(), which delegates
to FeedbackProcessor.apply_feedback(). The store mock uses store.get() and
store.delete()/store.update_metadata() as called by FeedbackProcessor.

FeedbackConfig defaults:
  positive_boost = 0.15
  negative_penalty = 0.2
  auto_delete_threshold = 3
  min_confidence_for_delete = 0.15

FeedbackProcessor.apply_feedback() returns:
  adjusted: {"action": "updated", "memory_id", "confidence", "priority", "negative_count"}
  deleted:  {"action": "deleted", "memory_id"}
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.feedback.auto_adjust import adjust_memory
from engram.models import EpisodicMemory, MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    confidence: float = 0.5,
    priority: int = 3,
    negative_count: int = 0,
    mem_id: str = "test-memory-id-full-1234",
) -> EpisodicMemory:
    """Build a minimal EpisodicMemory with given fields."""
    return EpisodicMemory(
        id=mem_id,
        content="test content",
        memory_type=MemoryType.FACT,
        priority=priority,
        confidence=confidence,
        negative_count=negative_count,
    )


def _make_store(memory: EpisodicMemory | None) -> MagicMock:
    """Build a mock EpisodicStore whose .get() returns the given memory."""
    store = MagicMock()
    store.get_recent = AsyncMock(return_value=[])
    store.get = AsyncMock(return_value=memory)
    store.update_metadata = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    return store


# ---------------------------------------------------------------------------
# Positive feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_feedback_increases_confidence():
    mem = _make_memory(confidence=0.5, priority=3)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "positive")
    assert result["action"] == "updated"
    assert abs(result["confidence"] - 0.65) < 1e-9


@pytest.mark.asyncio
async def test_positive_feedback_increases_priority():
    mem = _make_memory(confidence=0.5, priority=3)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "positive")
    assert result["priority"] == 4


@pytest.mark.asyncio
async def test_negative_count_unchanged_on_positive():
    mem = _make_memory(confidence=0.5, priority=3, negative_count=2)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "positive")
    # negative_count not incremented on positive feedback
    assert result["negative_count"] == 2


@pytest.mark.asyncio
async def test_multiple_positive_feedbacks_accumulate():
    mem = _make_memory(confidence=0.5, priority=3)
    store = _make_store(mem)

    # First positive: 0.5 + 0.15 = 0.65
    r1 = await adjust_memory(store, mem.id, "positive")
    assert abs(r1["confidence"] - 0.65) < 1e-9
    assert r1["priority"] == 4

    # Simulate updated memory for second call
    mem2 = _make_memory(confidence=r1["confidence"], priority=r1["priority"])
    store2 = _make_store(mem2)

    # Second positive: 0.65 + 0.15 = 0.80
    r2 = await adjust_memory(store2, mem.id, "positive")
    assert abs(r2["confidence"] - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# Negative feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_negative_feedback_decreases_confidence():
    mem = _make_memory(confidence=0.8, priority=4)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["action"] == "updated"
    assert abs(result["confidence"] - 0.6) < 1e-9


@pytest.mark.asyncio
async def test_negative_feedback_decreases_priority():
    mem = _make_memory(confidence=0.8, priority=4)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["priority"] == 3


@pytest.mark.asyncio
async def test_negative_count_incremented():
    mem = _make_memory(confidence=0.8, priority=4, negative_count=1)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["negative_count"] == 2


# ---------------------------------------------------------------------------
# Bounds / capping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confidence_capped_at_1():
    mem = _make_memory(confidence=0.95, priority=3)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "positive")
    assert result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_confidence_floored_at_0():
    mem = _make_memory(confidence=0.1, priority=3)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["confidence"] >= 0.0


@pytest.mark.asyncio
async def test_priority_capped_at_10():
    # priority 10 stays at 10 after positive feedback
    mem = _make_memory(confidence=0.5, priority=10)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "positive")
    assert result["priority"] == 10


@pytest.mark.asyncio
async def test_priority_floored_at_1():
    mem = _make_memory(confidence=0.5, priority=1)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["priority"] == 1


# ---------------------------------------------------------------------------
# Auto-delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_delete_after_3_negative_with_low_confidence():
    # confidence 0.1 - 0.2 = 0.0 (< 0.15), negative_count 2 → 3 (>= threshold 3) → delete
    mem = _make_memory(confidence=0.1, priority=2, negative_count=2)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["action"] == "deleted"
    store.delete.assert_called_once()


@pytest.mark.asyncio
async def test_no_auto_delete_at_2_negative():
    # negative_count 1 → 2 (< threshold 3) → no delete
    mem = _make_memory(confidence=0.1, priority=2, negative_count=1)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["action"] == "updated"
    store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_no_auto_delete_when_confidence_high():
    # negative_count 2 → 3 (>= threshold), but confidence 0.5 - 0.2 = 0.3 >= 0.15 → no delete
    mem = _make_memory(confidence=0.5, priority=2, negative_count=2)
    store = _make_store(mem)
    result = await adjust_memory(store, mem.id, "negative")
    assert result["action"] == "updated"
    store.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Invalid feedback value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_feedback_returns_error():
    store = _make_store(None)
    result = await adjust_memory(store, "some-id", "neutral")
    assert "error" in result


# ---------------------------------------------------------------------------
# Memory not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_not_found_returns_error():
    # store.get() returns None → FeedbackProcessor returns error
    store = _make_store(None)
    result = await adjust_memory(store, "nonexistent-id-longer-than-8", "positive")
    assert "error" in result


# ---------------------------------------------------------------------------
# Short ID prefix resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_prefix_resolved_via_get_recent():
    """8-char prefix is resolved to full ID via store.get_recent."""
    mem = _make_memory(mem_id="abcdef1234567890")
    store = _make_store(mem)
    # Simulate get_recent returning a memory whose id starts with "abcdef12"
    store.get_recent = AsyncMock(return_value=[mem])
    result = await adjust_memory(store, "abcdef12", "positive")
    # Should resolve and succeed
    assert result["action"] == "updated"
