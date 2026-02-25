"""Tests for feedback-driven confidence and importance adjustment."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.feedback.auto_adjust import adjust_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(meta: dict, mem_id: str = "test-memory-id-full-1234") -> MagicMock:
    """Build a minimal mock EpisodicStore with given metadata."""
    store = MagicMock()
    store.get_recent = AsyncMock(return_value=[])

    collection = MagicMock()
    collection.get = MagicMock(return_value={"ids": [mem_id], "metadatas": [meta]})
    store._ensure_collection = MagicMock(return_value=collection)
    store.update_metadata = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    return store


# ---------------------------------------------------------------------------
# Positive feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_feedback_increases_confidence():
    meta = {"confidence": 0.5, "priority": 3, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["action"] == "adjusted"
    assert abs(result["confidence"] - 0.65) < 1e-9


@pytest.mark.asyncio
async def test_positive_feedback_increases_importance():
    meta = {"confidence": 0.5, "priority": 3, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["importance"] == 4


@pytest.mark.asyncio
async def test_positive_count_incremented():
    meta = {"confidence": 0.5, "priority": 3, "positive_count": 2, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["positive_count"] == 3


@pytest.mark.asyncio
async def test_multiple_positive_feedbacks_accumulate():
    meta = {"confidence": 0.5, "priority": 3, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)

    # First positive
    r1 = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert r1["positive_count"] == 1
    assert abs(r1["confidence"] - 0.65) < 1e-9

    # Simulate updated metadata for second call
    meta2 = {
        "confidence": r1["confidence"],
        "priority": r1["importance"],
        "positive_count": r1["positive_count"],
        "negative_count": r1["negative_count"],
    }
    store2 = _make_store(meta2)
    r2 = await adjust_memory(store2, "test-memory-id-full-1234", "positive")
    assert r2["positive_count"] == 2
    assert abs(r2["confidence"] - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# Negative feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_negative_feedback_decreases_confidence():
    meta = {"confidence": 0.8, "priority": 4, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["action"] == "adjusted"
    assert abs(result["confidence"] - 0.6) < 1e-9


@pytest.mark.asyncio
async def test_negative_feedback_decreases_importance():
    meta = {"confidence": 0.8, "priority": 4, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["importance"] == 3


@pytest.mark.asyncio
async def test_negative_count_incremented():
    meta = {"confidence": 0.8, "priority": 4, "positive_count": 0, "negative_count": 1}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["negative_count"] == 2


# ---------------------------------------------------------------------------
# Bounds / capping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confidence_capped_at_1():
    meta = {"confidence": 0.95, "priority": 3, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_confidence_floored_at_0():
    meta = {"confidence": 0.1, "priority": 3, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["confidence"] >= 0.0


@pytest.mark.asyncio
async def test_importance_capped_at_5():
    meta = {"confidence": 0.5, "priority": 5, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["importance"] == 5


@pytest.mark.asyncio
async def test_importance_floored_at_1():
    meta = {"confidence": 0.5, "priority": 1, "positive_count": 0, "negative_count": 0}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["importance"] == 1


# ---------------------------------------------------------------------------
# Auto-delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_delete_after_3_negative_with_low_confidence():
    # confidence will drop from 0.1 to 0.0 after negative → auto-delete
    meta = {"confidence": 0.1, "priority": 2, "positive_count": 0, "negative_count": 2}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    assert result["action"] == "deleted"
    store.delete.assert_called_once()


@pytest.mark.asyncio
async def test_no_auto_delete_at_2_negative():
    # negative_count will become 2, not yet at threshold
    meta = {"confidence": 0.1, "priority": 2, "positive_count": 0, "negative_count": 1}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    # confidence 0.1 - 0.2 = 0.0, negative_count = 2 (< threshold 3) → no delete
    assert result["action"] == "adjusted"
    store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_no_auto_delete_when_confidence_high():
    # negative_count hits threshold but confidence is still high
    meta = {"confidence": 0.5, "priority": 2, "positive_count": 0, "negative_count": 2}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "negative")
    # confidence 0.5 - 0.2 = 0.3 >= 0.15 → no delete despite negative_count = 3
    assert result["action"] == "adjusted"
    store.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Tracking both counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_and_negative_counts_tracked_independently():
    meta = {"confidence": 0.5, "priority": 3, "positive_count": 3, "negative_count": 1}
    store = _make_store(meta)
    result = await adjust_memory(store, "test-memory-id-full-1234", "positive")
    assert result["positive_count"] == 4
    assert result["negative_count"] == 1  # unchanged


@pytest.mark.asyncio
async def test_memory_not_found_returns_error():
    store = MagicMock()
    store.get_recent = AsyncMock(return_value=[])
    collection = MagicMock()
    collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
    store._ensure_collection = MagicMock(return_value=collection)
    result = await adjust_memory(store, "nonexistent", "positive")
    assert "error" in result
