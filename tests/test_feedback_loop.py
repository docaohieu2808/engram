"""Tests for feedback loop — detection and confidence adjustment."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from engram.feedback.loop import detect_feedback, FeedbackProcessor
from engram.config import FeedbackConfig
from engram.models import EpisodicMemory, FeedbackType, MemoryType


class TestDetectFeedback:
    """Test feedback detection from user messages."""

    # Positive patterns
    def test_correct(self):
        assert detect_feedback("correct") == FeedbackType.POSITIVE

    def test_exactly(self):
        assert detect_feedback("exactly!") == FeedbackType.POSITIVE

    def test_dung_roi(self):
        assert detect_feedback("đúng rồi") == FeedbackType.POSITIVE

    def test_chuan(self):
        assert detect_feedback("chuẩn") == FeedbackType.POSITIVE

    def test_good_memory(self):
        assert detect_feedback("good memory!") == FeedbackType.POSITIVE

    def test_nho_dung(self):
        assert detect_feedback("nhớ đúng") == FeedbackType.POSITIVE

    # Negative patterns
    def test_wrong(self):
        assert detect_feedback("wrong") == FeedbackType.NEGATIVE

    def test_incorrect(self):
        assert detect_feedback("that's incorrect") == FeedbackType.NEGATIVE

    def test_sai(self):
        assert detect_feedback("sai rồi") == FeedbackType.NEGATIVE

    def test_khong_dung(self):
        assert detect_feedback("không đúng") == FeedbackType.NEGATIVE

    def test_nham(self):
        assert detect_feedback("nhầm rồi") == FeedbackType.NEGATIVE

    def test_nho_sai(self):
        assert detect_feedback("nhớ sai") == FeedbackType.NEGATIVE

    # No feedback
    def test_neutral_message(self):
        assert detect_feedback("tell me about PostgreSQL") is None

    def test_empty(self):
        assert detect_feedback("") is None

    def test_question(self):
        assert detect_feedback("What time is it?") is None


def _make_memory(confidence: float = 1.0, priority: int = 5, negative_count: int = 0) -> EpisodicMemory:
    return EpisodicMemory(
        id="mem-123",
        content="test memory",
        memory_type=MemoryType.FACT,
        priority=priority,
        confidence=confidence,
        negative_count=negative_count,
        timestamp=datetime.now(timezone.utc),
    )


class TestFeedbackProcessor:
    """Test feedback application to memories."""

    def _make_processor(self, memory: EpisodicMemory | None = None, config: FeedbackConfig | None = None):
        store = AsyncMock()
        store.get = AsyncMock(return_value=memory)
        store.update_metadata = AsyncMock(return_value=True)
        store.delete = AsyncMock(return_value=True)
        return FeedbackProcessor(store, config), store

    @pytest.mark.asyncio
    async def test_positive_increases_confidence(self):
        mem = _make_memory(confidence=0.5, priority=5)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.POSITIVE)

        assert result["action"] == "updated"
        assert result["confidence"] == pytest.approx(0.65)
        assert result["priority"] == 6

    @pytest.mark.asyncio
    async def test_positive_caps_confidence_at_1(self):
        mem = _make_memory(confidence=0.95)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.POSITIVE)
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_positive_caps_priority_at_10(self):
        mem = _make_memory(priority=10)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.POSITIVE)
        assert result["priority"] == 10

    @pytest.mark.asyncio
    async def test_negative_decreases_confidence(self):
        mem = _make_memory(confidence=0.8, priority=5)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.NEGATIVE)

        assert result["action"] == "updated"
        assert result["confidence"] == pytest.approx(0.6)
        assert result["priority"] == 4
        assert result["negative_count"] == 1

    @pytest.mark.asyncio
    async def test_negative_floors_confidence_at_0(self):
        mem = _make_memory(confidence=0.1)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.NEGATIVE)
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_negative_floors_priority_at_1(self):
        mem = _make_memory(priority=1)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.NEGATIVE)
        assert result["priority"] == 1

    @pytest.mark.asyncio
    async def test_auto_delete_after_threshold(self):
        """3 negatives + low confidence → auto-delete."""
        mem = _make_memory(confidence=0.1, negative_count=2)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.NEGATIVE)

        assert result["action"] == "deleted"
        store.delete.assert_called_once_with("mem-123")

    @pytest.mark.asyncio
    async def test_no_auto_delete_if_confidence_high(self):
        """3 negatives but high confidence → don't delete."""
        mem = _make_memory(confidence=0.8, negative_count=2)
        proc, store = self._make_processor(mem)
        result = await proc.apply_feedback("mem-123", FeedbackType.NEGATIVE)

        assert result["action"] == "updated"
        store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_not_found(self):
        proc, store = self._make_processor(memory=None)
        result = await proc.apply_feedback("nonexistent", FeedbackType.POSITIVE)
        assert result["error"] == "memory_not_found"

    @pytest.mark.asyncio
    async def test_custom_config(self):
        config = FeedbackConfig(positive_boost=0.3, negative_penalty=0.5)
        mem = _make_memory(confidence=0.5)
        proc, store = self._make_processor(mem, config)
        result = await proc.apply_feedback("mem-123", FeedbackType.POSITIVE)
        assert result["confidence"] == pytest.approx(0.8)
