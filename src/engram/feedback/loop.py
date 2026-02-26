"""Feedback loop — detect user feedback and adjust memory confidence.

Detects positive/negative signals in user messages and adjusts
confidence and importance of recently recalled memories.
Auto-deletes memories with repeated negative feedback and low confidence.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from engram.config import FeedbackConfig
from engram.models import FeedbackType

if TYPE_CHECKING:
    from engram.episodic.store import EpisodicStore

logger = logging.getLogger("engram")

# Patterns indicating user confirms a memory is correct
POSITIVE_PATTERNS = [
    re.compile(r"\b(correct|exactly|right|yes that'?s right)\b", re.IGNORECASE),
    re.compile(r"\b(đúng|chuẩn|chính xác|đúng rồi)\b", re.IGNORECASE),
    re.compile(r"\b(good memory|nhớ tốt|nhớ đúng|giỏi)\b", re.IGNORECASE),
]

# Patterns indicating user says a memory is wrong
NEGATIVE_PATTERNS = [
    re.compile(r"\b(wrong|incorrect|not right|that'?s not right)\b", re.IGNORECASE),
    re.compile(r"\b(sai|không đúng|nhầm|không phải|sai rồi)\b", re.IGNORECASE),
    re.compile(r"\b(bad memory|nhớ sai|không nhớ đúng)\b", re.IGNORECASE),
]


def detect_feedback(message: str) -> FeedbackType | None:
    """Detect if a message contains memory feedback.

    Returns FeedbackType.POSITIVE, FeedbackType.NEGATIVE, or None.
    """
    msg = message.strip()
    if not msg:
        return None

    # Check negative FIRST — "không đúng" contains "đúng" but is negative
    for pattern in NEGATIVE_PATTERNS:
        if pattern.search(msg):
            return FeedbackType.NEGATIVE

    for pattern in POSITIVE_PATTERNS:
        if pattern.search(msg):
            return FeedbackType.POSITIVE

    return None


class FeedbackProcessor:
    """Process user feedback to adjust memory confidence and importance."""

    def __init__(self, store: EpisodicStore, config: FeedbackConfig | None = None):
        self._store = store
        self._config = config or FeedbackConfig()
        self._locks: dict[str, asyncio.Lock] = {}

    async def apply_feedback(self, memory_id: str, feedback: FeedbackType) -> dict:
        """Apply feedback to a memory. Returns updated metadata dict.

        POSITIVE: confidence += boost, priority += 1 (capped at 10)
        NEGATIVE: confidence -= penalty, priority -= 1 (min 1), negative_count += 1
        Auto-delete if negative_count >= threshold AND confidence < min_confidence.
        Per-memory lock prevents lost updates from concurrent feedback.
        """
        if memory_id not in self._locks:
            self._locks[memory_id] = asyncio.Lock()

        async with self._locks[memory_id]:
            memory = await self._store.get(memory_id)
            if memory is None:
                logger.warning("Feedback target memory %s not found", memory_id)
                return {"error": "memory_not_found"}

            confidence = memory.confidence
            priority = memory.priority
            negative_count = memory.negative_count
            now = datetime.now(timezone.utc).isoformat()

            if feedback == FeedbackType.POSITIVE:
                confidence = min(1.0, confidence + self._config.positive_boost)
                priority = min(10, priority + 1)
            elif feedback == FeedbackType.NEGATIVE:
                confidence = max(0.0, confidence - self._config.negative_penalty)
                priority = max(1, priority - 1)
                negative_count += 1

            # Check auto-delete condition
            if (
                negative_count >= self._config.auto_delete_threshold
                and confidence < self._config.min_confidence_for_delete
            ):
                await self._store.delete(memory_id)
                logger.info(
                    "Auto-deleted memory %s (negative_count=%d, confidence=%.2f)",
                    memory_id[:8], negative_count, confidence,
                )
                result: dict = {"action": "deleted", "memory_id": memory_id}
            else:
                # Update metadata
                update = {
                    "confidence": confidence,
                    "priority": priority,
                    "negative_count": negative_count,
                    "last_feedback_at": now,
                }
                await self._store.update_metadata(memory_id, update)
                logger.debug(
                    "Applied %s feedback to %s: confidence=%.2f, priority=%d",
                    feedback.value, memory_id[:8], confidence, priority,
                )
                result = {
                    "action": "updated",
                    "memory_id": memory_id,
                    "confidence": confidence,
                    "priority": priority,
                    "negative_count": negative_count,
                }

        # Cleanup lock if no longer held (prevents unbounded growth)
        if memory_id in self._locks and not self._locks[memory_id].locked():
            del self._locks[memory_id]

        return result
