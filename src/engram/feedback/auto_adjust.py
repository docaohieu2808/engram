"""Feedback-driven confidence and importance adjustment.

Positive feedback boosts memories, negative feedback degrades them.
Auto-deletes memories after 3× negative with low confidence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.episodic.store import EpisodicStore

logger = logging.getLogger("engram")

# Adjustment constants (match FeedbackConfig defaults)
_POSITIVE_BOOST = 0.15
_NEGATIVE_PENALTY = 0.2
_AUTO_DELETE_THRESHOLD = 3
_MIN_CONFIDENCE_FOR_DELETE = 0.15


async def adjust_memory(
    store: "EpisodicStore",
    memory_id: str,
    feedback: str,  # "positive" or "negative"
) -> dict:
    """Adjust memory confidence and importance based on feedback.

    Rules:
    - positive: confidence +0.15, importance +1 (cap confidence at 1.0, importance at 5)
    - negative: confidence -0.2, importance -1 (floor confidence at 0.0, importance at 1)
    - Track counts in metadata: positive_count, negative_count
    - Auto-delete: if negative_count >= 3 AND confidence < 0.15

    Returns:
        dict with: memory_id, action ("adjusted" or "deleted"), confidence, importance,
        positive_count, negative_count
    """
    if feedback not in ("positive", "negative"):
        return {"error": "feedback must be 'positive' or 'negative'"}

    # Resolve 8-char prefix to full ID if needed
    resolved_id = memory_id
    if len(memory_id) <= 8:
        recent = await store.get_recent(n=500)
        for m in recent:
            if m.id.startswith(memory_id):
                resolved_id = m.id
                break

    # Fetch raw metadata from ChromaDB
    collection = store._ensure_collection()
    try:
        import asyncio
        raw = await asyncio.to_thread(
            collection.get,
            ids=[resolved_id],
            include=["metadatas"],
        )
    except Exception as e:
        logger.warning("Failed to fetch memory %s for feedback: %s", memory_id, e)
        return {"error": "memory_not_found", "memory_id": memory_id}

    if not raw["ids"]:
        return {"error": "memory_not_found", "memory_id": memory_id}

    meta = raw["metadatas"][0] if raw["metadatas"] else {}

    # Extract current values with defaults
    confidence = float(meta.get("confidence", 0.5))
    importance = int(meta.get("priority", 3))
    positive_count = int(meta.get("positive_count", 0))
    negative_count = int(meta.get("negative_count", 0))

    # Apply adjustment
    if feedback == "positive":
        confidence = min(1.0, confidence + _POSITIVE_BOOST)
        importance = min(5, importance + 1)
        positive_count += 1
    else:  # negative
        confidence = max(0.0, confidence - _NEGATIVE_PENALTY)
        importance = max(1, importance - 1)
        negative_count += 1

    # Check auto-delete condition
    if negative_count >= _AUTO_DELETE_THRESHOLD and confidence < _MIN_CONFIDENCE_FOR_DELETE:
        await store.delete(resolved_id)
        logger.info(
            "Auto-deleted memory %s (negative_count=%d, confidence=%.2f)",
            memory_id[:8], negative_count, confidence,
        )
        return {
            "memory_id": resolved_id,
            "action": "deleted",
            "confidence": confidence,
            "importance": importance,
            "positive_count": positive_count,
            "negative_count": negative_count,
        }

    # Update metadata with adjusted values
    updated = {
        "confidence": confidence,
        "priority": importance,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "last_feedback_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.update_metadata(resolved_id, updated)
    logger.debug(
        "Applied %s feedback to %s: confidence=%.2f, importance=%d",
        feedback, memory_id[:8], confidence, importance,
    )

    return {
        "memory_id": resolved_id,
        "action": "adjusted",
        "confidence": confidence,
        "importance": importance,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }
