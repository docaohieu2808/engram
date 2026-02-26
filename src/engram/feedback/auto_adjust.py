"""Feedback-driven confidence and importance adjustment.

Delegates to FeedbackProcessor (feedback/loop.py) for all logic.
This module exists as a thin adapter so callers don't need to
instantiate FeedbackProcessor directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engram.feedback.loop import FeedbackConfig, FeedbackProcessor, FeedbackType

if TYPE_CHECKING:
    from engram.episodic.store import EpisodicStore


async def adjust_memory(
    store: "EpisodicStore",
    memory_id: str,
    feedback: str,  # "positive" or "negative"
    config: FeedbackConfig | None = None,
) -> dict:
    """Adjust memory confidence and importance based on feedback.

    Delegates all logic to FeedbackProcessor, which uses config values
    (no hardcoded constants) and the store's public API (no private access).

    Args:
        store: EpisodicStore instance.
        memory_id: ID of the memory to adjust (8-char prefix accepted).
        feedback: "positive" or "negative".
        config: Optional FeedbackConfig; uses defaults if not provided.

    Returns:
        dict with action, memory_id, confidence, priority, negative_count.
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

    processor = FeedbackProcessor(store, config or FeedbackConfig())
    fb_type = FeedbackType.POSITIVE if feedback == "positive" else FeedbackType.NEGATIVE
    return await processor.apply_feedback(resolved_id, fb_type)
