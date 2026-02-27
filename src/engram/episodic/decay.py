"""Decay scoring for episodic memory activation.

Extracted from store.py to keep files under 200 LOC.
Implements Ebbinghaus forgetting curve with recency and frequency boosts.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from engram.config import ScoringConfig


def compute_activation_score(
    similarity: float,
    timestamp: datetime,
    access_count: int,
    decay_rate: float,
    now: datetime,
    scoring: ScoringConfig,
    decay_enabled: bool,
) -> float:
    """Compute composite activation score for a memory.

    Components: vector similarity, Ebbinghaus retention, recency, frequency.

    Args:
        similarity: Cosine similarity in [0, 1].
        timestamp: When the memory was created.
        access_count: How many times this memory has been accessed.
        decay_rate: Per-memory decay coefficient (default 0.1).
        now: Current UTC datetime for age calculation.
        scoring: Weight config from ScoringConfig.
        decay_enabled: When False, return raw similarity unchanged.
    """
    if not decay_enabled:
        return similarity

    # Ensure both datetimes are tz-aware for safe subtraction
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    days_old = max(0.0, (now - ts).total_seconds() / 86400)

    # Ebbinghaus retention with access reinforcement
    retention = math.exp(-decay_rate * days_old / (1 + 0.1 * access_count))
    # Recency boost (newer = higher)
    recency = 1.0 / (1.0 + days_old * 0.1)
    # Frequency boost (myelination metaphor)
    frequency = 1.0 + min(0.3, 0.05 * math.log1p(access_count))

    return (
        similarity * scoring.similarity_weight
        + retention * scoring.retention_weight
        + recency * scoring.recency_weight
        + frequency * scoring.frequency_weight
    )
