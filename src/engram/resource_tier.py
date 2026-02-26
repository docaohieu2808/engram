"""Resource-aware tier system for degraded operation when LLM/API is unavailable.

Tiers determine which features are available based on operational resources:
- FULL: All features — dual-memory, LLM reasoning, consolidation
- STANDARD: Dual-memory + LLM reasoning, no background consolidation
- BASIC: Vector search only, no LLM calls (rate limit / quota exhausted)
- READONLY: Read-only retrieval, no ingestion or writes

Tier is evaluated dynamically based on recent LLM call success/failure.
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum

logger = logging.getLogger("engram")

# Exponential backoff constants for tier recovery (M16 oscillation fix)
_BACKOFF_BASE = 2.0   # seconds
_BACKOFF_MAX = 3600.0  # 1 hour ceiling
_JITTER_RANGE = 0.25   # ±25% jitter


class ResourceTier(Enum):
    FULL = "full"
    STANDARD = "standard"
    BASIC = "basic"
    READONLY = "readonly"


class ResourceMonitor:
    """Tracks LLM call success/failure to determine current resource tier.

    Uses a sliding window of recent call outcomes. When failure rate
    exceeds threshold, tier degrades. Auto-recovers after cooldown.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        window_seconds: float = 300.0,
    ):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._window_seconds = window_seconds
        # Track recent failures: list of (timestamp, error_type)
        self._recent_failures: list[tuple[float, str]] = []
        self._last_success: float = time.monotonic()
        self._forced_tier: ResourceTier | None = None
        # Backoff state for tier recovery (M16 oscillation prevention)
        self._consecutive_failures: int = 0
        self._recovery_cooldown_until: float = 0.0

    def record_success(self) -> None:
        """Record a successful LLM call."""
        self._last_success = time.monotonic()
        self._consecutive_failures = 0
        self._recovery_cooldown_until = 0.0

    def record_failure(self, error_type: str = "unknown") -> None:
        """Record a failed LLM call (rate limit, quota, timeout)."""
        now = time.monotonic()
        self._recent_failures.append((now, error_type))
        self._consecutive_failures += 1
        # Exponential backoff with jitter to prevent oscillation (M16)
        backoff = min(_BACKOFF_BASE ** self._consecutive_failures, _BACKOFF_MAX)
        jitter = backoff * _JITTER_RANGE * (2.0 * random.random() - 1.0)
        self._recovery_cooldown_until = now + backoff + jitter

    def force_tier(self, tier: ResourceTier | None) -> None:
        """Manually override tier (for admin/testing). Pass None to clear."""
        self._forced_tier = tier

    def get_tier(self) -> ResourceTier:
        """Evaluate current resource tier based on recent LLM call history."""
        if self._forced_tier is not None:
            return self._forced_tier

        now = time.monotonic()
        # Prune old failures outside window
        cutoff = now - self._window_seconds
        self._recent_failures = [(t, e) for t, e in self._recent_failures if t > cutoff]

        failure_count = len(self._recent_failures)

        if failure_count == 0:
            return ResourceTier.FULL

        # Respect exponential backoff cooldown before attempting recovery (M16)
        if now < self._recovery_cooldown_until:
            # Still in backoff period — stay degraded
            pass
        elif self._recent_failures:
            last_failure_time = self._recent_failures[-1][0]
            seconds_since_failure = now - last_failure_time
            if seconds_since_failure > self._cooldown_seconds and failure_count < self._failure_threshold:
                return ResourceTier.STANDARD

        # Check error types — rate limits and quota errors are more severe
        rate_limit_count = sum(1 for _, e in self._recent_failures if e in ("rate_limit", "quota"))
        if rate_limit_count >= self._failure_threshold:
            return ResourceTier.BASIC
        if failure_count >= self._failure_threshold:
            return ResourceTier.BASIC

        return ResourceTier.STANDARD

    def can_use_llm(self) -> bool:
        """Check if LLM calls are currently advisable."""
        tier = self.get_tier()
        return tier in (ResourceTier.FULL, ResourceTier.STANDARD)

    def can_write(self) -> bool:
        """Check if write operations are allowed."""
        return self.get_tier() != ResourceTier.READONLY

    def status(self) -> dict:
        """Return current resource status for diagnostics."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        recent = [(t, e) for t, e in self._recent_failures if t > cutoff]
        return {
            "tier": self.get_tier().value,
            "recent_failures": len(recent),
            "failure_threshold": self._failure_threshold,
            "seconds_since_last_success": round(now - self._last_success, 1),
            "forced": self._forced_tier.value if self._forced_tier else None,
        }


# Module-level singleton
_monitor: ResourceMonitor = ResourceMonitor()


def get_resource_monitor() -> ResourceMonitor:
    """Return the global resource monitor."""
    return _monitor


def setup_resource_monitor(
    failure_threshold: int = 3,
    cooldown_seconds: float = 60.0,
) -> ResourceMonitor:
    """Create and register the global resource monitor."""
    global _monitor
    _monitor = ResourceMonitor(
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
    )
    return _monitor
