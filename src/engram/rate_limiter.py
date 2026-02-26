"""Sliding-window rate limiter using Redis sorted sets. Gracefully degrades when Redis is unavailable."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("engram")

try:
    from redis.asyncio import Redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RateLimiter:
    """Sliding window rate limiter per tenant using Redis sorted sets.

    Returns (allowed, remaining, reset_at) tuples.
    S-H4: When Redis is unavailable, requests are DENIED (fail closed) to prevent
    rate limit bypass during Redis outages.
    """

    def __init__(self, redis_url: str, requests_per_minute: int = 60, burst: int = 10) -> None:
        self._redis: Any = None
        self._url = redis_url
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        # Window size in seconds
        self._window = 60

    async def connect(self) -> None:
        """Connect to Redis. Silently skips if Redis unavailable or unreachable."""
        if not _REDIS_AVAILABLE:
            return
        try:
            self._redis = Redis.from_url(self._url, decode_responses=False)
            await self._redis.ping()
        except Exception:
            self._redis = None  # graceful degradation

    async def check(self, tenant_id: str) -> tuple[bool, int, int]:
        """Check rate limit for tenant.

        Returns:
            (allowed, remaining, reset_at) where reset_at is Unix timestamp.
            S-H4: When Redis is down, returns (False, 0, 0) — fail closed to prevent bypass.
        """
        if not self._redis:
            # S-H4: fail closed — deny when rate limiter is unavailable
            logger.warning("rate_limiter: Redis unavailable, denying request for tenant %s", tenant_id)
            return False, 0, 0

        key = f"engram:ratelimit:{tenant_id}"
        now = time.time()
        window_start = now - self._window
        limit = self.requests_per_minute + self.burst

        try:
            # Atomic check-and-record via single pipeline
            pipe = self._redis.pipeline(transaction=True)
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self._window * 2)
            results = await pipe.execute()

            current_count = results[1]  # count BEFORE the zadd
            reset_at = int(now) + self._window

            if current_count >= limit:
                # Over limit — remove the entry we just added
                await self._redis.zrem(key, str(now))
                return False, 0, reset_at

            remaining = max(0, limit - current_count - 1)
            return True, remaining, reset_at

        except Exception:
            # S-H4: Redis pipeline error → fail closed (deny) to prevent rate limit bypass
            logger.warning("rate_limiter: Redis pipeline error, denying request for tenant %s", tenant_id)
            return False, 0, 0
