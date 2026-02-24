"""Sliding-window rate limiter using Redis sorted sets. Gracefully degrades when Redis is unavailable."""

from __future__ import annotations

import time
from typing import Any

try:
    from redis.asyncio import Redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RateLimiter:
    """Sliding window rate limiter per tenant using Redis sorted sets.

    Returns (allowed, remaining, reset_at) tuples.
    When Redis is unavailable, all requests are allowed (graceful degradation).
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
            When Redis is down, returns (True, requests_per_minute, 0).
        """
        if not self._redis:
            return True, self.requests_per_minute, 0

        key = f"engram:ratelimit:{tenant_id}"
        now = time.time()
        window_start = now - self._window
        limit = self.requests_per_minute + self.burst

        try:
            # Phase 1: atomically remove expired entries and count current window
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = await pipe.execute()

            current_count = results[1]
            reset_at = int(now) + self._window

            if current_count >= limit:
                return False, 0, reset_at

            # Phase 2: request is allowed — record it atomically
            pipe2 = self._redis.pipeline()
            pipe2.zadd(key, {str(now): now})
            pipe2.expire(key, self._window * 2)
            await pipe2.execute()

            remaining = max(0, limit - current_count - 1)
            return True, remaining, reset_at

        except Exception:
            # Redis error → allow request (graceful degradation)
            return True, self.requests_per_minute, 0
