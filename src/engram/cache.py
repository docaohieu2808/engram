"""Redis-backed cache for engram memory operations. Gracefully degrades when Redis is unavailable."""

from __future__ import annotations

import hashlib
import json
from typing import Any

try:
    from redis.asyncio import Redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class EngramCache:
    """Tenant-scoped Redis cache with graceful degradation."""

    def __init__(self, redis_url: str) -> None:
        self._redis: Any = None
        self._url = redis_url

    async def connect(self) -> None:
        """Connect to Redis. Silently skips if Redis unavailable or unreachable."""
        if not _REDIS_AVAILABLE:
            return
        try:
            self._redis = Redis.from_url(self._url, decode_responses=False)
            await self._redis.ping()
        except Exception:
            self._redis = None  # graceful degradation

    async def get(self, tenant_id: str, operation: str, params: dict) -> dict | None:
        """Return cached result or None if not found / Redis down."""
        if not self._redis:
            return None
        key = self._cache_key(tenant_id, operation, params)
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    async def set(
        self,
        tenant_id: str,
        operation: str,
        params: dict,
        result: Any,
        ttl: int = 300,
    ) -> None:
        """Store result in cache with TTL (seconds). Silently skips on failure."""
        if not self._redis:
            return
        key = self._cache_key(tenant_id, operation, params)
        try:
            await self._redis.setex(key, ttl, json.dumps(result, default=str))
        except Exception:
            pass

    async def invalidate(self, tenant_id: str, operation: str = "*") -> None:
        """Delete all cache entries for a tenant+operation pattern."""
        if not self._redis:
            return
        pattern = f"engram:{tenant_id}:{operation}:*"
        try:
            async for key in self._redis.scan_iter(pattern):
                await self._redis.delete(key)
        except Exception:
            pass

    def _cache_key(self, tenant_id: str, operation: str, params: dict) -> str:
        """Deterministic cache key: engram:{tenant}:{op}:{hash}."""
        h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        return f"engram:{tenant_id}:{operation}:{h}"
