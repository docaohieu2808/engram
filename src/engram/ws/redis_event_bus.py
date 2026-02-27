"""Redis pub/sub adapter for cross-process event delivery.

Uses redis.asyncio (bundled with redis>=4.2.0). Install with:
    pip install redis[hiredis]

Channel naming: engram:events:{tenant_id}
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from engram.ws.event_bus import EventBus, EventHandler

logger = logging.getLogger("engram.ws")


def _require_redis():
    try:
        import redis.asyncio as aioredis
        return aioredis
    except ImportError as e:
        raise ImportError(
            "redis package is required for RedisEventBus. "
            "Install with: pip install 'redis[hiredis]'"
        ) from e


class RedisEventBus(EventBus):
    """EventBus backed by Redis pub/sub for cross-process delivery.

    Local in-process handlers are also called (same as parent EventBus.emit).
    Redis messages are published to all subscribers across processes.
    """

    # Redis channel prefix — per-tenant: engram:events:{tenant_id}
    _CHANNEL_PREFIX = "engram:events:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        super().__init__()
        self._redis_url = redis_url
        self._origin_id = uuid.uuid4().hex[:12]  # unique per-process to skip self-published
        self._reader_task: asyncio.Task | None = None
        self._publish_client: Any = None   # redis.asyncio.Redis
        self._subscribe_client: Any = None  # redis.asyncio.Redis (pubsub conn)
        self._pubsub: Any = None

    async def start(self) -> None:
        """Connect to Redis and start background reader task."""
        aioredis = _require_redis()
        self._publish_client = aioredis.from_url(
            self._redis_url, decode_responses=True, auto_close_connection_pool=False,
        )
        self._subscribe_client = aioredis.from_url(
            self._redis_url, decode_responses=True, auto_close_connection_pool=False,
        )
        self._pubsub = self._subscribe_client.pubsub()
        # Subscribe to all tenant channels via pattern
        await self._pubsub.psubscribe(f"{self._CHANNEL_PREFIX}*")
        self._reader_task = asyncio.create_task(self._reader_loop(), name="redis-event-bus-reader")
        logger.info("RedisEventBus started: %s", self._redis_url)

    async def close(self) -> None:
        """Stop reader and close Redis connections."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.punsubscribe(f"{self._CHANNEL_PREFIX}*")
            await self._pubsub.close()
        if self._subscribe_client:
            await self._subscribe_client.aclose()
        if self._publish_client:
            await self._publish_client.aclose()
        logger.info("RedisEventBus closed")

    async def emit(self, tenant_id: str, event: str, data: dict[str, Any]) -> None:
        """Emit to local handlers AND publish to Redis channel."""
        # Local delivery first (same process handlers)
        await super().emit(tenant_id, event, data)

        if self._publish_client is None:
            return  # Redis not started yet; skip remote delivery
        try:
            channel = f"{self._CHANNEL_PREFIX}{tenant_id}"
            payload = json.dumps({"origin": self._origin_id, "tenant_id": tenant_id, "event": event, "data": data})
            await self._publish_client.publish(channel, payload)
        except Exception as exc:
            logger.warning("RedisEventBus publish error: %s", exc)

    async def _reader_loop(self) -> None:
        """Background task: receive messages from Redis and call local handlers."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] != "pmessage":
                    continue
                try:
                    payload = json.loads(message["data"])
                    # Skip self-published messages (local handlers already called in emit())
                    if payload.get("origin") == self._origin_id:
                        continue
                    tenant_id = payload["tenant_id"]
                    event = payload["event"]
                    data = payload["data"]
                    # Call local handlers for messages from other processes
                    for handler in list(self._handlers):
                        try:
                            await handler(tenant_id, event, data)
                        except Exception as exc:
                            logger.warning("redis-event-bus handler error: %s", exc)
                except Exception as exc:
                    logger.warning("RedisEventBus message parse error: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("RedisEventBus reader loop died: %s", exc)
