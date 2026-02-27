"""In-process event bus — store operations emit events, WS manager broadcasts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from engram.config import EventBusConfig

logger = logging.getLogger("engram.ws")

# Async handler signature: (tenant_id, event_name, data) -> None
EventHandler = Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    """Simple pub/sub: emit(tenant, event, data) calls all subscribers."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a specific handler (e.g. on WebSocket disconnect)."""
        self._handlers = [h for h in self._handlers if h is not handler]

    def clear(self) -> None:
        """Remove all handlers (useful for testing)."""
        self._handlers.clear()

    async def emit(self, tenant_id: str, event: str, data: dict[str, Any]) -> None:
        for handler in self._handlers:
            try:
                await handler(tenant_id, event, data)
            except Exception as exc:
                logger.warning("event-bus handler error: %s", exc)


def make_event_bus(config: "EventBusConfig | None" = None) -> EventBus:
    """Factory: returns RedisEventBus if config requests it, else in-process EventBus."""
    if config and config.enabled and config.backend == "redis":
        from engram.ws.redis_event_bus import RedisEventBus
        return RedisEventBus(config.redis_url)
    return EventBus()


# Module-level singleton
event_bus = EventBus()
