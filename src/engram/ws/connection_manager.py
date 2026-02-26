"""WebSocket connection manager — tracks connections per tenant, broadcasts events."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from fastapi import WebSocket

from engram.ws.protocol import WSEvent

logger = logging.getLogger("engram.ws")


class ConnectionManager:
    """Track active WebSocket connections grouped by tenant_id."""

    def __init__(self) -> None:
        # tenant_id -> set of (websocket, subject_id)
        self._connections: dict[str, set[tuple[WebSocket, str]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, tenant_id: str, sub: str) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[tenant_id].add((ws, sub))
        logger.info("ws connected: sub=%s tenant=%s", sub, tenant_id)

    async def disconnect(self, ws: WebSocket, tenant_id: str, sub: str) -> None:
        async with self._lock:
            self._connections[tenant_id].discard((ws, sub))
            if not self._connections[tenant_id]:
                del self._connections[tenant_id]
        logger.info("ws disconnected: sub=%s tenant=%s", sub, tenant_id)

    async def broadcast(self, tenant_id: str, event: WSEvent, exclude_sub: str = "") -> None:
        """Send event to all connections in tenant, optionally excluding sender."""
        async with self._lock:
            targets = list(self._connections.get(tenant_id, set()))
        payload = event.model_dump()
        for ws, sub in targets:
            if sub == exclude_sub:
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                logger.debug("ws broadcast failed for sub=%s, will clean on disconnect", sub)

    @property
    def active_connections(self) -> int:
        """Total active connections across all tenants."""
        return sum(len(conns) for conns in self._connections.values())


# Module-level singleton
manager = ConnectionManager()
