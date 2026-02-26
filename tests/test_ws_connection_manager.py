"""Tests for WebSocket connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.ws.connection_manager import ConnectionManager
from engram.ws.protocol import WSEvent


def _mock_ws(accept_ok: bool = True) -> MagicMock:
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    if not accept_ok:
        ws.accept.side_effect = RuntimeError("accept failed")
    return ws


@pytest.fixture
def mgr():
    return ConnectionManager()


@pytest.mark.asyncio
async def test_connect_and_disconnect(mgr):
    ws = _mock_ws()
    await mgr.connect(ws, "tenant-a", "agent-1")
    assert mgr.active_connections == 1

    await mgr.disconnect(ws, "tenant-a", "agent-1")
    assert mgr.active_connections == 0


@pytest.mark.asyncio
async def test_broadcast_reaches_same_tenant(mgr):
    ws1, ws2 = _mock_ws(), _mock_ws()
    await mgr.connect(ws1, "t1", "a1")
    await mgr.connect(ws2, "t1", "a2")

    event = WSEvent(event="memory_created", tenant_id="t1", data={"id": "m1"})
    await mgr.broadcast("t1", event)

    ws1.send_json.assert_called_once()
    ws2.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_does_not_cross_tenants(mgr):
    ws1, ws2 = _mock_ws(), _mock_ws()
    await mgr.connect(ws1, "t1", "a1")
    await mgr.connect(ws2, "t2", "a2")

    event = WSEvent(event="memory_created", tenant_id="t1", data={})
    await mgr.broadcast("t1", event)

    ws1.send_json.assert_called_once()
    ws2.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_exclude_sub(mgr):
    ws1, ws2 = _mock_ws(), _mock_ws()
    await mgr.connect(ws1, "t1", "a1")
    await mgr.connect(ws2, "t1", "a2")

    event = WSEvent(event="memory_created", tenant_id="t1", data={})
    await mgr.broadcast("t1", event, exclude_sub="a1")

    ws1.send_json.assert_not_called()
    ws2.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_failed_send_does_not_crash(mgr):
    ws1 = _mock_ws()
    ws2 = _mock_ws()
    ws1.send_json.side_effect = RuntimeError("connection closed")
    await mgr.connect(ws1, "t1", "a1")
    await mgr.connect(ws2, "t1", "a2")

    event = WSEvent(event="test", tenant_id="t1", data={})
    # Should not raise even though ws1.send_json fails
    await mgr.broadcast("t1", event)
    ws2.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_nonexistent_is_safe(mgr):
    ws = _mock_ws()
    # Should not raise
    await mgr.disconnect(ws, "nonexistent", "nobody")
    assert mgr.active_connections == 0
