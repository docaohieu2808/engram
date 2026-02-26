"""Tests for WebSocket event bus."""

from __future__ import annotations

import pytest

from engram.ws.event_bus import EventBus


@pytest.fixture
def bus():
    return EventBus()


@pytest.mark.asyncio
async def test_emit_calls_subscriber(bus):
    received = []

    async def handler(tenant_id, event, data):
        received.append((tenant_id, event, data))

    bus.subscribe(handler)
    await bus.emit("t1", "memory_created", {"id": "m1"})
    assert len(received) == 1
    assert received[0] == ("t1", "memory_created", {"id": "m1"})


@pytest.mark.asyncio
async def test_emit_calls_multiple_subscribers(bus):
    counts = [0, 0]

    async def h0(t, e, d):
        counts[0] += 1

    async def h1(t, e, d):
        counts[1] += 1

    bus.subscribe(h0)
    bus.subscribe(h1)
    await bus.emit("t1", "test", {})
    assert counts == [1, 1]


@pytest.mark.asyncio
async def test_handler_exception_does_not_break_others(bus):
    called = []

    async def bad_handler(t, e, d):
        raise ValueError("boom")

    async def good_handler(t, e, d):
        called.append(True)

    bus.subscribe(bad_handler)
    bus.subscribe(good_handler)
    await bus.emit("t1", "test", {})
    assert called == [True]


@pytest.mark.asyncio
async def test_clear_removes_all_handlers(bus):
    received = []

    async def handler(t, e, d):
        received.append(True)

    bus.subscribe(handler)
    bus.clear()
    await bus.emit("t1", "test", {})
    assert received == []
