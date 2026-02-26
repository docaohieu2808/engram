"""Tests for auto-consolidation trigger."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from engram.consolidation.auto_trigger import AutoConsolidationTrigger
from engram.config import ConsolidationConfig


class TestAutoConsolidationTrigger:

    def _make_trigger(self, threshold: int = 5) -> tuple[AutoConsolidationTrigger, AsyncMock]:
        engine = AsyncMock()
        engine.consolidate = AsyncMock(return_value={"consolidated": 3})
        config = ConsolidationConfig(auto_trigger_threshold=threshold)
        trigger = AutoConsolidationTrigger(engine, config)
        return trigger, engine

    @pytest.mark.asyncio
    async def test_no_trigger_below_threshold(self):
        trigger, engine = self._make_trigger(threshold=5)
        for _ in range(4):
            result = await trigger.on_message()
            assert result is False
        engine.consolidate.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_at_threshold(self):
        trigger, engine = self._make_trigger(threshold=3)
        await trigger.on_message()
        await trigger.on_message()
        result = await trigger.on_message()
        assert result is True
        # M13: create_task is fire-and-forget, yield to let the task run
        await asyncio.sleep(0)
        engine.consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_counter_resets_after_trigger(self):
        trigger, engine = self._make_trigger(threshold=2)
        await trigger.on_message()
        await trigger.on_message()  # triggers
        assert trigger.message_count == 0

    @pytest.mark.asyncio
    async def test_consolidation_failure_doesnt_crash(self):
        trigger, engine = self._make_trigger(threshold=1)
        engine.consolidate = AsyncMock(side_effect=Exception("LLM down"))
        result = await trigger.on_message()
        assert result is True  # still returns True (was triggered)

    def test_reset(self):
        trigger, _ = self._make_trigger()
        trigger._message_count = 10
        trigger.reset()
        assert trigger.message_count == 0

    def test_threshold_property(self):
        trigger, _ = self._make_trigger(threshold=20)
        assert trigger.threshold == 20

    @pytest.mark.asyncio
    async def test_default_config(self):
        engine = AsyncMock()
        trigger = AutoConsolidationTrigger(engine)
        assert trigger.threshold == 20
