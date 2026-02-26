"""Auto-consolidation trigger — run consolidation after N messages.

Counts ingested messages and triggers background consolidation
when threshold is reached. Non-blocking, fire-and-forget.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from engram.config import ConsolidationConfig

if TYPE_CHECKING:
    from engram.consolidation.engine import ConsolidationEngine

logger = logging.getLogger("engram")


class AutoConsolidationTrigger:
    """Trigger memory consolidation after a configurable message count."""

    def __init__(self, engine: "ConsolidationEngine", config: ConsolidationConfig | None = None):
        self._engine = engine
        self._config = config or ConsolidationConfig()
        self._message_count: int = 0
        self._running: bool = False

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def threshold(self) -> int:
        return self._config.auto_trigger_threshold

    async def on_message(self) -> bool:
        """Call after each message ingested. Returns True if consolidation triggered."""
        self._message_count += 1
        if self._message_count >= self._config.auto_trigger_threshold:
            self._message_count = 0
            # Fire-and-forget: create_task returns immediately, doesn't block caller (M13)
            asyncio.create_task(self._run_background())
            return True
        return False

    async def _run_background(self) -> None:
        """Run consolidation in background without blocking."""
        if self._running:
            logger.debug("Consolidation already running, skipping")
            return
        self._running = True
        try:
            result = await self._engine.consolidate()
            logger.info("Auto-consolidation complete: %s", result)
        except Exception as e:
            logger.warning("Auto-consolidation failed: %s", e)
        finally:
            self._running = False

    def reset(self) -> None:
        """Reset message counter."""
        self._message_count = 0
