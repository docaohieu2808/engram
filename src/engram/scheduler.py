"""Background scheduler for periodic memory maintenance tasks.

Uses recursive asyncio.sleep (not setInterval) for overlap protection —
next tick only scheduled after current tick completes.

Tasks:
- consolidate_memories: Cluster + LLM-summarize related memories (nightly)
- cleanup_expired: Remove expired memories (daily)
- decay_report: Log decay statistics (daily)

Each task checks resource tier before running; skips expensive operations
on BASIC tier. State persisted to ~/.engram/scheduler_state.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from engram.config import SchedulerConfig

logger = logging.getLogger("engram")


@dataclass
class ScheduledTask:
    """A scheduled background task."""
    name: str
    interval_seconds: float
    func: Callable[[], Coroutine[Any, Any, dict[str, Any]]]
    requires_llm: bool = False  # Skip if resource tier < STANDARD
    last_run: float = 0.0
    run_count: int = 0
    last_error: str | None = None


class MemoryScheduler:
    """Asyncio-based scheduler for periodic memory maintenance.

    Overlap-safe: uses recursive setTimeout pattern (next tick after current completes).
    """

    def __init__(
        self,
        state_path: str = "~/.engram/scheduler_state.json",
        tick_interval: float = 60.0,
        task_timeout: float = 300.0,
    ):
        self._state_path = Path(state_path).expanduser()
        self._tick_interval = tick_interval
        self._task_timeout = task_timeout
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._task_handle: asyncio.Task | None = None

    def register(
        self,
        name: str,
        func: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        interval_seconds: float,
        requires_llm: bool = False,
    ) -> None:
        """Register a periodic task."""
        self._tasks[name] = ScheduledTask(
            name=name,
            interval_seconds=interval_seconds,
            func=func,
            requires_llm=requires_llm,
        )

    def start(self) -> None:
        """Start the scheduler loop (non-blocking)."""
        if self._running:
            logger.debug("Scheduler already running")
            return
        self._running = True
        self._load_state()
        self._task_handle = asyncio.ensure_future(self._loop())
        logger.info("Memory scheduler started (%d tasks)", len(self._tasks))

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False
        if self._task_handle:
            self._task_handle.cancel()
            try:
                await self._task_handle
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Memory scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop — recursive setTimeout pattern."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e)
            # Next tick only after current completes (overlap protection)
            await asyncio.sleep(self._tick_interval)

    async def _tick(self) -> None:
        """Execute all due tasks."""
        from engram.resource_tier import get_resource_monitor

        now = time.monotonic()
        monitor = get_resource_monitor()

        for task in self._tasks.values():
            elapsed = now - task.last_run
            if elapsed < task.interval_seconds:
                continue

            # Skip LLM-requiring tasks on BASIC tier
            if task.requires_llm and not monitor.can_use_llm():
                logger.debug("Skipping %s (resource tier too low for LLM)", task.name)
                continue

            # Execute task with timeout
            try:
                result = await asyncio.wait_for(task.func(), timeout=self._task_timeout)
                task.last_run = now
                task.run_count += 1
                task.last_error = None
                logger.info("Scheduler task %s completed: %s", task.name, result)
            except asyncio.TimeoutError:
                task.last_error = f"timeout ({self._task_timeout}s)"
                logger.warning("Scheduler task %s timed out", task.name)
            except Exception as e:
                task.last_error = str(e)[:200]
                logger.warning("Scheduler task %s failed: %s", task.name, e)

        self._save_state()

    def status(self) -> list[dict[str, Any]]:
        """Return status of all scheduled tasks."""
        now = time.monotonic()
        result = []
        for task in self._tasks.values():
            elapsed = now - task.last_run if task.last_run > 0 else None
            next_in = max(0, task.interval_seconds - elapsed) if elapsed is not None else 0
            result.append({
                "name": task.name,
                "interval_seconds": task.interval_seconds,
                "run_count": task.run_count,
                "last_error": task.last_error,
                "next_run_in": round(next_in),
                "requires_llm": task.requires_llm,
            })
        return result

    def _load_state(self) -> None:
        """Load task run times from persistent state file."""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            for name, state in data.items():
                if name in self._tasks:
                    self._tasks[name].run_count = state.get("run_count", 0)
                    # Convert stored timestamp to monotonic offset
                    last_iso = state.get("last_run_iso")
                    if last_iso:
                        last_dt = datetime.fromisoformat(last_iso)
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        age = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        self._tasks[name].last_run = time.monotonic() - age
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def _save_state(self) -> None:
        """Persist task run state to file."""
        data = {}
        now_mono = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        for task in self._tasks.values():
            if task.last_run > 0:
                age = now_mono - task.last_run
                last_utc = now_utc.timestamp() - age
                last_iso = datetime.fromtimestamp(last_utc, tz=timezone.utc).isoformat()
            else:
                last_iso = None
            data[task.name] = {
                "run_count": task.run_count,
                "last_run_iso": last_iso,
                "last_error": task.last_error,
            }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


def create_default_scheduler(
    episodic_store: Any,
    consolidation_engine: Any | None = None,
    config: SchedulerConfig | None = None,
) -> MemoryScheduler:
    """Create scheduler with default maintenance tasks.

    Args:
        episodic_store: EpisodicStore instance for cleanup/stats
        consolidation_engine: ConsolidationEngine instance (optional, for consolidation task)
        config: SchedulerConfig with intervals and limits
    """
    cfg = config or SchedulerConfig()
    scheduler = MemoryScheduler(
        tick_interval=cfg.tick_interval_seconds,
        task_timeout=cfg.task_timeout_seconds,
    )

    # Task: drain pending queue (retry failed embeddings — legacy JSONL queue)
    async def drain_pending_queue() -> dict[str, Any]:
        from engram.episodic.pending_queue import get_pending_queue
        queue = get_pending_queue()
        if queue.count() == 0:
            return {"skipped": True}
        success, failed = await queue.drain(episodic_store)
        logger.info("Drained %d from pending queue (%d still pending)", success, failed)
        return {"success": success, "failed": failed}

    scheduler.register(
        "drain_pending_queue", drain_pending_queue,
        interval_seconds=cfg.queue_drain_interval_seconds,
        requires_llm=False,
    )

    # Task: process embedding queue (SQLite-backed retry, runs every 5 min)
    async def process_embedding_queue() -> dict[str, Any]:
        from engram.resource_tier import get_resource_monitor, ResourceTier
        monitor = get_resource_monitor()
        if monitor.get_tier() == ResourceTier.READONLY:
            return {"skipped": True, "reason": "readonly tier"}
        from engram.episodic.embedding_queue import process_embedding_queue as _process, get_embedding_queue
        queue = get_embedding_queue()
        if queue.pending_count() == 0:
            return {"skipped": True}
        return await _process(episodic_store, queue)

    scheduler.register(
        "process_embedding_queue", process_embedding_queue,
        interval_seconds=300,  # 5 minutes
        requires_llm=False,
    )

    # Task: cleanup expired memories (no LLM needed)
    async def cleanup_expired() -> dict[str, Any]:
        count = await episodic_store.cleanup_expired()
        return {"deleted": count}

    scheduler.register(
        "cleanup_expired", cleanup_expired,
        interval_seconds=cfg.cleanup_interval_seconds,
        requires_llm=False,
    )

    # Task: consolidate related memories (requires LLM)
    if consolidation_engine is not None:
        async def consolidate_memories() -> dict[str, Any]:
            new_ids = await consolidation_engine.consolidate(limit=50)
            return {"consolidated": len(new_ids)}

        scheduler.register(
            "consolidate_memories", consolidate_memories,
            interval_seconds=cfg.consolidate_interval_seconds,
            requires_llm=True,
        )

    # Task: log decay statistics (no LLM needed)
    async def decay_report() -> dict[str, Any]:
        import math
        memories = await episodic_store.get_recent(n=50)
        now = datetime.now(timezone.utc)
        low_retention = 0
        for mem in memories:
            ts = mem.timestamp if mem.timestamp.tzinfo else mem.timestamp.replace(tzinfo=timezone.utc)
            days = (now - ts).total_seconds() / 86400
            retention = math.exp(-mem.decay_rate * days / (1 + cfg.decay_access_multiplier * mem.access_count))
            if retention < 0.3:
                low_retention += 1
        return {"total_checked": len(memories), "low_retention": low_retention}

    scheduler.register(
        "decay_report", decay_report,
        interval_seconds=cfg.decay_report_interval_seconds,
        requires_llm=False,
    )

    return scheduler
