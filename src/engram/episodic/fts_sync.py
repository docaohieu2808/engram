"""Async helper functions for FTS5 index synchronization.

Wraps FtsIndex calls in asyncio.to_thread to avoid blocking the event loop.
These are standalone async functions (not a mixin) used by crud and maintenance mixins.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.episodic.fts_index import FtsIndex

logger = logging.getLogger("engram")


async def _do_fts_insert(fts: "FtsIndex", memory_id: str, content: str, memory_type: str) -> None:
    """Inner coroutine that performs the actual FTS insert (used by fire-and-forget task)."""
    try:
        await asyncio.to_thread(fts.insert, memory_id, content, memory_type)
    except Exception as e:
        logger.warning("FTS5 insert failed for %s: %s", memory_id, e)


async def fts_insert(fts: "FtsIndex | None", memory_id: str, content: str, memory_type: str) -> None:
    """Schedule FTS5 insert as a background task (fire-and-forget). Returns immediately."""
    if fts is None:
        return
    asyncio.create_task(_do_fts_insert(fts, memory_id, content, memory_type))


async def _do_fts_insert_batch(fts: "FtsIndex", entries: list[tuple[str, str, str]]) -> None:
    """Inner coroutine for batch FTS insert (used by fire-and-forget task)."""
    try:
        await asyncio.to_thread(fts.insert_batch, entries)
    except Exception as e:
        logger.warning("FTS5 batch insert failed: %s", e)


async def fts_insert_batch(fts: "FtsIndex | None", entries: list[tuple[str, str, str]]) -> None:
    """Schedule FTS5 batch insert as a background task (fire-and-forget). Returns immediately."""
    if fts is None:
        return
    asyncio.create_task(_do_fts_insert_batch(fts, entries))


async def fts_delete(fts: "FtsIndex | None", memory_id: str) -> None:
    """Delete an entry from the FTS5 index (non-blocking)."""
    if fts is None:
        return
    try:
        await asyncio.to_thread(fts.delete, memory_id)
    except Exception as e:
        logger.debug("FTS5 delete failed for %s: %s", memory_id, e)


async def fts_search(fts: "FtsIndex | None", query: str, limit: int) -> list:
    """Search the FTS5 index (non-blocking). Returns list of FtsResult objects."""
    if fts is None:
        return []
    return await asyncio.to_thread(fts.search, query, limit)


async def fts_get_all_ids(fts: "FtsIndex | None") -> set[str]:
    """Retrieve all IDs from the FTS5 index (non-blocking)."""
    if fts is None:
        return set()
    return set(await asyncio.to_thread(fts.get_all_ids))
