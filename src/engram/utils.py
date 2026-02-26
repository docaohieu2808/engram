"""Shared utility functions for engram."""

from __future__ import annotations

import asyncio
import concurrent.futures
import unicodedata
from typing import Any


def strip_diacritics(text: str) -> str:
    """Remove diacritics for fuzzy matching (e.g. 'Trâm' -> 'Tram')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def run_async(coro: Any) -> Any:
    """Run async coroutine from sync context.

    A-C2: Detects whether an event loop is already running (e.g. inside
    Jupyter, FastAPI, or any other async framework) and falls back to a
    ThreadPoolExecutor to avoid the 'asyncio.run() cannot be called from
    a running event loop' RuntimeError.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)

    # A running loop exists; submit to a new thread with its own event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
