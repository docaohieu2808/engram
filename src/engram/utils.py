"""Shared utility functions for engram."""

from __future__ import annotations

import asyncio
import unicodedata
from typing import Any


def strip_diacritics(text: str) -> str:
    """Remove diacritics for fuzzy matching (e.g. 'Trâm' -> 'Tram')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def run_async(coro: Any) -> Any:
    """Run async coroutine from sync context."""
    return asyncio.run(coro)
