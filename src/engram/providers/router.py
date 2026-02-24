"""Smart query router — classifies queries and fans out to relevant providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engram.providers.base import MemoryProvider, ProviderResult

logger = logging.getLogger("engram.providers.router")

# Queries matching these keywords skip external providers (fast path)
INTERNAL_KEYWORDS = {
    "decision", "error", "preference", "todo", "context", "workflow",
    "quyết định", "lỗi", "ưu tiên", "ghi nhớ",
    "what did i", "what decision", "my preference",
}

# Queries matching these keywords always include external providers
DOMAIN_KEYWORDS = {
    "how to", "what is", "guide", "tutorial", "docs", "best practice",
    "hướng dẫn", "cách", "là gì", "tài liệu",
    "example", "implement", "setup", "configure", "deploy",
}


def classify_query(query: str) -> str:
    """Classify query as 'internal' (agent memory only) or 'domain' (include providers).

    Returns 'internal' or 'domain'.
    """
    q_lower = query.lower()

    # Check internal keywords
    if any(kw in q_lower for kw in INTERNAL_KEYWORDS):
        return "internal"

    # Check domain keywords
    if any(kw in q_lower for kw in DOMAIN_KEYWORDS):
        return "domain"

    # Default: include external providers for any non-internal query
    return "domain"


async def federated_search(
    query: str,
    providers: list[MemoryProvider],
    limit: int = 5,
    timeout_seconds: float = 3.0,
    force_federation: bool = False,
) -> list[ProviderResult]:
    """Search across all active providers with timeout and merge.

    Args:
        query: Search query text.
        providers: List of registered providers.
        limit: Max results per provider.
        timeout_seconds: Max wait time for slow providers.
        force_federation: If True, skip classification and search all providers.

    Returns:
        Merged, deduplicated, ranked results from all providers.
    """
    if not providers:
        return []

    # Classify query unless forced
    if not force_federation:
        classification = classify_query(query)
        if classification == "internal":
            logger.debug("Query classified as internal, skipping providers")
            return []

    # Fan-out to all active providers with timeout
    active = [p for p in providers if p.is_active]
    if not active:
        return []

    tasks = [
        asyncio.wait_for(p.tracked_search(query, limit), timeout=timeout_seconds)
        for p in active
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    all_results: list[ProviderResult] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.warning("Provider '%s' failed: %s", active[i].name, result)
            continue
        if isinstance(result, list):
            all_results.extend(result)

    # Deduplicate by content similarity (exact match)
    seen_content: set[str] = set()
    deduped: list[ProviderResult] = []
    for r in all_results:
        content_key = r.content[:200].strip().lower()
        if content_key not in seen_content:
            seen_content.add(content_key)
            deduped.append(r)

    # Sort by score descending
    deduped.sort(key=lambda r: r.score, reverse=True)

    return deduped[:limit * 2]  # Return more than limit so caller can merge with internal
