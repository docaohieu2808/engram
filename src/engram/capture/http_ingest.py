"""HTTP proxy for ingesting messages via oracle's /api/v1/ingest endpoint.

Uses the entity-gated ingest pipeline on oracle: extract entities first,
only store messages that have at least one entity. This prevents noise
from being stored in episodic memory.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from engram.models import IngestResult

logger = logging.getLogger("engram")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60)
    return _client


async def http_ingest_messages(
    messages: list[dict[str, Any]],
    base_url: str,
    source: str = "",
) -> IngestResult:
    """Send messages to oracle's /api/v1/ingest endpoint (entity-gated)."""
    client = _get_client()
    ingest_url = f"{base_url.rstrip('/')}/api/v1/ingest"

    # Filter out empty content
    valid = [m for m in messages if m.get("content", "").strip()]
    if not valid:
        return IngestResult()

    try:
        resp = await client.post(
            ingest_url,
            json={"messages": valid, "source": source},
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        return IngestResult(
            episodic_count=result.get("episodic_count", 0),
            semantic_nodes=result.get("semantic_nodes", 0),
            semantic_edges=result.get("semantic_edges", 0),
        )
    except Exception as e:
        print(f"HTTP ingest failed: {e}", flush=True)
        logger.warning("HTTP ingest to %s failed: %s", ingest_url, e)
        return IngestResult()
