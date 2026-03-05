"""HTTP proxy for ingesting messages via oracle's /api/v1/remember endpoint.

Used by watcher nodes (e.g. server-1) that capture chats and forward
to the oracle engram server instead of writing directly to Qdrant.
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
        _client = httpx.AsyncClient(timeout=30)
    return _client


async def http_ingest_messages(
    messages: list[dict[str, Any]],
    base_url: str,
    source: str = "",
) -> IngestResult:
    """Send each message to oracle's /api/v1/remember endpoint."""
    client = _get_client()
    remember_url = f"{base_url.rstrip('/')}/api/v1/remember"
    count = 0

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        try:
            resp = await client.post(
                remember_url,
                json={"content": content, "source": source},
            )
            resp.raise_for_status()
            count += 1
        except Exception as e:
            print(f"HTTP ingest failed: {e}", flush=True)
            logger.warning("HTTP ingest to %s failed: %s", remember_url, e)

    return IngestResult(episodic_count=count)
