"""Webhook fire-and-forget support for engram memory events."""

from __future__ import annotations

import asyncio
import logging
import urllib.request
import json

logger = logging.getLogger("engram")


def fire_hook(url: str | None, data: dict) -> None:
    """Fire a webhook POST in the background — fire-and-forget, never raises.

    Uses stdlib urllib.request to avoid new dependencies.
    Runs in a background thread to avoid blocking the event loop.

    Args:
        url: Webhook URL to POST to. If None or empty, no-op.
        data: Dict payload serialized to JSON.
    """
    if not url:
        return

    def _post() -> None:
        try:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug("Hook %s responded %s", url, resp.status)
        except Exception as exc:
            logger.debug("Hook %s failed (ignored): %s", url, exc)

    # Schedule in background thread — don't block caller
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _post)
    except RuntimeError:
        # No running event loop — run inline
        _post()
