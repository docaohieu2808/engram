"""Webhook fire-and-forget support for engram memory events."""

from __future__ import annotations

import asyncio
import logging
import urllib.request
import json

logger = logging.getLogger("engram")

# M5: Allowed URL schemes
_ALLOWED_SCHEMES = ("http://", "https://")

# M5: Blocked internal host prefixes/values (simple string check)
_BLOCKED_HOSTS = (
    "127.",
    "localhost",
    "0.",
    "10.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.",
    "169.254.",  # link-local
    "::1",        # IPv6 loopback
    "[::1]",
)


def _is_safe_webhook_url(url: str) -> bool:
    """Return True if URL is safe to call (M5: SSRF protection)."""
    url_lower = url.lower()

    # Must start with http:// or https://
    if not any(url_lower.startswith(scheme) for scheme in _ALLOWED_SCHEMES):
        return False

    # Extract host portion (between scheme and first / or end)
    try:
        after_scheme = url.split("//", 1)[1]
        host_part = after_scheme.split("/")[0].split("?")[0].split("#")[0]
        # Strip port
        host = host_part.rsplit(":", 1)[0].lower()
    except (IndexError, ValueError):
        return False

    for blocked in _BLOCKED_HOSTS:
        if host == blocked.rstrip(".") or host.startswith(blocked):
            return False

    return True


def fire_hook(url: str | None, data: dict) -> None:
    """Fire a webhook POST in the background — fire-and-forget, never raises.

    Uses stdlib urllib.request to avoid new dependencies.
    Runs in a background thread to avoid blocking the event loop.
    Validates URL scheme and blocks internal IPs (M5: SSRF protection).

    Args:
        url: Webhook URL to POST to. If None or empty, no-op.
        data: Dict payload serialized to JSON.
    """
    if not url:
        return

    # M5: SSRF validation
    if not _is_safe_webhook_url(url):
        logger.warning("Hook URL blocked (SSRF protection): %s", url)
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
