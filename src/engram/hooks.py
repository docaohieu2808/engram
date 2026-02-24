"""Webhook fire-and-forget support for engram memory events."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import urllib.parse
import urllib.request
import json

logger = logging.getLogger("engram")

# C2 fix: allowed URL schemes (whitelist, not blocklist)
_ALLOWED_SCHEMES = {"http", "https"}


def _is_private_ip(ip_str: str) -> bool:
    """Return True if the IP address is in a private/reserved range.

    C2 fix: uses ipaddress module for robust IP range checking, handles
    IPv4, IPv6, and IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1).
    """
    try:
        addr = ipaddress.ip_address(ip_str)
        # Unwrap IPv6-mapped IPv4 (e.g. ::ffff:127.0.0.1 → 127.0.0.1)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        )
    except ValueError:
        # Unparseable IP string — treat as unsafe
        return True


def _is_safe_webhook_url(url: str) -> bool:
    """Return True if URL is safe to call (C2/SSRF protection).

    Validates:
    1. Scheme is http or https only.
    2. No credentials (user@) in the authority section.
    3. DNS resolves to a non-private IP (blocks 127.x, 10.x, 172.16-31.x, 192.168.x,
       IPv6 loopback, link-local, and IPv6-mapped private addresses).

    Note: DNS rebinding risk is mitigated here by resolving at validation time.
    For higher security, consider re-resolving at request time.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False

    # Must be http or https
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False

    # Block URLs with credentials (user:pass@ or user@) in the authority
    if parsed.username is not None or parsed.password is not None:
        logger.warning("Hook URL blocked (credentials in URL): %s", url)
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Resolve hostname to IP and validate against private ranges
    try:
        # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
        results = socket.getaddrinfo(hostname, None)
        if not results:
            return False
        for result in results:
            sockaddr = result[4]
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                logger.warning(
                    "Hook URL blocked (resolves to private IP %s): %s", ip_str, url
                )
                return False
    except (socket.gaierror, OSError) as exc:
        # DNS resolution failure — block the URL (fail closed)
        logger.warning("Hook URL blocked (DNS resolution failed: %s): %s", exc, url)
        return False

    return True


def fire_hook(url: str | None, data: dict) -> None:
    """Fire a webhook POST in the background — fire-and-forget, never raises.

    Uses stdlib urllib.request to avoid new dependencies.
    Runs in a background thread to avoid blocking the event loop.
    Validates URL scheme and blocks internal IPs (C2: SSRF protection).

    Args:
        url: Webhook URL to POST to. If None or empty, no-op.
        data: Dict payload serialized to JSON.
    """
    if not url:
        return

    # C2: SSRF validation — resolves DNS and checks IP ranges
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
