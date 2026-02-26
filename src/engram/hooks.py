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


def _validate_and_resolve_webhook_url(url: str) -> tuple[bool, str | None]:
    """Validate URL safety and return (is_safe, resolved_ip).

    S-H3: DNS TOCTOU fix — resolves DNS once here, returns the pinned IP so the
    caller can use it directly in the HTTP request, eliminating the window between
    validation and connection where DNS rebinding could redirect to a private IP.

    Returns:
        (True, resolved_ip_str) if safe, or (False, None) if blocked.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, None

    # Must be http or https
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, None

    # Block URLs with credentials (user:pass@ or user@) in the authority
    if parsed.username is not None or parsed.password is not None:
        logger.warning("Hook URL blocked (credentials in URL): %s", url)
        return False, None

    hostname = parsed.hostname
    if not hostname:
        return False, None

    # Resolve hostname to IP and validate against private ranges
    try:
        # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
        results = socket.getaddrinfo(hostname, None)
        if not results:
            return False, None
        for result in results:
            sockaddr = result[4]
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                logger.warning(
                    "Hook URL blocked (resolves to private IP %s): %s", ip_str, url
                )
                return False, None
        # Return first resolved IP to pin for the actual request (S-H3 TOCTOU fix)
        resolved_ip = results[0][4][0]
    except (socket.gaierror, OSError) as exc:
        # DNS resolution failure — block the URL (fail closed)
        logger.warning("Hook URL blocked (DNS resolution failed: %s): %s", exc, url)
        return False, None

    return True, resolved_ip


def fire_hook(url: str | None, data: dict) -> None:
    """Fire a webhook POST in the background — fire-and-forget, never raises.

    Uses stdlib urllib.request to avoid new dependencies.
    Runs in a background thread to avoid blocking the event loop.
    Validates URL scheme and blocks internal IPs (C2: SSRF protection).
    S-H3: DNS resolved once at validation time; pinned IP used in the request.

    Args:
        url: Webhook URL to POST to. If None or empty, no-op.
        data: Dict payload serialized to JSON.
    """
    if not url:
        return

    # C2 + S-H3: SSRF validation — resolves DNS once and pins the IP
    safe, resolved_ip = _validate_and_resolve_webhook_url(url)
    if not safe or resolved_ip is None:
        logger.warning("Hook URL blocked (SSRF protection): %s", url)
        return

    # Build a URL with the resolved IP to prevent DNS rebinding (S-H3)
    parsed = urllib.parse.urlparse(url)
    port = parsed.port
    if port:
        netloc_pinned = f"{resolved_ip}:{port}"
    else:
        netloc_pinned = resolved_ip
    pinned_url = parsed._replace(netloc=netloc_pinned).geturl()
    original_host = parsed.hostname or url

    def _post() -> None:
        try:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                pinned_url,
                data=payload,
                # Pass original Host header so the server can route correctly
                headers={"Content-Type": "application/json", "Host": original_host},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: B310 — SSRF guard applied above via validate_webhook_url()
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
