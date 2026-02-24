"""Auto-discovery for external memory services (local + remote)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import ipaddress
import socket

import aiohttp

from engram.config import DiscoveryConfig, ProviderEntry

logger = logging.getLogger("engram.providers.discovery")

# Known services with ports, paths, and API fingerprints
KNOWN_SERVICES: list[dict[str, Any]] = [
    {
        "name": "cognee",
        "ports": [8000],
        "paths": ["~/.cognee/"],
        "health": "/health",
        "fingerprint": "/api/v1/search",
        "default_config": {
            "type": "rest",
            "search_endpoint": "/api/v1/search",
            "search_method": "POST",
            "search_body": '{"query": "{query}"}',
            "result_path": "data[].text",
        },
    },
    {
        "name": "mem0",
        "ports": [8080],
        "paths": ["~/.mem0/"],
        "health": "/health",
        "fingerprint": "/v1/memories",
        "default_config": {
            "type": "rest",
            "search_endpoint": "/v1/memories/search",
            "search_method": "POST",
            "search_body": '{"query": "{query}", "limit": {limit}}',
            "result_path": "results[].memory",
        },
    },
    {
        "name": "lightrag",
        "ports": [9520],
        "paths": [],
        "health": "/health",
        "fingerprint": "/query",
        "default_config": {
            "type": "rest",
            "search_endpoint": "/query",
            "search_method": "POST",
            "search_body": '{"query": "{query}", "mode": "hybrid"}',
            "result_path": "response",
        },
    },
    {
        "name": "openclaw",
        "ports": [],
        "paths": ["~/.openclaw/workspace/memory/"],
        "health": None,
        "fingerprint": None,
        "default_config": {
            "type": "file",
            "pattern": "*.md",
        },
    },
    {
        "name": "graphiti",
        "ports": [8000],
        "paths": [],
        "health": "/healthcheck",
        "fingerprint": "/search",
        "default_config": {
            "type": "rest",
            "search_endpoint": "/search",
            "search_method": "POST",
            "search_body": '{"query": "{query}"}',
            "result_path": "results[].content",
        },
    },
]


def _is_safe_discovery_host(host: str) -> bool:
    """Check host resolves to a non-loopback, non-link-local address."""
    try:
        addrs = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        for _fam, _type, _proto, _canon, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_loopback or ip.is_link_local:
                continue
            return True  # at least one routable address
        return False
    except (socket.gaierror, ValueError):
        return False


async def _check_health(url: str, health_path: str, timeout: float = 2.0) -> bool:
    """Check if a service is reachable via health endpoint."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(f"{url}{health_path}") as resp:
                return resp.status < 500
    except Exception:
        return False


async def _fingerprint_match(url: str, fingerprint_path: str, timeout: float = 2.0) -> bool:
    """Verify service identity by probing a known endpoint."""
    if not fingerprint_path:
        return True
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.options(f"{url}{fingerprint_path}") as resp:
                # Accept any non-404 response as a match
                return resp.status != 404
    except Exception:
        return False


async def discover(config: DiscoveryConfig | None = None) -> list[ProviderEntry]:
    """Scan for available memory services.

    Tier 1: local ports + paths (if config.local is True)
    Tier 2: remote hosts from config
    Tier 3: direct endpoints from config
    Tier 4: MCP config files
    """
    config = config or DiscoveryConfig()
    found: list[ProviderEntry] = []
    seen_names: set[str] = set()

    # Build host list
    hosts: list[str] = []
    if config.local:
        hosts.append("localhost")
    hosts.extend(config.hosts)

    # 1. Port scan on all hosts (skip private IPs for remote hosts)
    for host in hosts:
        if host != "localhost" and not _is_safe_discovery_host(host):
            logger.warning("Skipping unsafe discovery host: %s", host)
            continue
        for svc in KNOWN_SERVICES:
            if svc["name"] in seen_names:
                continue
            for port in svc["ports"]:
                url = f"http://{host}:{port}"
                health_path = svc.get("health")
                if not health_path:
                    continue

                if await _check_health(url, health_path):
                    if await _fingerprint_match(url, svc.get("fingerprint", "")):
                        entry = _build_entry_from_service(svc, url=url)
                        found.append(entry)
                        seen_names.add(svc["name"])
                        logger.info("Discovered %s at %s", svc["name"], url)
                        break  # found on this host, skip other ports

    # 2. Direct endpoints (user-provided, skip scan)
    for endpoint in config.endpoints:
        svc = await _detect_service_type(endpoint)
        if svc and svc["name"] not in seen_names:
            entry = _build_entry_from_service(svc, url=endpoint)
            found.append(entry)
            seen_names.add(svc["name"])
            logger.info("Discovered %s at %s (direct endpoint)", svc["name"], endpoint)

    # 3. Local path scan (file-based providers)
    for svc in KNOWN_SERVICES:
        if svc["name"] in seen_names:
            continue
        for path in svc["paths"]:
            expanded = Path(os.path.expanduser(path))
            if expanded.exists() and expanded.is_dir():
                entry = _build_entry_from_service(svc, path=str(expanded))
                found.append(entry)
                seen_names.add(svc["name"])
                logger.info("Discovered %s at %s (file path)", svc["name"], expanded)

    # 4. MCP config scan
    for mcp_config_path in ["~/.claude/settings.json", "~/.cursor/settings.json"]:
        mcp_path = Path(os.path.expanduser(mcp_config_path))
        if mcp_path.exists():
            mcp_providers = _parse_mcp_config(mcp_path, seen_names)
            found.extend(mcp_providers)
            for p in mcp_providers:
                seen_names.add(p.name)

    return found


async def _detect_service_type(endpoint: str) -> dict[str, Any] | None:
    """Try to identify which known service lives at an endpoint."""
    for svc in KNOWN_SERVICES:
        fp = svc.get("fingerprint")
        if not fp:
            continue
        if await _fingerprint_match(endpoint, fp):
            return svc
    return None


def _build_entry_from_service(
    svc: dict[str, Any],
    url: str = "",
    path: str = "",
) -> ProviderEntry:
    """Create a ProviderEntry from a known service definition."""
    defaults = svc["default_config"]
    return ProviderEntry(
        name=svc["name"],
        type=defaults["type"],
        enabled=True,
        url=url,
        path=path,
        search_endpoint=defaults.get("search_endpoint", ""),
        search_method=defaults.get("search_method", "POST"),
        search_body=defaults.get("search_body", ""),
        result_path=defaults.get("result_path", ""),
        pattern=defaults.get("pattern", "*.md"),
    )


def _parse_mcp_config(config_path: Path, seen: set[str]) -> list[ProviderEntry]:
    """Parse MCP server config and extract memory-related servers."""
    entries: list[ProviderEntry] = []
    try:
        data = json.loads(config_path.read_text())
        servers = data.get("mcpServers", {})

        # Known MCP memory tool names
        memory_tools = {"search_memory", "recall", "search", "query_memory"}

        for server_name, server_config in servers.items():
            name_lower = server_name.lower()
            # Skip if already discovered or not memory-related
            if server_name in seen:
                continue
            # Heuristic: check if server name suggests memory
            is_memory = any(kw in name_lower for kw in ["memory", "mem0", "cognee", "engram"])
            if not is_memory:
                continue

            command = server_config.get("command", "")
            args = server_config.get("args", [])
            full_cmd = f"{command} {' '.join(args)}".strip()

            entries.append(ProviderEntry(
                name=f"mcp-{server_name}",
                type="mcp",
                enabled=False,  # disabled by default, user confirms
                command=full_cmd,
                tool_name="search_memory",
                env=server_config.get("env", {}),
            ))
            logger.info("Found MCP memory server: %s", server_name)

    except Exception as e:
        logger.debug("Failed to parse MCP config %s: %s", config_path, e)

    return entries
