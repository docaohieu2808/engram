"""Provider registry — loads providers from config and entry_points."""

from __future__ import annotations

import logging
from typing import Any

from engram.config import Config, ProviderEntry
from engram.providers.base import MemoryProvider

logger = logging.getLogger("engram.providers.registry")

# Map of type string → adapter class (lazy imports to avoid hard deps)
_ADAPTER_FACTORIES: dict[str, type] = {}
_BUILTINS_LOADED = False


def _get_adapter_class(provider_type: str) -> type | None:
    """Get adapter class by type string, with lazy import."""
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        from engram.providers.rest_adapter import RestAdapter
        from engram.providers.file_adapter import FileAdapter
        from engram.providers.postgres_adapter import PostgresAdapter
        from engram.providers.mcp_adapter import McpAdapter

        _ADAPTER_FACTORIES.update({
            "rest": RestAdapter,
            "file": FileAdapter,
            "postgres": PostgresAdapter,
            "mcp": McpAdapter,
        })
        _BUILTINS_LOADED = True

    # Check built-in adapters
    if provider_type in _ADAPTER_FACTORIES:
        return _ADAPTER_FACTORIES[provider_type]

    # Check entry_points for third-party adapters
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="engram.providers")
        for ep in eps:
            if ep.name == provider_type:
                cls = ep.load()
                _ADAPTER_FACTORIES[provider_type] = cls
                return cls
    except Exception as e:
        logger.debug("Failed to load entry_points: %s", e)

    return None


def _build_provider(entry: ProviderEntry) -> MemoryProvider | None:
    """Build a MemoryProvider instance from a ProviderEntry config."""
    cls = _get_adapter_class(entry.type)
    if cls is None:
        logger.warning("Unknown provider type '%s' for '%s', skipping", entry.type, entry.name)
        return None

    # Build kwargs from entry fields relevant to this adapter type
    common = {
        "name": entry.name,
        "enabled": entry.enabled,
        "debug": entry.debug,
        "max_consecutive_errors": entry.max_consecutive_errors,
    }

    type_kwargs: dict[str, Any] = {}
    if entry.type == "rest":
        type_kwargs = {
            "url": entry.url,
            "search_endpoint": entry.search_endpoint,
            "search_method": entry.search_method,
            "search_body": entry.search_body,
            "result_path": entry.result_path,
            "headers": entry.headers,
            "timeout_seconds": entry.timeout_seconds,
            "auth_login_endpoint": entry.auth_login_endpoint,
            "auth_username": entry.auth_username,
            "auth_password": entry.auth_password,
        }
    elif entry.type == "file":
        type_kwargs = {
            "path": entry.path,
            "pattern": entry.pattern,
        }
    elif entry.type == "postgres":
        type_kwargs = {
            "dsn": entry.dsn,
            "search_query": entry.search_query,
        }
    elif entry.type == "mcp":
        type_kwargs = {
            "command": entry.command,
            "tool_name": entry.tool_name,
            "args": entry.args,
            "env": entry.env,
        }

    try:
        return cls(**common, **type_kwargs)
    except Exception as e:
        logger.error("Failed to create provider '%s': %s", entry.name, e)
        return None


class ProviderRegistry:
    """Manages all registered memory providers."""

    def __init__(self) -> None:
        self._providers: dict[str, MemoryProvider] = {}

    def load_from_config(self, config: Config) -> None:
        """Load all providers from config entries."""
        for entry in config.providers:
            provider = _build_provider(entry)
            if provider:
                self._providers[provider.name] = provider
                logger.info("Registered provider: %s (%s)", provider.name, provider.provider_type)

    def register(self, provider: MemoryProvider) -> None:
        """Manually register a provider instance."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> MemoryProvider | None:
        return self._providers.get(name)

    def get_active(self) -> list[MemoryProvider]:
        """Return all enabled and non-auto-disabled providers."""
        return [p for p in self._providers.values() if p.is_active]

    def get_all(self) -> list[MemoryProvider]:
        return list(self._providers.values())

    async def remove(self, name: str) -> bool:
        """Remove provider by name, closing it if it has a close() method."""
        provider = self._providers.pop(name, None)
        if provider is None:
            return False
        if hasattr(provider, "close"):
            try:
                await provider.close()
            except Exception as e:
                logger.warning("Error closing provider '%s': %s", name, e)
        return True

    async def close_all(self) -> None:
        """Close all providers that have a close() method."""
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception as e:
                    logger.warning("Error closing provider '%s': %s", provider.name, e)
