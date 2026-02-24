"""Federated memory providers for engram.

Connects engram to external memory/knowledge systems via a universal adapter interface.
Supports REST APIs, file-based stores, PostgreSQL, and MCP servers.
"""

from engram.providers.base import MemoryProvider, ProviderResult

__all__ = ["MemoryProvider", "ProviderResult"]
