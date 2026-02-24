"""Abstract base class for federated memory providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ProviderResult(BaseModel):
    """A single result returned by a memory provider."""

    content: str
    score: float = 0.0
    source: str = ""  # provider name
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderStats(BaseModel):
    """Runtime stats tracked per provider."""

    query_count: int = 0
    hit_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.query_count if self.query_count else 0.0


class MemoryProvider(ABC):
    """Abstract interface for external memory/knowledge providers.

    All adapters (REST, file, postgres, MCP) implement this interface.
    Includes built-in stats tracking and auto-disable on consecutive errors.
    """

    def __init__(
        self,
        name: str,
        provider_type: str,
        enabled: bool = True,
        debug: bool = False,
        max_consecutive_errors: int = 5,
    ):
        self.name = name
        self.provider_type = provider_type
        self.enabled = enabled
        self.debug = debug
        self.max_consecutive_errors = max_consecutive_errors
        self.stats = ProviderStats()
        self._auto_disabled = False

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        """Search this provider. Returns list of results with content + score."""

    @abstractmethod
    async def health(self) -> bool:
        """Check if provider is reachable."""

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str | None:
        """Optional: store content in this provider. Returns ID if supported."""
        return None

    async def tracked_search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        """Search with stats tracking and auto-disable logic."""
        if not self.enabled or self._auto_disabled:
            return []

        start = time.monotonic()
        try:
            results = await self.search(query, limit)
            elapsed_ms = (time.monotonic() - start) * 1000

            self.stats.query_count += 1
            self.stats.total_latency_ms += elapsed_ms
            self.stats.consecutive_errors = 0
            if results:
                self.stats.hit_count += 1

            # Tag results with source
            for r in results:
                r.source = self.name

            return results
        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            self.stats.query_count += 1
            self.stats.error_count += 1
            self.stats.consecutive_errors += 1
            self.stats.total_latency_ms += elapsed_ms

            # Auto-disable after too many consecutive errors
            if self.stats.consecutive_errors >= self.max_consecutive_errors:
                self._auto_disabled = True

            return []

    def re_enable(self) -> None:
        """Re-enable a provider that was auto-disabled."""
        self._auto_disabled = False
        self.stats.consecutive_errors = 0

    @property
    def is_active(self) -> bool:
        return self.enabled and not self._auto_disabled

    @property
    def status_label(self) -> str:
        if self._auto_disabled:
            return f"disabled ({self.stats.consecutive_errors} errors)"
        return "active" if self.enabled else "disabled"
