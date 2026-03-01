"""Base class for federation memory providers in the engram setup wizard."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engram.config import ProviderEntry


class FederationProvider(ABC):
    """Abstract base for external memory federation providers.

    Subclasses represent external services (Mem0, Cognee, Zep) that can be
    wired into engram as additional recall sources via the providers config list.
    """

    name: str           # Machine name, e.g. "mem0"
    display_name: str   # Human label, e.g. "Mem0"

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this provider appears to be available/configured.

        Checks env vars, local services, or config files without prompting.
        """

    @abstractmethod
    def prompt_config(self) -> ProviderEntry | None:
        """Interactively prompt the user for connection details.

        Returns a configured ProviderEntry ready to append to config.yaml,
        or None if the user skips / cancels.
        """
