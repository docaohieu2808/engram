"""Base classes for agent/IDE connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionResult:
    """Result of probing the system for an installed agent."""

    installed: bool
    name: str               # Display name e.g. "Claude Code"
    version: str | None     # Detected version string, or None
    config_path: Path | None  # Primary config file path, or None
    details: str            # Extra info for display (binary path, etc.)


@dataclass
class ConfigureResult:
    """Result of writing engram config into an agent's config files."""

    success: bool
    message: str                          # Human-readable summary
    files_modified: list[Path] = field(default_factory=list)
    backup_path: Path | None = None       # Backup of original config, if made


class AgentConnector(ABC):
    """Abstract base for all agent/IDE connectors.

    Subclasses declare class-level attributes:
        name         — machine name, e.g. "claude-code"
        display_name — human label, e.g. "Claude Code"
        tier         — priority tier (1 = highest)
    """

    name: str
    display_name: str
    tier: int

    @abstractmethod
    def detect(self) -> DetectionResult:
        """Check whether this agent is installed on the current system."""

    @abstractmethod
    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Write engram configuration into the agent's config files.

        Args:
            dry_run: If True, preview changes without writing anything.
        """

    @abstractmethod
    def verify(self) -> bool:
        """Return True if the current configuration is working correctly."""
