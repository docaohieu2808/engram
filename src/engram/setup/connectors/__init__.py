"""Connector registry for the engram setup wizard.

New connectors are added by:
  1. Creating a module under setup/connectors/ (e.g. claude_code.py)
  2. Importing the class here and appending it to _ALL_CONNECTORS
"""

from __future__ import annotations

from engram.setup.connectors.base import AgentConnector, ConfigureResult, DetectionResult

# Registry — append connector classes here as they are implemented in later phases
_ALL_CONNECTORS: list[type[AgentConnector]] = []


def get_all_connectors() -> list[AgentConnector]:
    """Instantiate and return all registered connectors, sorted by tier (ascending)."""
    return sorted([cls() for cls in _ALL_CONNECTORS], key=lambda c: c.tier)


__all__ = [
    "AgentConnector",
    "ConfigureResult",
    "DetectionResult",
    "get_all_connectors",
]

# Import connector modules so they self-register into _ALL_CONNECTORS
# Tier 1
from engram.setup.connectors import claude_code, cursor, openclaw, windsurf  # noqa: E402, F401
# Tier 2
from engram.setup.connectors import antigravity, aider, cline, void_editor, zed  # noqa: E402, F401
