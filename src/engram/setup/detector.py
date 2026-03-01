"""Agent detection engine for the engram setup wizard."""

from __future__ import annotations

from engram.setup.connectors import get_all_connectors
from engram.setup.connectors.base import AgentConnector, DetectionResult


def scan_agents() -> list[tuple[AgentConnector, DetectionResult]]:
    """Run detection on all registered connectors.

    Returns:
        List of (connector, result) pairs for every registered connector,
        ordered by connector tier (ascending).  Includes non-installed agents
        so callers can show a full inventory.
    """
    results: list[tuple[AgentConnector, DetectionResult]] = []
    for connector in get_all_connectors():
        try:
            result = connector.detect()
        except Exception as exc:  # noqa: BLE001
            # Graceful degradation: detection errors become "not installed"
            result = DetectionResult(
                installed=False,
                name=connector.display_name,
                version=None,
                config_path=None,
                details=f"Detection error: {exc}",
            )
        results.append((connector, result))
    return results
