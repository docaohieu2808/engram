"""Federation provider registry for the engram setup wizard.

Provides get_all_providers() which returns one instance of each registered
FederationProvider, ordered consistently for display in the wizard.
"""

from __future__ import annotations

from engram.setup.federation.base import FederationProvider
from engram.setup.federation.cognee_provider import CogneeProvider
from engram.setup.federation.mem0_provider import Mem0Provider
from engram.setup.federation.zep_provider import ZepProvider

__all__ = [
    "FederationProvider",
    "get_all_providers",
]

# Registry — add new providers here in display order
_ALL_PROVIDERS: list[type[FederationProvider]] = [
    Mem0Provider,
    CogneeProvider,
    ZepProvider,
]


def get_all_providers() -> list[FederationProvider]:
    """Return instantiated list of all registered federation providers."""
    return [cls() for cls in _ALL_PROVIDERS]
