"""Abstract backend protocol for semantic graph storage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage backends (SQLite, PostgreSQL, etc.)."""

    async def initialize(self) -> None:
        """Create tables/indexes if they don't exist."""
        ...

    async def load_nodes(self) -> list[tuple[str, str, str, str]]:
        """Return all nodes as (key, type, name, attrs_json) tuples."""
        ...

    async def load_edges(self) -> list[tuple[str, str, str, str]]:
        """Return all edges as (key, from_key, to_key, relation) tuples."""
        ...

    async def save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        """Upsert a single node."""
        ...

    async def save_edge(self, key: str, from_key: str, to_key: str, relation: str) -> None:
        """Upsert a single edge."""
        ...

    async def save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple nodes in one transaction."""
        ...

    async def save_edges_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple edges in one transaction."""
        ...

    async def delete_node(self, key: str) -> None:
        """Delete a node by key."""
        ...

    async def delete_edges_for_node(self, key: str) -> None:
        """Delete all edges connected to a node (from or to)."""
        ...

    async def delete_edge(self, key: str) -> None:
        """Delete an edge by key."""
        ...

    async def close(self) -> None:
        """Release resources (connections, pools)."""
        ...
