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

    async def load_edges(self) -> list[tuple[str, str, str, str, float, str]]:
        """Return all edges as (key, from_key, to_key, relation, weight, attrs_json) tuples."""
        ...

    async def save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        """Upsert a single node."""
        ...

    async def save_edge(self, key: str, from_key: str, to_key: str, relation: str,
                        weight: float = 1.0, attrs_json: str = "{}") -> None:
        """Upsert a single edge."""
        ...

    async def save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple nodes in one transaction."""
        ...

    async def save_edges_batch(self, rows: list[tuple[str, str, str, str, float, str]]) -> None:
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

    async def query_nodes_by_name(self, pattern: str, limit: int, offset: int) -> list[dict]:
        """Return nodes whose key matches pattern (SQL LIKE) without loading full graph.

        Each dict has keys: key, type, name, attributes (JSON string).
        """
        ...

    async def get_node_by_key(self, key: str) -> dict | None:
        """Return a single node dict by exact key, or None if not found."""
        ...

    async def get_related_nodes(self, node_key: str, depth: int) -> list[dict]:
        """Return nodes related to node_key up to depth hops.

        Stub implementations may return [] — full graph traversal falls back to NetworkX.
        """
        ...

    async def close(self) -> None:
        """Release resources (connections, pools)."""
        ...
