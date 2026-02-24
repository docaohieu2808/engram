"""NetworkX in-memory graph with SQLite persistence for semantic memory."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx

from engram.config import SemanticConfig
from engram.models import SemanticEdge, SemanticNode
from engram.utils import strip_diacritics as _strip_diacritics

logger = logging.getLogger("engram")


class SemanticGraph:
    """NetworkX in-memory graph with SQLite persistence."""

    def __init__(self, config: SemanticConfig) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._db_path = Path(os.path.expanduser(config.path))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._loaded: bool = False  # Deferred graph loading flag

    def _connect(self) -> sqlite3.Connection:
        """Return cached connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_loaded(self) -> None:
        """Load graph from SQLite on first access (lazy initialization)."""
        if not self._loaded:
            self._load()
            self._loaded = True

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def _load(self) -> None:
        """Create tables, indexes, and load all data into NetworkX."""
        conn = self._connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS nodes "
            "(key TEXT PRIMARY KEY, type TEXT, name TEXT, attributes TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges "
            "(key TEXT PRIMARY KEY, from_key TEXT, to_key TEXT, relation TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)")
        conn.commit()
        for key, type_, name, attrs in conn.execute("SELECT key, type, name, attributes FROM nodes"):
            node = SemanticNode(type=type_, name=name, attributes=json.loads(attrs or "{}"))
            self._graph.add_node(key, data=node)
        for key, from_key, to_key, relation in conn.execute("SELECT key, from_key, to_key, relation FROM edges"):
            edge = SemanticEdge(from_node=from_key, to_node=to_key, relation=relation)
            self._graph.add_edge(from_key, to_key, key=key, data=edge)

    def _save_node(self, node: SemanticNode) -> None:
        """Upsert single node to SQLite and NetworkX."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
            (node.key, node.type, node.name, json.dumps(node.attributes)),
        )
        conn.commit()
        self._graph.add_node(node.key, data=node)

    def _save_edge(self, edge: SemanticEdge) -> None:
        """Upsert single edge to SQLite and NetworkX."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
            (edge.key, edge.from_node, edge.to_node, edge.relation),
        )
        conn.commit()
        self._graph.add_edge(edge.from_node, edge.to_node, key=edge.key, data=edge)

    async def add_node(self, node: SemanticNode) -> bool:
        """Add to both stores. Return True if new, False if updated."""
        self._ensure_loaded()
        is_new = node.key not in self._graph
        self._save_node(node)
        return is_new

    async def add_edge(self, edge: SemanticEdge) -> bool:
        """Add to both stores. Return True if new."""
        self._ensure_loaded()
        is_new = not self._graph.has_edge(edge.from_node, edge.to_node)
        self._save_edge(edge)
        return is_new

    async def add_nodes_batch(self, nodes: list[SemanticNode]) -> None:
        """Add multiple nodes in a single SQLite transaction and bulk NetworkX add."""
        if not nodes:
            return
        self._ensure_loaded()
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
                [(n.key, n.type, n.name, json.dumps(n.attributes)) for n in nodes],
            )
        for node in nodes:
            self._graph.add_node(node.key, data=node)

    async def add_edges_batch(self, edges: list[SemanticEdge]) -> None:
        """Add multiple edges in a single SQLite transaction and bulk NetworkX add."""
        if not edges:
            return
        self._ensure_loaded()
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
                [(e.key, e.from_node, e.to_node, e.relation) for e in edges],
            )
        for edge in edges:
            self._graph.add_edge(edge.from_node, edge.to_node, key=edge.key, data=edge)

    async def get_node(self, key: str) -> SemanticNode | None:
        """Lookup node from NetworkX by key."""
        self._ensure_loaded()
        if key in self._graph:
            return self._graph.nodes[key].get("data")
        return None

    async def get_nodes(self, type: str | None = None) -> list[SemanticNode]:
        """Get all nodes, optionally filtered by type."""
        self._ensure_loaded()
        nodes = [data["data"] for _, data in self._graph.nodes(data=True) if "data" in data]
        if type is not None:
            nodes = [n for n in nodes if n.type == type]
        return nodes

    async def get_edges(self, node_key: str | None = None) -> list[SemanticEdge]:
        """Get edges, optionally filtered by connected node."""
        self._ensure_loaded()
        all_edges = [
            data["data"]
            for _, _, data in self._graph.edges(data=True)
            if "data" in data
        ]
        if node_key is not None:
            all_edges = [e for e in all_edges if e.from_node == node_key or e.to_node == node_key]
        return all_edges

    async def remove_node(self, key: str) -> bool:
        """Remove from both stores, also removing connected edges."""
        self._ensure_loaded()
        if key not in self._graph:
            return False
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE from_key=? OR to_key=?", (key, key))
        conn.execute("DELETE FROM nodes WHERE key=?", (key,))
        conn.commit()
        self._graph.remove_node(key)
        return True

    async def remove_edge(self, key: str) -> bool:
        """Remove edge from both stores by edge key."""
        self._ensure_loaded()
        # Single pass to find and collect edges to remove
        to_remove = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("key") == key
        ]
        if not to_remove:
            return False
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE key=?", (key,))
        conn.commit()
        for u, v in to_remove:
            self._graph.remove_edge(u, v)
        return True

    async def query(self, keyword: str, type: str | None = None) -> list[SemanticNode]:
        """Search nodes by keyword in name/attributes, optionally filter by type.

        Supports Vietnamese diacritics: query 'tram' matches 'Tram'.
        """
        self._ensure_loaded()
        kw = keyword.lower()
        kw_stripped = _strip_diacritics(kw)
        results: list[SemanticNode] = []
        for _, data in self._graph.nodes(data=True):
            node: SemanticNode | None = data.get("data")
            if node is None:
                continue
            if type is not None and node.type != type:
                continue
            name_lower = node.name.lower()
            attrs_lower = json.dumps(node.attributes, ensure_ascii=False).lower()
            if (kw in name_lower or kw in attrs_lower
                    or kw_stripped in _strip_diacritics(name_lower)
                    or kw_stripped in _strip_diacritics(attrs_lower)):
                results.append(node)
        return results

    async def get_related(self, entity_names: list[str], depth: int = 1) -> dict[str, Any]:
        """BFS traversal from entities up to depth hops."""
        self._ensure_loaded()
        result: dict[str, Any] = {}
        for name in entity_names:
            matching = [k for k in self._graph.nodes if k.endswith(f":{name}")]
            nodes: list[SemanticNode] = []
            edges: list[SemanticEdge] = []
            for start in matching:
                visited = nx.single_source_shortest_path_length(self._graph, start, cutoff=depth)
                for key in visited:
                    node_data: SemanticNode | None = self._graph.nodes[key].get("data")
                    if node_data:
                        nodes.append(node_data)
                for u, v, data in self._graph.edges(data=True):
                    if u in visited or v in visited:
                        edge_data: SemanticEdge | None = data.get("data")
                        if edge_data and edge_data not in edges:
                            edges.append(edge_data)
            result[name] = {"nodes": nodes, "edges": edges}
        return result

    async def stats(self) -> dict[str, Any]:
        """Return node_count, edge_count, node_types breakdown."""
        self._ensure_loaded()
        type_counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            node: SemanticNode | None = data.get("data")
            if node:
                type_counts[node.type] = type_counts.get(node.type, 0) + 1
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "node_types": type_counts,
        }

    async def dump(self) -> dict[str, Any]:
        """Export all nodes and edges as dict."""
        self._ensure_loaded()
        return {
            "nodes": [n.model_dump() for n in await self.get_nodes()],
            "edges": [e.model_dump() for e in await self.get_edges()],
        }
