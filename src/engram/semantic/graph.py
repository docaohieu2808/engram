"""NetworkX in-memory graph with SQLite persistence for semantic memory."""

from __future__ import annotations

import json
import os
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import networkx as nx

from engram.config import SemanticConfig
from engram.models import SemanticEdge, SemanticNode


def _strip_diacritics(text: str) -> str:
    """Remove diacritics for fuzzy matching (e.g. 'Trâm' -> 'Tram')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class SemanticGraph:
    """NetworkX in-memory graph with SQLite persistence."""

    def __init__(self, config: SemanticConfig) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._db_path = Path(os.path.expanduser(config.path))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _load(self) -> None:
        """Create tables if not exist and load all data into NetworkX."""
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS nodes "
                "(key TEXT PRIMARY KEY, type TEXT, name TEXT, attributes TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS edges "
                "(key TEXT PRIMARY KEY, from_key TEXT, to_key TEXT, relation TEXT)"
            )
            conn.commit()
            for key, type_, name, attrs in conn.execute("SELECT key, type, name, attributes FROM nodes"):
                node = SemanticNode(type=type_, name=name, attributes=json.loads(attrs or "{}"))
                self._graph.add_node(key, data=node)
            for key, from_key, to_key, relation in conn.execute("SELECT key, from_key, to_key, relation FROM edges"):
                edge = SemanticEdge(from_node=from_key, to_node=to_key, relation=relation)
                self._graph.add_edge(from_key, to_key, key=key, data=edge)

    def _save_node(self, node: SemanticNode) -> None:
        """Upsert single node to SQLite and NetworkX."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
                (node.key, node.type, node.name, json.dumps(node.attributes)),
            )
            conn.commit()
        self._graph.add_node(node.key, data=node)

    def _save_edge(self, edge: SemanticEdge) -> None:
        """Upsert single edge to SQLite and NetworkX."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
                (edge.key, edge.from_node, edge.to_node, edge.relation),
            )
            conn.commit()
        self._graph.add_edge(edge.from_node, edge.to_node, key=edge.key, data=edge)

    async def add_node(self, node: SemanticNode) -> bool:
        """Add to both stores. Return True if new, False if updated."""
        is_new = node.key not in self._graph
        self._save_node(node)
        return is_new

    async def add_edge(self, edge: SemanticEdge) -> bool:
        """Add to both stores. Return True if new."""
        is_new = not self._graph.has_edge(edge.from_node, edge.to_node)
        self._save_edge(edge)
        return is_new

    async def get_node(self, key: str) -> SemanticNode | None:
        """Lookup node from NetworkX by key."""
        if key in self._graph:
            return self._graph.nodes[key].get("data")
        return None

    async def get_nodes(self, type: str | None = None) -> list[SemanticNode]:
        """Get all nodes, optionally filtered by type."""
        nodes = [data["data"] for _, data in self._graph.nodes(data=True) if "data" in data]
        if type is not None:
            nodes = [n for n in nodes if n.type == type]
        return nodes

    async def get_edges(self, node_key: str | None = None) -> list[SemanticEdge]:
        """Get edges, optionally filtered by connected node."""
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
        if key not in self._graph:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE from_key=? OR to_key=?", (key, key))
            conn.execute("DELETE FROM nodes WHERE key=?", (key,))
            conn.commit()
        self._graph.remove_node(key)
        return True

    async def remove_edge(self, key: str) -> bool:
        """Remove edge from both stores by edge key."""
        found = any(
            data.get("key") == key
            for _, _, data in self._graph.edges(data=True)
        )
        if not found:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE key=?", (key,))
            conn.commit()
        to_remove = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("key") == key
        ]
        for u, v in to_remove:
            self._graph.remove_edge(u, v)
        return True

    async def query(self, keyword: str, type: str | None = None) -> list[SemanticNode]:
        """Search nodes by keyword in name/attributes, optionally filter by type.

        Supports Vietnamese diacritics: query 'tram' matches 'Trâm'.
        """
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
            # Match exact or with stripped diacritics
            if (kw in name_lower or kw in attrs_lower
                    or kw_stripped in _strip_diacritics(name_lower)
                    or kw_stripped in _strip_diacritics(attrs_lower)):
                results.append(node)
        return results

    async def get_related(self, entity_names: list[str], depth: int = 1) -> dict[str, Any]:
        """BFS traversal from entities up to depth hops."""
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
        return {
            "nodes": [n.model_dump() for n in await self.get_nodes()],
            "edges": [e.model_dump() for e in await self.get_edges()],
        }
