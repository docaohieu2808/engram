"""NetworkX in-memory graph with pluggable persistence backend."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

from engram.models import SemanticEdge, SemanticNode
from engram.semantic.backend import GraphBackend
from engram.utils import strip_diacritics as _strip_diacritics

if TYPE_CHECKING:
    from engram.audit import AuditLogger

logger = logging.getLogger("engram")


def _normalize_node_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _canonicalize_node_key(key: str) -> str:
    if ":" not in key:
        return key
    type_, name = key.split(":", 1)
    return f"{type_}:{_normalize_node_name(name)}"


class SemanticGraph:
    """NetworkX in-memory graph with pluggable storage backend (SQLite or PostgreSQL)."""

    def __init__(self, backend: GraphBackend, audit: "AuditLogger | None" = None, tenant_id: str = "default", max_nodes: int = 50_000) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._backend = backend
        self._loaded: bool = False  # deferred loading flag
        self._initialized: bool = False  # backend.initialize() called flag
        self._audit = audit
        self._tenant_id = tenant_id  # M10: used in audit log calls
        self._max_nodes = max_nodes
        self._load_lock = asyncio.Lock()  # D-C1: guard concurrent loads

    async def _ensure_loaded(self) -> None:
        """Initialize backend and load graph on first access (D-C1: lock-guarded)."""
        if self._loaded:
            return
        async with self._load_lock:
            # Double-check after acquiring the lock
            if self._loaded:
                return
            if not self._initialized:
                await self._backend.initialize()
                self._initialized = True
            nodes = await self._backend.load_nodes()
            if len(nodes) > self._max_nodes:
                logger.warning(
                    "Graph has %d nodes (max_nodes=%d). Loading first %d only.",
                    len(nodes), self._max_nodes, self._max_nodes,
                )
                nodes = nodes[:self._max_nodes]
            loaded_keys: set[str] = set()
            key_map: dict[str, str] = {}
            for key, type_, name, attrs in nodes:
                node = SemanticNode(type=type_, name=name, attributes=json.loads(attrs or "{}"))
                canonical_key = node.key
                key_map[key] = canonical_key
                loaded_keys.add(canonical_key)

                # Merge case-variant duplicates (e.g. "engram" + "Engram" -> "Engram")
                if canonical_key in self._graph:
                    existing: SemanticNode | None = self._graph.nodes[canonical_key].get("data")
                    merged_attrs = dict(existing.attributes if existing else {})
                    merged_attrs.update(node.attributes)
                    if existing and existing.name != node.name:
                        aliases = list(merged_attrs.get("aliases", []))
                        for alias in (existing.name, node.name):
                            if alias and alias != node.name and alias not in aliases:
                                aliases.append(alias)
                        if aliases:
                            merged_attrs["aliases"] = aliases
                    self._graph.nodes[canonical_key]["data"] = SemanticNode(
                        type=node.type,
                        name=node.name,
                        attributes=merged_attrs,
                    )
                else:
                    self._graph.add_node(canonical_key, data=node)
            # D-H5: skip edges whose endpoints were truncated due to max_nodes cap
            for row in await self._backend.load_edges():
                key, from_key, to_key, relation = row[0], row[1], row[2], row[3]
                mapped_from = key_map.get(from_key, from_key)
                mapped_to = key_map.get(to_key, to_key)
                if mapped_from not in loaded_keys or mapped_to not in loaded_keys:
                    continue  # skip edges referencing truncated nodes
                weight = float(row[4]) if len(row) > 4 else 1.0
                attrs = json.loads(row[5]) if len(row) > 5 and row[5] else {}
                edge = SemanticEdge(from_node=mapped_from, to_node=mapped_to, relation=relation,
                                    weight=weight, attributes=attrs)
                self._graph.add_edge(mapped_from, mapped_to, key=edge.key, data=edge)
            self._loaded = True

    async def close(self) -> None:
        """Close the backend (connection pool or file handle)."""
        await self._backend.close()

    async def add_node(self, node: SemanticNode) -> bool:
        """Add to both stores. Return True if new, False if updated."""
        await self._ensure_loaded()
        is_new = node.key not in self._graph
        await self._backend.save_node(node.key, node.type, node.name, json.dumps(node.attributes))
        self._graph.add_node(node.key, data=node)
        if self._audit:
            op = "semantic.add_node" if is_new else "semantic.update_node"
            self._audit.log(tenant_id=self._tenant_id, actor="system", operation=op,
                            resource_id=node.key, details={"type": node.type, "name": node.name})
        return is_new

    async def add_edge(self, edge: SemanticEdge) -> bool:
        """Add to both stores. Return True if new."""
        await self._ensure_loaded()
        from_key = _canonicalize_node_key(edge.from_node)
        to_key = _canonicalize_node_key(edge.to_node)
        normalized_edge = SemanticEdge(
            from_node=from_key,
            to_node=to_key,
            relation=edge.relation,
            weight=edge.weight,
            attributes=edge.attributes,
        )
        is_new = not self._graph.has_edge(from_key, to_key)
        await self._backend.save_edge(
            normalized_edge.key, from_key, to_key, normalized_edge.relation,
            normalized_edge.weight, json.dumps(normalized_edge.attributes),
        )
        self._graph.add_edge(from_key, to_key, key=normalized_edge.key, data=normalized_edge)
        if self._audit:
            self._audit.log(tenant_id=self._tenant_id, actor="system", operation="semantic.add_edge",
                            resource_id=normalized_edge.key, details={"relation": normalized_edge.relation,
                            "from": normalized_edge.from_node, "to": normalized_edge.to_node})
        return is_new

    async def add_nodes_batch(self, nodes: list[SemanticNode]) -> None:
        """Add multiple nodes in one backend transaction."""
        if not nodes:
            return
        await self._ensure_loaded()
        rows = [(n.key, n.type, n.name, json.dumps(n.attributes)) for n in nodes]
        await self._backend.save_nodes_batch(rows)
        for node in nodes:
            self._graph.add_node(node.key, data=node)

    async def add_edges_batch(self, edges: list[SemanticEdge]) -> None:
        """Add multiple edges in one backend transaction."""
        if not edges:
            return
        await self._ensure_loaded()
        normalized_edges = [
            SemanticEdge(
                from_node=_canonicalize_node_key(e.from_node),
                to_node=_canonicalize_node_key(e.to_node),
                relation=e.relation,
                weight=e.weight,
                attributes=e.attributes,
            )
            for e in edges
        ]
        rows = [(e.key, e.from_node, e.to_node, e.relation, e.weight, json.dumps(e.attributes)) for e in normalized_edges]
        await self._backend.save_edges_batch(rows)
        for edge in normalized_edges:
            self._graph.add_edge(edge.from_node, edge.to_node, key=edge.key, data=edge)

    async def get_node(self, key: str) -> SemanticNode | None:
        """Lookup node from NetworkX by key."""
        await self._ensure_loaded()
        if key in self._graph:
            return self._graph.nodes[key].get("data")
        return None

    async def get_nodes(self, type: str | None = None) -> list[SemanticNode]:
        """Get all nodes, optionally filtered by type."""
        await self._ensure_loaded()
        nodes = [data["data"] for _, data in self._graph.nodes(data=True) if "data" in data]
        if type is not None:
            nodes = [n for n in nodes if n.type == type]
        return nodes

    async def get_edges(self, node_key: str | None = None) -> list[SemanticEdge]:
        """Get edges, optionally filtered by connected node."""
        await self._ensure_loaded()
        all_edges = [
            data["data"]
            for _, _, data in self._graph.edges(data=True)
            if "data" in data
        ]
        if node_key is not None:
            all_edges = [e for e in all_edges if e.from_node == node_key or e.to_node == node_key]
        return all_edges

    async def remove_node(self, key: str) -> bool:
        """Remove from both stores, also removing connected edges (D-C2: rollback on failure)."""
        await self._ensure_loaded()
        if key not in self._graph:
            return False
        try:
            await self._backend.delete_edges_for_node(key)
            await self._backend.delete_node(key)
        except Exception:
            # Backend deletion partially failed; mark graph as dirty so next
            # _ensure_loaded will reload from the backend (source of truth).
            self._loaded = False
            raise
        self._graph.remove_node(key)  # NetworkX also removes connected edges
        if self._audit:
            self._audit.log(tenant_id=self._tenant_id, actor="system", operation="semantic.remove_node",
                            resource_id=key)
        return True

    async def remove_edge(self, key: str) -> bool:
        """Remove edge from both stores by edge key."""
        await self._ensure_loaded()
        to_remove = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("key") == key
        ]
        if not to_remove:
            return False
        await self._backend.delete_edge(key)
        for u, v in to_remove:
            self._graph.remove_edge(u, v)
        if self._audit:
            self._audit.log(tenant_id=self._tenant_id, actor="system", operation="semantic.remove_edge",
                            resource_id=key)
        return True

    async def query(self, keyword: str, type: str | None = None) -> list[SemanticNode]:
        """Search nodes by keyword in name/attributes, optionally filter by type.

        Supports Vietnamese diacritics: query 'tram' matches 'Tram'.
        """
        await self._ensure_loaded()
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
        await self._ensure_loaded()
        result: dict[str, Any] = {}
        for name in entity_names:
            canonical_name = _normalize_node_name(name)
            matching = [k for k in self._graph.nodes if k.endswith(f":{canonical_name}")]
            nodes: list[SemanticNode] = []
            edges: list[SemanticEdge] = []
            for start in matching:
                visited = nx.single_source_shortest_path_length(self._graph, start, cutoff=depth)
                for key in visited:
                    node_data: SemanticNode | None = self._graph.nodes[key].get("data")
                    if node_data:
                        nodes.append(node_data)
                # H5 fix: only scan edges of visited nodes, not all edges
                seen_edges: set[tuple[str, str, str]] = set()
                for node_key in visited:
                    for u, v, data in self._graph.edges(node_key, data=True):
                        edge_data: SemanticEdge | None = data.get("data")
                        if edge_data:
                            edge_id = (edge_data.from_node, edge_data.relation, edge_data.to_node)
                            if edge_id not in seen_edges:
                                seen_edges.add(edge_id)
                                edges.append(edge_data)
            result[name] = {"nodes": nodes, "edges": edges}
        return result

    async def stats(self) -> dict[str, Any]:
        """Return node_count, edge_count, node_types breakdown."""
        await self._ensure_loaded()
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
        await self._ensure_loaded()
        return {
            "nodes": [n.model_dump() for n in await self.get_nodes()],
            "edges": [e.model_dump() for e in await self.get_edges()],
        }
