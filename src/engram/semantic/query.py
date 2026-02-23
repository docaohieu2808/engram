"""Graph query helpers for semantic memory traversal and search."""

from __future__ import annotations

import networkx as nx

from engram.models import SemanticEdge, SemanticNode
from engram.semantic.graph import SemanticGraph


async def query_by_keyword(
    graph: SemanticGraph,
    keyword: str,
    type: str | None = None,
) -> list[SemanticNode]:
    """Search nodes by keyword in name/attributes, optionally filtered by type."""
    return await graph.query(keyword, type=type)


async def query_related(
    graph: SemanticGraph,
    node_key: str,
    depth: int = 1,
) -> dict[str, list]:
    """BFS from node_key up to depth hops. Return nodes and edges within depth."""
    g = graph._graph
    if node_key not in g:
        return {"nodes": [], "edges": []}

    visited = nx.single_source_shortest_path_length(g, node_key, cutoff=depth)
    nodes: list[SemanticNode] = []
    for key in visited:
        node_data: SemanticNode | None = g.nodes[key].get("data")
        if node_data:
            nodes.append(node_data)

    edges: list[SemanticEdge] = []
    seen_keys: set[str] = set()
    for u, v, data in g.edges(data=True):
        if u in visited or v in visited:
            edge_data: SemanticEdge | None = data.get("data")
            if edge_data and edge_data.key not in seen_keys:
                edges.append(edge_data)
                seen_keys.add(edge_data.key)

    return {"nodes": nodes, "edges": edges}


async def query_path(
    graph: SemanticGraph,
    from_key: str,
    to_key: str,
) -> list[str] | None:
    """Find shortest path between two nodes. Return list of node keys or None."""
    g = graph._graph
    try:
        return nx.shortest_path(g, from_key, to_key)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
