"""MCP tools for semantic knowledge graph: add_entity, add_relation, query_graph."""

from __future__ import annotations

from typing import Any

from engram.models import SemanticEdge, SemanticNode


def register(mcp, get_graph) -> None:
    """Register semantic MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_add_entity(
        name: str,
        type: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add an entity node to the semantic knowledge graph.

        Args:
            name: Entity name (e.g. "PostgreSQL", "API-Service")
            type: Entity type from schema (e.g. "Technology", "Service", "Server", "Person")
            attributes: Optional attributes dict (e.g. {"version": "16", "provider": "AWS"})
        """
        graph = get_graph()
        node = SemanticNode(type=type, name=name, attributes=attributes or {})
        is_new = await graph.add_node(node)
        return f"{'Added' if is_new else 'Updated'} entity {node.key}"

    @mcp.tool()
    async def engram_add_relation(
        from_entity: str,
        to_entity: str,
        relation: str,
    ) -> str:
        """Add a relationship edge between two entities in the knowledge graph.

        Args:
            from_entity: Source entity key as "Type:Name" (e.g. "Service:API-Service")
            to_entity: Target entity key as "Type:Name" (e.g. "Technology:PostgreSQL")
            relation: Relationship type (e.g. "uses", "runs_on", "depends_on", "managed_by")
        """
        graph = get_graph()
        edge = SemanticEdge(from_node=from_entity, to_node=to_entity, relation=relation)
        is_new = await graph.add_edge(edge)
        return f"{'Added' if is_new else 'Updated'} relation {edge.key}"

    @mcp.tool()
    async def engram_query_graph(
        keyword: str | None = None,
        type: str | None = None,
        related_to: str | None = None,
    ) -> str:
        """Query the semantic knowledge graph for entities and relationships.

        Args:
            keyword: Search keyword to match entity names/attributes
            type: Filter by entity type (e.g. "Technology", "Service")
            related_to: Get entities related to this name (BFS traversal, depth=2)
        """
        graph = get_graph()

        if related_to:
            results = await graph.get_related([related_to], depth=2)
            if not results:
                return f"No entities related to '{related_to}' found."
            lines = [f"Related to: {related_to}"]
            for entity, data in results.items():
                lines.append(f"\n{entity}:")
                if isinstance(data, dict):
                    nodes = data.get("nodes", [])
                    edges = data.get("edges", [])
                    for n in nodes:
                        attrs = (
                            f" ({', '.join(f'{k}={v}' for k, v in n.attributes.items())})"
                            if n.attributes else ""
                        )
                        lines.append(f"  node: {n.type}:{n.name}{attrs}")
                    for e in edges:
                        lines.append(f"  edge: {e.from_node} --{e.relation}--> {e.to_node}")
            return "\n".join(lines)

        if keyword:
            nodes = await graph.query(keyword, type=type)
        else:
            nodes = await graph.get_nodes(type=type)

        if not nodes:
            return "No entities found."

        lines = []
        for n in nodes:
            attrs = (
                f" ({', '.join(f'{k}={v}' for k, v in n.attributes.items())})"
                if n.attributes else ""
            )
            lines.append(f"{n.key}{attrs}")
        return "\n".join(lines)
