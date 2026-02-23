"""MCP server exposing engram memory tools to Claude Code and other MCP clients."""

from __future__ import annotations

import json
from typing import Any

from mcp.server import FastMCP

from engram.config import load_config
from engram.episodic.store import EpisodicStore
from engram.models import MemoryType, SemanticEdge, SemanticNode
from engram.reasoning.engine import ReasoningEngine
from engram.schema.loader import load_schema
from engram.semantic.graph import SemanticGraph

# Create MCP server
mcp = FastMCP("engram", instructions="AI agent brain with dual memory (episodic + semantic)")

# Lazy-initialized shared instances
_instances: dict[str, Any] = {}


def _get_config():
    if "config" not in _instances:
        _instances["config"] = load_config()
    return _instances["config"]


def _get_episodic() -> EpisodicStore:
    if "episodic" not in _instances:
        cfg = _get_config()
        _instances["episodic"] = EpisodicStore(cfg.episodic, cfg.embedding)
    return _instances["episodic"]


def _get_graph() -> SemanticGraph:
    if "graph" not in _instances:
        cfg = _get_config()
        _instances["graph"] = SemanticGraph(cfg.semantic)
    return _instances["graph"]


def _get_engine() -> ReasoningEngine:
    if "engine" not in _instances:
        cfg = _get_config()
        _instances["engine"] = ReasoningEngine(_get_episodic(), _get_graph(), model=cfg.llm.model)
    return _instances["engine"]


# === Episodic Memory Tools ===


@mcp.tool()
async def engram_remember(
    content: str,
    memory_type: str = "fact",
    priority: int = 5,
    entities: list[str] | None = None,
) -> str:
    """Store a memory in engram's episodic (vector) memory.

    Args:
        content: The memory content to store
        memory_type: Type of memory (fact, decision, preference, todo, error, context, workflow)
        priority: Priority level 1-10 (5=normal, 7=high, 10=critical)
        entities: Optional list of related entity names to link with semantic graph
    """
    store = _get_episodic()
    mem_type = MemoryType(memory_type)
    mem_id = await store.remember(content, memory_type=mem_type, priority=priority, entities=entities or [])
    return f"Remembered (id={mem_id[:8]}, type={memory_type}, priority={priority})"


@mcp.tool()
async def engram_recall(
    query: str,
    limit: int = 5,
    memory_type: str | None = None,
) -> str:
    """Search episodic memories by semantic similarity.

    Args:
        query: Search query text
        limit: Maximum results to return (default 5)
        memory_type: Optional filter by type (fact, decision, preference, etc.)
    """
    store = _get_episodic()
    filters = {"memory_type": memory_type} if memory_type else None
    results = await store.search(query, limit=limit, filters=filters)

    if not results:
        return "No memories found."

    lines = []
    for mem in results:
        ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
        entities_str = f" [entities: {', '.join(mem.entities)}]" if mem.entities else ""
        lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}{entities_str}")
    return "\n".join(lines)


# === Semantic Memory Tools ===


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
    graph = _get_graph()
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
    graph = _get_graph()
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
    graph = _get_graph()

    if related_to:
        results = await graph.get_related([related_to], depth=2)
        if not results:
            return f"No entities related to '{related_to}' found."
        lines = [f"Related to: {related_to}"]
        for entity, data in results.items():
            lines.append(f"\n{entity}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    if keyword:
        nodes = await graph.query(keyword, type=type)
    else:
        nodes = await graph.get_nodes(type=type)

    if not nodes:
        return "No entities found."

    lines = []
    for n in nodes:
        attrs = f" ({', '.join(f'{k}={v}' for k, v in n.attributes.items())})" if n.attributes else ""
        lines.append(f"{n.key}{attrs}")
    return "\n".join(lines)


# === Combined Reasoning ===


@mcp.tool()
async def engram_think(question: str) -> str:
    """Combined reasoning across both episodic and semantic memory.

    Searches vector DB for relevant experiences, traverses knowledge graph
    for entity relationships, then synthesizes an answer using LLM.

    Args:
        question: Question to reason about (e.g. "What database issues happened recently?")
    """
    engine = _get_engine()
    return await engine.think(question)


# === Status ===


@mcp.tool()
async def engram_status() -> str:
    """Show engram memory statistics - counts for both episodic and semantic stores."""
    episodic = _get_episodic()
    graph = _get_graph()

    ep_stats = await episodic.stats()
    sem_stats = await graph.stats()

    lines = [
        "Engram Memory Status:",
        f"  Episodic: {ep_stats.get('count', 0)} memories",
        f"  Semantic: {sem_stats.get('node_count', 0)} nodes, {sem_stats.get('edge_count', 0)} edges",
    ]
    if "node_types" in sem_stats:
        for t, c in sem_stats["node_types"].items():
            lines.append(f"    {t}: {c}")
    return "\n".join(lines)


# === Dual Ingest ===


@mcp.tool()
async def engram_ingest(messages_json: str) -> str:
    """Dual ingest: extract entities to graph AND remember context to vector.

    Processes chat messages through LLM extraction to build the knowledge graph
    while simultaneously storing each message as an episodic memory.

    Args:
        messages_json: JSON array of message objects [{"role": "user", "content": "..."}]
    """
    from engram.capture.extractor import EntityExtractor

    cfg = _get_config()
    schema = load_schema(cfg.semantic.schema_name)
    extractor = EntityExtractor(model=cfg.llm.model, schema=schema)
    graph = _get_graph()
    episodic = _get_episodic()

    messages = json.loads(messages_json)

    # Extract entities → graph
    result = await extractor.extract_entities(messages)
    for node in result.nodes:
        await graph.add_node(node)
    for edge in result.edges:
        await graph.add_edge(edge)

    # Remember context → vector
    entity_names = [n.name for n in result.nodes]
    count = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            await episodic.remember(content, entities=entity_names)
            count += 1

    return f"Ingested: {count} memories, {len(result.nodes)} entities, {len(result.edges)} relations"


def main():
    """Run the MCP server via stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
