"""MCP tools for episodic memory: remember, recall, ingest, cleanup."""

from __future__ import annotations

import json
from typing import Union

from engram.models import MemoryType


def register(mcp, get_episodic, get_graph, get_config) -> None:
    """Register episodic MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_remember(
        content: str,
        memory_type: str = "fact",
        priority: int = 5,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Store a memory in engram's episodic (vector) memory.

        Args:
            content: The memory content to store
            memory_type: Type of memory (fact, decision, preference, todo, error, context, workflow)
            priority: Priority level 1-10 (5=normal, 7=high, 10=critical)
            entities: Optional list of related entity names to link with semantic graph
            tags: Optional list of tags for filtering
            namespace: Override config namespace for this operation
        """
        store = _get_store(get_episodic, get_config, namespace)
        mem_type = MemoryType(memory_type)
        mem_id = await store.remember(
            content, memory_type=mem_type, priority=priority,
            entities=entities or [], tags=tags or [],
        )
        return f"Remembered (id={mem_id[:8]}, type={memory_type}, priority={priority})"

    @mcp.tool()
    async def engram_recall(
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Search episodic memories by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return (default 5)
            memory_type: Optional filter by type (fact, decision, preference, etc.)
            tags: Optional list of tags to filter by (all must match)
            namespace: Override config namespace for this operation
        """
        store = _get_store(get_episodic, get_config, namespace)
        filters = {"memory_type": memory_type} if memory_type else None
        results = await store.search(query, limit=limit, filters=filters, tags=tags)

        if not results:
            return "No memories found."

        lines = []
        for mem in results:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            entities_str = f" [entities: {', '.join(mem.entities)}]" if mem.entities else ""
            tags_str = f" [tags: {', '.join(mem.tags)}]" if mem.tags else ""
            lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}{entities_str}{tags_str}")
        return "\n".join(lines)

    @mcp.tool()
    async def engram_cleanup(namespace: str | None = None) -> str:
        """Delete all expired memories from the episodic store.

        Args:
            namespace: Override config namespace for this operation
        """
        store = _get_store(get_episodic, get_config, namespace)
        deleted = await store.cleanup_expired()
        if deleted == 0:
            return "No expired memories found."
        return f"Cleaned up {deleted} expired {'memory' if deleted == 1 else 'memories'}."

    @mcp.tool()
    async def engram_ingest(messages: Union[list[dict], str]) -> str:
        """Dual ingest: extract entities to graph AND remember context to vector.

        Processes chat messages through LLM extraction to build the knowledge graph
        while simultaneously storing each message as an episodic memory.

        Args:
            messages: List of message objects or JSON string of
                      [{"role": "user", "content": "..."}]
        """
        from engram.capture.extractor import EntityExtractor
        from engram.schema.loader import load_schema

        # Accept both list[dict] and JSON string for backward compatibility
        if isinstance(messages, str):
            messages = json.loads(messages)

        cfg = get_config()
        schema = load_schema(cfg.semantic.schema_name)
        extractor = EntityExtractor(model=cfg.llm.model, schema=schema)
        graph = get_graph()
        episodic = get_episodic()

        result = await extractor.extract_entities(messages)
        for node in result.nodes:
            await graph.add_node(node)
        for edge in result.edges:
            await graph.add_edge(edge)

        entity_names = [n.name for n in result.nodes]
        count = 0
        for msg in messages:
            content = msg.get("content", "")
            if content:
                await episodic.remember(content, entities=entity_names)
                count += 1

        return f"Ingested: {count} memories, {len(result.nodes)} entities, {len(result.edges)} relations"


def _get_store(get_episodic, get_config, namespace: str | None):
    """Return episodic store, optionally with a namespace override."""
    if namespace is None:
        return get_episodic()
    # Build a fresh store with the requested namespace (don't mutate shared instance)
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(
        cfg.episodic, cfg.embedding,
        namespace=namespace,
        on_remember_hook=cfg.hooks.on_remember,
    )
