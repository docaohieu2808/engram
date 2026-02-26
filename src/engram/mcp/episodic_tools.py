"""MCP tools for episodic memory: remember, recall, ingest, cleanup."""

from __future__ import annotations

import json
from typing import Union

from engram.models import MemoryType


def register(mcp, get_episodic, get_graph, get_config, get_providers=None) -> None:
    """Register episodic MCP tools on the FastMCP instance."""

    @mcp.tool()
    async def engram_remember(
        content: str,
        memory_type: str = "fact",
        priority: int = 5,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        namespace: str | None = None,
        topic_key: str | None = None,
    ) -> str:
        """Store a memory in engram's episodic (vector) memory.

        Args:
            content: The memory content to store
            memory_type: Type of memory (fact, decision, preference, todo, error, context, workflow)
            priority: Priority level 1-10 (5=normal, 7=high, 10=critical)
            entities: Optional list of related entity names to link with semantic graph
            tags: Optional list of tags for filtering
            namespace: Override config namespace for this operation
            topic_key: Optional unique key — if same key exists, updates the existing memory instead of creating new
        """
        store = _get_store(get_episodic, get_config, namespace)
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            valid = ", ".join(t.value for t in MemoryType)
            return f"Invalid memory_type '{memory_type}'. Valid: {valid}"
        # Auto-inject session_id from active session
        metadata = None
        try:
            from engram.session.store import SessionStore
            cfg = get_config()
            sess_store = SessionStore(cfg.session.sessions_dir)
            active_id = sess_store.get_active_id()
            if active_id:
                metadata = {"session_id": active_id}
        except Exception:
            pass
        mem_id = await store.remember(
            content, memory_type=mem_type, priority=priority,
            entities=entities or [], tags=tags or [],
            topic_key=topic_key, metadata=metadata,
        )
        return f"Remembered (id={mem_id[:8]}, type={memory_type}, priority={priority})"

    @mcp.tool()
    async def engram_recall(
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        namespace: str | None = None,
        compact: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Search episodic memories by semantic similarity.

        By default returns compact format with 8-char ID prefix and 120-char preview.
        Use compact=False for full content, or engram_get_memory(id) for a single full entry.

        Args:
            query: Search query text
            limit: Maximum results to return (default 5)
            memory_type: Optional filter by type (fact, decision, preference, etc.)
            tags: Optional list of tags to filter by (all must match)
            namespace: Override config namespace for this operation
            compact: If True (default), return short ID + truncated content; False returns full content
            start_date: Optional ISO date to filter memories after this date
            end_date: Optional ISO date to filter memories before this date
        """
        store = _get_store(get_episodic, get_config, namespace)
        from engram.recall.decision import should_skip_recall
        if should_skip_recall(query):
            return "No memories found."

        # Resolve pronouns using entity graph context (best-effort, graceful fallback)
        try:
            graph = get_graph()
            if graph:
                nodes = await graph.query(query)
                person_names = [n.name for n in nodes if n.type == "Person"][:5]
                if person_names:
                    from engram.recall.pronoun_resolver import resolve_pronouns
                    query = resolve_pronouns(query, person_names)
        except Exception:
            pass  # graceful fallback — never block recall on resolution failure

        if start_date or end_date:
            from engram.episodic.search import temporal_search
            results = await temporal_search(store, query, start_date, end_date, limit)
        else:
            filters = {"memory_type": memory_type} if memory_type else None
            results = await store.search(query, limit=limit, filters=filters, tags=tags)

        lines = []

        # Also search semantic graph for matching entities
        graph = get_graph()
        graph_nodes = await graph.query(query) if graph else []
        for node in graph_nodes[:3]:
            attrs = ", ".join(f"{k}={v}" for k, v in node.attributes.items()) if node.attributes else ""
            line = f"[graph] {node.type}:{node.name}" + (f" ({attrs})" if attrs else "")
            lines.append(line)
            related = await graph.get_related([node.name])
            for edge in related.get(node.name, {}).get("edges", [])[:5]:
                lines.append(f"  {edge.from_node} --{edge.relation}--> {edge.to_node}")

        # Federated search across external providers
        if get_providers:
            from engram.providers.router import federated_search
            providers = get_providers()
            if providers:
                provider_results = await federated_search(query, providers, limit=3)
                for r in provider_results:
                    lines.append(f"[{r.source}] {r.content[:300]}")

        if not results and not lines:
            return "No memories found."

        for mem in results:
            if compact:
                date = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                preview = mem.content[:120]
                suffix = "..." if len(mem.content) > 120 else ""
                entities_hint = f" [{', '.join(mem.entities)}]" if mem.entities else ""
                tags_hint = f" #{','.join(mem.tags)}" if mem.tags else ""
                lines.append(f"[{mem.id[:8]}] {date} ({mem.memory_type.value}) {preview}{suffix}{entities_hint}{tags_hint}")
            else:
                ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
                entities_str = f" [entities: {', '.join(mem.entities)}]" if mem.entities else ""
                tags_str = f" [tags: {', '.join(mem.tags)}]" if mem.tags else ""
                lines.append(f"[{ts}] ({mem.memory_type.value}) {mem.content}{entities_str}{tags_str}")

        if compact and results:
            lines.append("\nUse engram_get_memory(id) for full content.")

        return "\n".join(lines)

    @mcp.tool()
    async def engram_get_memory(
        memory_id: str,
        namespace: str | None = None,
    ) -> str:
        """Retrieve the full untruncated content of a specific memory by ID.

        Supports full UUID or 8-character prefix from engram_recall output.

        Args:
            memory_id: Full UUID or 8-char prefix of the memory ID
            namespace: Override config namespace for this operation
        """
        store = _get_store(get_episodic, get_config, namespace)
        mem = await store.get(memory_id)
        # Try prefix lookup if direct ID not found
        if mem is None and len(memory_id) <= 8:
            recent = await store.get_recent(n=200)
            for m in recent:
                if m.id.startswith(memory_id):
                    mem = m
                    break
        if not mem:
            return f"Memory '{memory_id}' not found."
        lines = [
            f"ID: {mem.id}",
            f"Type: {mem.memory_type.value}  Priority: {mem.priority}",
            f"Timestamp: {mem.timestamp.isoformat()}",
        ]
        if mem.topic_key:
            lines.append(f"Topic Key: {mem.topic_key} (revision {mem.revision_count})")
        lines.extend([
            f"Tags: {', '.join(mem.tags) or 'none'}",
            f"Entities: {', '.join(mem.entities) or 'none'}",
            f"Access Count: {mem.access_count}",
            "",
            mem.content,
        ])
        return "\n".join(lines)

    @mcp.tool()
    async def engram_timeline(
        memory_id: str,
        window_minutes: int = 30,
        namespace: str | None = None,
    ) -> str:
        """Return chronological context: memories created around the same time as a given memory.

        Useful for understanding what was happening around a key event or decision.

        Args:
            memory_id: ID of the anchor memory (full UUID or 8-char prefix)
            window_minutes: Minutes before/after to include (default 30)
            namespace: Override config namespace for this operation
        """
        store = _get_store(get_episodic, get_config, namespace)
        # Resolve anchor
        anchor = await store.get(memory_id)
        if anchor is None and len(memory_id) <= 8:
            recent = await store.get_recent(n=200)
            for m in recent:
                if m.id.startswith(memory_id):
                    anchor = m
                    break
        if not anchor:
            return f"Memory '{memory_id}' not found."

        from datetime import timedelta
        window = timedelta(minutes=window_minutes)
        all_recent = await store.get_recent(n=200)
        nearby = [
            m for m in all_recent
            if abs((m.timestamp - anchor.timestamp).total_seconds()) <= window.total_seconds()
            and m.id != anchor.id
        ]
        nearby.sort(key=lambda m: m.timestamp.isoformat() if m.timestamp else "")

        lines = [f"Timeline around [{memory_id[:8]}] (±{window_minutes}min):"]
        for m in nearby:
            ts = m.timestamp.strftime("%H:%M")
            lines.append(f"  [{ts}] ({m.memory_type.value}) {m.content[:100]}")
        # Show anchor
        lines.append(f">>> [{anchor.timestamp.strftime('%H:%M')}] (anchor) {anchor.content[:100]}")

        return "\n".join(lines) if len(nearby) > 0 else f"No nearby memories within ±{window_minutes}min."

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
    async def engram_cleanup_dedup(
        threshold: float = 0.85,
        dry_run: bool = False,
        namespace: str | None = None,
    ) -> str:
        """Retroactively deduplicate existing memories by cosine similarity.

        Scans all memories, merges near-duplicates into the higher-priority winner,
        and deletes the losers. Use dry_run=True first to preview what would change.

        Args:
            threshold: Similarity cutoff 0.0-1.0 (default 0.85). Lower = more aggressive.
            dry_run: If True, report what WOULD be merged without deleting anything.
            namespace: Override config namespace for this operation.
        """
        store = _get_store(get_episodic, get_config, namespace)
        result = await store.cleanup_dedup(threshold=threshold, dry_run=dry_run)
        mode = "[DRY RUN] " if dry_run else ""
        merged = result["merged"]
        deleted = result["deleted"]
        remaining = result["remaining"]
        if merged == 0:
            return f"{mode}No duplicates found (threshold={threshold})."
        return (
            f"{mode}Dedup complete: {merged} duplicate group(s) merged, "
            f"{deleted} {'would be ' if dry_run else ''}deleted, "
            f"{remaining} memories remaining."
        )

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
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"

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

    @mcp.tool()
    async def engram_feedback(
        memory_id: str,
        feedback: str,
        namespace: str | None = None,
    ) -> str:
        """Provide feedback on a recalled memory to improve future retrieval.

        Args:
            memory_id: Full or 8-char prefix of memory ID
            feedback: "positive" (correct/helpful) or "negative" (wrong/unhelpful)
            namespace: Override namespace
        """
        store = _get_store(get_episodic, get_config, namespace)
        from engram.feedback.auto_adjust import adjust_memory
        result = await adjust_memory(store, memory_id, feedback)
        if result.get("error"):
            return f"Error: {result['error']}"
        if result["action"] == "deleted":
            return f"Memory {memory_id[:8]} auto-deleted (3× negative, confidence {result['confidence']:.2f})"
        return f"Memory {memory_id[:8]} adjusted: confidence={result['confidence']:.2f}, importance={result['importance']}"


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
        guard_enabled=cfg.ingestion.poisoning_guard,
    )
