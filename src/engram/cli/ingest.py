"""Ingest helpers used by CLI watch/serve/ingest commands.

``_do_ingest`` and ``_do_ingest_messages`` are module-level async functions so
they can be imported by system.py without creating circular dependencies.
``_ingest_lock`` serialises concurrent asyncpg pool writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from engram.models import IngestResult

logger = logging.getLogger("engram")

_ingest_lock = asyncio.Lock()


async def do_ingest(file: Path, dry_run: bool, get_extractor, get_graph, get_episodic) -> IngestResult:
    """Run dual ingest on a chat file (entities + episodic memories)."""
    from rich.console import Console
    console = Console()

    with open(file) as f:
        data = json.load(f)

    messages = data if isinstance(data, list) else data.get("messages", [])
    if not messages:
        return IngestResult()

    extractor = get_extractor()
    result = await extractor.extract_entities(messages)

    if dry_run:
        console.print("[bold]Dry run - extracted entities:[/bold]")
        for n in result.nodes:
            console.print(f"  [cyan]Node:[/cyan] {n.key}")
        for e in result.edges:
            console.print(f"  [green]Edge:[/green] {e.key}")
        return IngestResult(semantic_nodes=len(result.nodes), semantic_edges=len(result.edges))

    graph = get_graph()
    episodic = get_episodic()
    async with _ingest_lock:
        if result.nodes:
            await graph.add_nodes_batch(result.nodes)
        if result.edges:
            await graph.add_edges_batch(result.edges)
    entity_names = [n.name for n in result.nodes]
    episodic_count = 0
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        ctx = messages[max(0, i - 2): min(len(messages), i + 3)]
        per_content = extractor.filter_entities_for_content(content, entity_names, context_messages=ctx)
        if per_content:
            await episodic.remember(content, entities=per_content)
            episodic_count += 1

    return IngestResult(
        episodic_count=episodic_count,
        semantic_nodes=len(result.nodes),
        semantic_edges=len(result.edges),
    )


async def do_ingest_messages(
    messages: list[dict], get_extractor, get_graph, get_episodic, source: str = "",
) -> IngestResult:
    """Ingest messages (called by watcher/server).

    Only saves messages that have entities extracted (entity-gated ingestion).
    Messages without any entity/node/edge are noise and get skipped.
    Uses _ingest_lock to prevent concurrent asyncpg pool contention.
    """
    episodic = get_episodic()
    episodic_count = 0
    semantic_nodes = 0
    semantic_edges = 0

    try:
        extractor = get_extractor()
        graph = get_graph()
        result = await extractor.extract_entities(messages)

        if not result.nodes and not result.edges:
            logger.debug("No entities extracted — skipping all %d messages", len(messages))
            return IngestResult()

        # Store nodes and edges into semantic graph
        async with _ingest_lock:
            if result.nodes:
                await graph.add_nodes_batch(result.nodes)
            if result.edges:
                await graph.add_edges_batch(result.edges)
        semantic_nodes = len(result.nodes)
        semantic_edges = len(result.edges)

        # Build full entity name list (extracted + existing graph)
        entity_names = [n.name for n in result.nodes]
        try:
            async with _ingest_lock:
                all_nodes = await graph.get_nodes()
            seen = {n.casefold() for n in entity_names}
            for kn in [n.name for n in all_nodes]:
                if kn.casefold() not in seen:
                    entity_names.append(kn)
                    seen.add(kn.casefold())
        except Exception:
            pass  # Fall back to LLM-only names

        # Only save messages that mention at least one known entity
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue
            ctx = messages[max(0, i - 2): min(len(messages), i + 3)]
            per_content = extractor.filter_entities_for_content(
                content, entity_names, context_messages=ctx,
            )
            if per_content:
                await episodic.remember(content, entities=per_content, source=source)
                episodic_count += 1

    except Exception as e:
        logger.warning("Entity extraction failed — no messages stored: %s", e)

    return IngestResult(
        episodic_count=episodic_count,
        semantic_nodes=semantic_nodes,
        semantic_edges=semantic_edges,
    )
