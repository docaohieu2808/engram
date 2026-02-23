"""engram CLI - Memory traces for AI agents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from engram.config import (
    Config,
    _expand_path,
    get_config_value,
    load_config,
    save_config,
    set_config_value,
)
from engram.models import IngestResult, MemoryType

app = typer.Typer(name="engram", help="Memory traces for AI agents - Think like human")
add_app = typer.Typer(help="Add nodes/edges to semantic memory")
remove_app = typer.Typer(help="Remove nodes/edges from semantic memory")
schema_app = typer.Typer(help="Manage semantic schemas")
config_app = typer.Typer(help="Manage configuration")

app.add_typer(add_app, name="add")
app.add_typer(remove_app, name="remove")
app.add_typer(schema_app, name="schema")
app.add_typer(config_app, name="config")

console = Console()
_config: Config | None = None


def _run(coro: Any) -> Any:
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_episodic():
    from engram.episodic.store import EpisodicStore
    cfg = _get_config()
    return EpisodicStore(cfg.episodic, cfg.embedding)


def _get_graph():
    from engram.semantic.graph import SemanticGraph
    cfg = _get_config()
    return SemanticGraph(cfg.semantic)


def _get_extractor():
    from engram.capture.extractor import EntityExtractor
    from engram.schema.loader import load_schema
    cfg = _get_config()
    schema = load_schema(cfg.semantic.schema_name)
    return EntityExtractor(model=cfg.llm.model, schema=schema)


def _get_engine():
    from engram.reasoning.engine import ReasoningEngine
    cfg = _get_config()
    return ReasoningEngine(_get_episodic(), _get_graph(), model=cfg.llm.model)


# === Episodic Commands ===


@app.command()
def remember(
    content: str = typer.Argument(..., help="Content to remember"),
    type: str = typer.Option("fact", "--type", "-t", help="Memory type"),
    priority: int = typer.Option(5, "--priority", "-p", help="Priority (1-10)"),
):
    """Store a memory in episodic store."""
    store = _get_episodic()
    mem_type = MemoryType(type)
    mem_id = _run(store.remember(content, memory_type=mem_type, priority=priority))
    console.print(f"[green]Remembered[/green] (id={mem_id[:8]}..., type={type})")


@app.command()
def recall(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l"),
    type: Optional[str] = typer.Option(None, "--type", "-t"),
):
    """Search episodic memories."""
    store = _get_episodic()
    filters = {}
    if type:
        filters["memory_type"] = type
    results = _run(store.search(query, limit=limit, filters=filters if filters else None))

    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    for mem in results:
        ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
        console.print(f"[cyan][{ts}][/cyan] ({mem.memory_type.value}) {mem.content}")
        if mem.entities:
            console.print(f"  [dim]entities: {', '.join(mem.entities)}[/dim]")


# === Semantic Commands ===


@add_app.command("node")
def add_node(
    name: str = typer.Argument(..., help="Node name"),
    type: str = typer.Option(..., "--type", "-t", help="Node type"),
):
    """Add a node to semantic graph."""
    from engram.models import SemanticNode
    graph = _get_graph()
    node = SemanticNode(type=type, name=name)
    is_new = _run(graph.add_node(node))
    action = "Added" if is_new else "Updated"
    console.print(f"[green]{action}[/green] node {node.key}")


@add_app.command("edge")
def add_edge(
    from_node: str = typer.Argument(..., help="Source node (Type:Name)"),
    to_node: str = typer.Argument(..., help="Target node (Type:Name)"),
    relation: str = typer.Option(..., "--relation", "-r", help="Relationship type"),
):
    """Add an edge to semantic graph."""
    from engram.models import SemanticEdge
    graph = _get_graph()
    edge = SemanticEdge(from_node=from_node, to_node=to_node, relation=relation)
    is_new = _run(graph.add_edge(edge))
    action = "Added" if is_new else "Updated"
    console.print(f"[green]{action}[/green] edge {edge.key}")


@remove_app.command("node")
def remove_node(key: str = typer.Argument(..., help="Node key (Type:Name)")):
    """Remove a node from semantic graph."""
    graph = _get_graph()
    removed = _run(graph.remove_node(key))
    if removed:
        console.print(f"[red]Removed[/red] node {key}")
    else:
        console.print(f"[yellow]Not found:[/yellow] {key}")


@remove_app.command("edge")
def remove_edge(key: str = typer.Argument(..., help="Edge key")):
    """Remove an edge from semantic graph."""
    graph = _get_graph()
    removed = _run(graph.remove_edge(key))
    if removed:
        console.print(f"[red]Removed[/red] edge {key}")
    else:
        console.print(f"[yellow]Not found:[/yellow] {key}")


@app.command()
def query(
    keyword: str = typer.Argument(None, help="Search keyword"),
    type: Optional[str] = typer.Option(None, "--type", "-t"),
    related_to: Optional[str] = typer.Option(None, "--related-to", "-r"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table|json"),
):
    """Query semantic graph."""
    graph = _get_graph()

    if related_to:
        results = _run(graph.get_related([related_to], depth=2))
        if format == "json":
            console.print_json(json.dumps(results, default=str))
        else:
            console.print(f"[bold]Related to: {related_to}[/bold]")
            _print_related(results)
        return

    if keyword:
        nodes = _run(graph.query(keyword, type=type))
    else:
        nodes = _run(graph.get_nodes(type=type))

    if format == "json":
        console.print_json(json.dumps([n.model_dump() for n in nodes], default=str))
    else:
        _print_nodes_table(nodes)


def _print_nodes_table(nodes):
    if not nodes:
        console.print("[dim]No nodes found.[/dim]")
        return
    table = Table(title="Semantic Nodes")
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Name")
    table.add_column("Attributes", style="dim")
    for n in nodes:
        attrs = ", ".join(f"{k}={v}" for k, v in n.attributes.items()) if n.attributes else ""
        table.add_row(n.key, n.type, n.name, attrs)
    console.print(table)


def _print_related(results):
    if not results:
        console.print("[dim]No related entities found.[/dim]")
        return
    for entity, data in results.items():
        console.print(f"\n[bold cyan]{entity}[/bold cyan]")
        if isinstance(data, dict):
            for key, value in data.items():
                console.print(f"  {key}: {value}")


# === Unified Commands ===


@app.command()
def think(question: str = typer.Argument(..., help="Question to reason about")):
    """Combined reasoning across both memories."""
    engine = _get_engine()
    answer = _run(engine.think(question))
    console.print(answer)


@app.command()
def ingest(
    file: Path = typer.Argument(..., help="Chat JSON file to ingest"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Dual ingest: extract entities + remember context."""
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    result = _run(_do_ingest(file, dry_run))
    console.print(
        f"[green]Ingested:[/green] {result.episodic_count} memories, "
        f"{result.semantic_nodes} nodes, {result.semantic_edges} edges"
    )


async def _do_ingest(file: Path, dry_run: bool = False) -> IngestResult:
    """Run dual ingest on a chat file."""
    with open(file) as f:
        data = json.load(f)

    messages = data if isinstance(data, list) else data.get("messages", [])
    if not messages:
        return IngestResult()

    extractor = _get_extractor()
    result = await extractor.extract_entities(messages)

    if dry_run:
        console.print("[bold]Dry run - extracted entities:[/bold]")
        for n in result.nodes:
            console.print(f"  [cyan]Node:[/cyan] {n.key}")
        for e in result.edges:
            console.print(f"  [green]Edge:[/green] {e.key}")
        return IngestResult(semantic_nodes=len(result.nodes), semantic_edges=len(result.edges))

    graph = _get_graph()
    episodic = _get_episodic()

    # Store entities in graph
    for node in result.nodes:
        await graph.add_node(node)
    for edge in result.edges:
        await graph.add_edge(edge)

    # Store messages in episodic memory, linked to entities
    entity_names = [n.name for n in result.nodes]
    for msg in messages:
        content = msg.get("content", "")
        if content:
            await episodic.remember(content, entities=entity_names)

    return IngestResult(
        episodic_count=len(messages),
        semantic_nodes=len(result.nodes),
        semantic_edges=len(result.edges),
    )


@app.command()
def status():
    """Show memory stats for both stores."""
    episodic = _get_episodic()
    graph = _get_graph()

    ep_stats = _run(episodic.stats())
    sem_stats = _run(graph.stats())

    console.print("[bold]Episodic Memory[/bold]")
    console.print(f"  Memories: {ep_stats.get('count', 0)}")

    console.print("[bold]Semantic Memory[/bold]")
    console.print(f"  Nodes: {sem_stats.get('node_count', 0)}")
    console.print(f"  Edges: {sem_stats.get('edge_count', 0)}")
    if "node_types" in sem_stats:
        for t, c in sem_stats["node_types"].items():
            console.print(f"    {t}: {c}")


@app.command()
def dump(format: str = typer.Option("json", "--format", "-f")):
    """Export all memory data."""
    episodic = _get_episodic()
    graph = _get_graph()

    data = {
        "episodic": _run(episodic.stats()),
        "semantic": _run(graph.dump()),
    }
    console.print_json(json.dumps(data, default=str))


# === Schema Commands ===


@schema_app.command("show")
def schema_show():
    """Show current schema."""
    from engram.schema.loader import load_schema, schema_to_prompt
    cfg = _get_config()
    schema = load_schema(cfg.semantic.schema_name)
    console.print(schema_to_prompt(schema))


@schema_app.command("init")
def schema_init(template: str = typer.Option("devops", "--template", "-t")):
    """Initialize schema from template."""
    from engram.schema.loader import load_schema
    load_schema(template)  # Validate it exists
    cfg = _get_config()
    cfg.semantic.schema_name = template
    save_config(cfg)
    console.print(f"[green]Schema set to:[/green] {template}")


@schema_app.command("validate")
def schema_validate(file: str = typer.Argument(...)):
    """Validate a schema file."""
    from engram.schema.loader import validate_schema
    errors = validate_schema(file)
    if errors:
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
    else:
        console.print("[green]Schema is valid.[/green]")


# === Config Commands ===


@config_app.command("show")
def config_show():
    """Show current configuration."""
    cfg = _get_config()
    console.print_json(cfg.model_dump_json(indent=2))


@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)):
    """Set a config value (dot notation: llm.model)."""
    global _config
    _config = set_config_value(key, value)
    console.print(f"[green]Set[/green] {key} = {value}")


@config_app.command("get")
def config_get(key: str = typer.Argument(...)):
    """Get a config value."""
    cfg = _get_config()
    val = get_config_value(cfg, key)
    console.print(f"{key} = {val}")


# === Watch & Serve Commands ===


@app.command()
def watch(
    daemon: bool = typer.Option(False, "--daemon", "-d"),
    stop: bool = typer.Option(False, "--stop"),
):
    """Watch inbox for chat files and auto-ingest."""
    from engram.capture.watcher import InboxWatcher, daemonize, is_daemon_running, stop_daemon

    if stop:
        if stop_daemon():
            console.print("[green]Watcher stopped.[/green]")
        else:
            console.print("[yellow]No watcher running.[/yellow]")
        return

    if is_daemon_running():
        console.print("[yellow]Watcher already running.[/yellow]")
        return

    cfg = _get_config()

    async def ingest_messages(messages):
        return await _do_ingest_messages(messages)

    watcher = InboxWatcher(cfg.capture.inbox, ingest_messages, cfg.capture.poll_interval)

    if daemon:
        daemonize()

    _run(watcher.start())


async def _do_ingest_messages(messages: list[dict]) -> IngestResult:
    """Ingest messages (called by watcher)."""
    extractor = _get_extractor()
    graph = _get_graph()
    episodic = _get_episodic()

    result = await extractor.extract_entities(messages)

    for node in result.nodes:
        await graph.add_node(node)
    for edge in result.edges:
        await graph.add_edge(edge)

    entity_names = [n.name for n in result.nodes]
    for msg in messages:
        content = msg.get("content", "")
        if content:
            await episodic.remember(content, entities=entity_names)

    return IngestResult(
        episodic_count=len(messages),
        semantic_nodes=len(result.nodes),
        semantic_edges=len(result.edges),
    )


@app.command()
def migrate(
    file: Path = typer.Argument(..., help="JSON file to import (agent-memory export)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without importing"),
):
    """Import data from old agent-memory/neural-memory JSON exports.

    Parses structured entity data (Server:, Technology:, Person:, Project:,
    Relationship:) and imports nodes/edges into semantic graph.
    Plain text entries go to episodic memory.
    """
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    with open(file) as f:
        data = json.load(f)

    messages = data if isinstance(data, list) else data.get("messages", [])
    if not messages:
        console.print("[yellow]No data to import.[/yellow]")
        return

    # Parse structured entity data
    nodes_to_add: list[tuple[str, str, dict]] = []  # (type, name, attributes)
    edges_to_add: list[tuple[str, str, str]] = []  # (from_key, to_key, relation)
    episodic_texts: list[str] = []

    entity_prefixes = {
        "Server": "Server", "Technology": "Technology", "Person": "Person",
        "Project": "Project", "Environment": "Environment", "Script": "Script",
        "Service": "Service",
    }

    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if not content:
            continue

        parsed = False
        # Parse "Relationship: X --rel--> Y"
        if content.startswith("Relationship:"):
            import re
            match = re.match(r"Relationship:\s*(.+?)\s*--(\w+)-->\s*(.+)", content)
            if match:
                from_name, relation, to_name = match.groups()
                edges_to_add.append((from_name.strip(), to_name.strip(), relation.strip()))
                parsed = True

        # Parse "Type: Name - {json_attrs}"
        if not parsed:
            for prefix, node_type in entity_prefixes.items():
                if content.startswith(f"{prefix}:"):
                    rest = content[len(prefix) + 1:].strip()
                    name = rest
                    attrs = {}
                    if " - " in rest:
                        name, json_part = rest.split(" - ", 1)
                        try:
                            attrs = json.loads(json_part)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    nodes_to_add.append((node_type, name.strip(), attrs))
                    parsed = True
                    break

        if not parsed:
            episodic_texts.append(content)

    console.print(f"[bold]Parsed:[/bold] {len(nodes_to_add)} nodes, {len(edges_to_add)} edges, {len(episodic_texts)} episodic")

    if dry_run:
        for ntype, name, attrs in nodes_to_add[:10]:
            console.print(f"  [cyan]Node:[/cyan] {ntype}:{name} {attrs if attrs else ''}")
        for fk, tk, rel in edges_to_add[:10]:
            console.print(f"  [green]Edge:[/green] {fk} --{rel}--> {tk}")
        if len(nodes_to_add) > 10:
            console.print(f"  ... and {len(nodes_to_add) - 10} more nodes")
        if len(edges_to_add) > 10:
            console.print(f"  ... and {len(edges_to_add) - 10} more edges")
        return

    # Import into semantic graph
    from engram.models import SemanticEdge, SemanticNode
    graph = _get_graph()
    added_nodes = 0
    added_edges = 0

    for ntype, name, attrs in nodes_to_add:
        node = SemanticNode(type=ntype, name=name, attributes=attrs)
        is_new = _run(graph.add_node(node))
        if is_new:
            added_nodes += 1

    # Build a name→key lookup for edge resolution
    all_nodes = _run(graph.get_nodes())
    name_to_key = {n.name: n.key for n in all_nodes}

    for from_name, to_name, relation in edges_to_add:
        from_key = name_to_key.get(from_name, f"Project:{from_name}")
        to_key = name_to_key.get(to_name, f"Technology:{to_name}")
        edge = SemanticEdge(from_node=from_key, to_node=to_name, relation=relation)
        is_new = _run(graph.add_edge(edge))
        if is_new:
            added_edges += 1

    # Import episodic texts
    episodic_count = 0
    if episodic_texts:
        store = _get_episodic()
        for text in episodic_texts:
            _run(store.remember(text, memory_type=MemoryType.CONTEXT, priority=4))
            episodic_count += 1

    console.print(
        f"[green]Imported:[/green] {added_nodes} new nodes, "
        f"{added_edges} new edges, {episodic_count} episodic memories"
    )


@app.command()
def serve(
    port: Optional[int] = typer.Option(None, "--port", "-p"),
    host: Optional[str] = typer.Option(None, "--host"),
):
    """Start HTTP webhook server."""
    from engram.capture.server import run_server

    cfg = _get_config()
    if port:
        cfg.serve.port = port
    if host:
        cfg.serve.host = host

    run_server(_get_episodic(), _get_graph(), _get_engine(), cfg, _do_ingest_messages)


if __name__ == "__main__":
    app()
