"""CLI commands for system operations: status, dump, watch, serve, ingest, cleanup, summarize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from engram.models import IngestResult
from engram.utils import run_async

console = Console()


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register system commands on the main Typer app."""

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    def _get_episodic():
        from engram.episodic.store import EpisodicStore
        cfg = get_config()
        return EpisodicStore(
            cfg.episodic, cfg.embedding,
            namespace=_resolve_namespace(),
            on_remember_hook=cfg.hooks.on_remember,
        )

    def _get_graph():
        from engram.semantic.graph import SemanticGraph
        cfg = get_config()
        return SemanticGraph(cfg.semantic)

    def _get_engine():
        from engram.reasoning.engine import ReasoningEngine
        cfg = get_config()
        return ReasoningEngine(
            _get_episodic(), _get_graph(),
            model=cfg.llm.model,
            on_think_hook=cfg.hooks.on_think,
        )

    def _get_extractor():
        from engram.capture.extractor import EntityExtractor
        from engram.schema.loader import load_schema
        cfg = get_config()
        schema = load_schema(cfg.semantic.schema_name)
        return EntityExtractor(model=cfg.llm.model, schema=schema)

    @app.command()
    def status():
        """Show memory stats for both stores."""
        episodic = _get_episodic()
        graph = _get_graph()
        ep_stats = run_async(episodic.stats())
        sem_stats = run_async(graph.stats())
        console.print("[bold]Episodic Memory[/bold]")
        console.print(f"  Memories: {ep_stats.get('count', 0)}")
        if ep_stats.get("namespace"):
            console.print(f"  Namespace: {ep_stats['namespace']}")
        console.print("[bold]Semantic Memory[/bold]")
        console.print(f"  Nodes: {sem_stats.get('node_count', 0)}")
        console.print(f"  Edges: {sem_stats.get('edge_count', 0)}")
        if "node_types" in sem_stats:
            for t, c in sem_stats["node_types"].items():
                console.print(f"    {t}: {c}")

    @app.command()
    def dump(format: str = typer.Option("json", "--format", "-f")):
        """Export all memory data."""
        data = {
            "episodic": run_async(_get_episodic().stats()),
            "semantic": run_async(_get_graph().dump()),
        }
        console.print_json(json.dumps(data, default=str))

    @app.command()
    def cleanup():
        """Delete all expired memories from episodic store."""
        store = _get_episodic()
        deleted = run_async(store.cleanup_expired())
        if deleted == 0:
            console.print("[dim]No expired memories found.[/dim]")
        else:
            console.print(f"[green]Cleaned up[/green] {deleted} expired {'memory' if deleted == 1 else 'memories'}.")

    @app.command()
    def summarize(
        count: int = typer.Option(20, "--count", "-n", help="Number of recent memories to summarize"),
        save: bool = typer.Option(False, "--save", help="Store summary as a new memory"),
    ):
        """Summarize recent N memories into key insights using LLM."""
        engine = _get_engine()
        summary = run_async(engine.summarize(n=count, save=save))
        console.print("[bold]Summary[/bold]")
        console.print(summary)

    @app.command()
    def ingest(
        file: Path = typer.Argument(..., help="Chat JSON file to ingest"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ):
        """Dual ingest: extract entities + remember context."""
        if not file.exists():
            console.print(f"[red]File not found:[/red] {file}")
            raise typer.Exit(1)
        result = run_async(_do_ingest(file, dry_run, _get_extractor, _get_graph, _get_episodic))
        console.print(
            f"[green]Ingested:[/green] {result.episodic_count} memories, "
            f"{result.semantic_nodes} nodes, {result.semantic_edges} edges"
        )

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

        cfg = get_config()

        async def ingest_messages(messages):
            return await _do_ingest_messages(messages, _get_extractor, _get_graph, _get_episodic)

        watcher = InboxWatcher(cfg.capture.inbox, ingest_messages, cfg.capture.poll_interval)
        if daemon:
            daemonize()
        run_async(watcher.start())

    @app.command()
    def serve(
        port: Optional[int] = typer.Option(None, "--port", "-p"),
        host: Optional[str] = typer.Option(None, "--host"),
    ):
        """Start HTTP webhook server."""
        from engram.capture.server import run_server

        cfg = get_config()
        if port:
            cfg.serve.port = port
        if host:
            cfg.serve.host = host

        async def ingest_messages(messages):
            return await _do_ingest_messages(messages, _get_extractor, _get_graph, _get_episodic)

        run_server(_get_episodic(), _get_graph(), _get_engine(), cfg, ingest_messages)


async def _do_ingest(file: Path, dry_run: bool, get_extractor, get_graph, get_episodic) -> IngestResult:
    """Run dual ingest on a chat file."""
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


async def _do_ingest_messages(messages: list[dict], get_extractor, get_graph, get_episodic) -> IngestResult:
    """Ingest messages (called by watcher/server)."""
    extractor = get_extractor()
    graph = get_graph()
    episodic = get_episodic()
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
