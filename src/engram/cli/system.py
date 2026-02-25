"""CLI commands for system operations: status, dump, watch, serve, ingest, cleanup, summarize."""

from __future__ import annotations

import asyncio
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
        from engram.semantic import create_graph
        cfg = get_config()
        return create_graph(cfg.semantic)

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
        """Watch inbox for chat files and auto-ingest. Also watches OpenClaw sessions if enabled."""
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

        if daemon:
            daemonize()

        async def _run_watchers():
            tasks = []
            # Inbox watcher (always)
            inbox = InboxWatcher(cfg.capture.inbox, ingest_messages, cfg.capture.poll_interval)
            tasks.append(asyncio.create_task(inbox.start()))
            console.print("[dim]Inbox watcher started[/dim]")

            # OpenClaw watcher (if enabled)
            if cfg.capture.openclaw.enabled:
                from engram.capture import openclaw_watcher as ocw
                oc = ocw.OpenClawWatcher(cfg.capture.openclaw.sessions_dir, ingest_messages)
                tasks.append(asyncio.create_task(oc.start()))
                console.print("[dim]OpenClaw watcher started[/dim]")

            await asyncio.gather(*tasks)

        run_async(_run_watchers())

    @app.command()
    def decay(
        limit: int = typer.Option(20, "--limit", "-l"),
    ):
        """Show Ebbinghaus decay report for recent memories."""
        import math
        from datetime import datetime, timezone
        from rich.table import Table

        episodic = _get_episodic()
        memories = run_async(episodic.get_recent(n=limit))
        now = datetime.now(timezone.utc)

        table = Table(title="Memory Decay Report")
        table.add_column("Age", style="dim", width=8)
        table.add_column("Access", justify="right", width=6)
        table.add_column("Retention", justify="right", width=10)
        table.add_column("Content", max_width=60)

        for mem in memories:
            ts = mem.timestamp if mem.timestamp.tzinfo else mem.timestamp.replace(tzinfo=timezone.utc)
            days = (now - ts).total_seconds() / 86400
            retention = math.exp(-mem.decay_rate * days / (1 + 0.1 * mem.access_count))
            age_str = f"{days:.1f}d" if days >= 1 else f"{days * 24:.1f}h"
            ret_pct = f"{retention * 100:.0f}%"
            style = "green" if retention > 0.7 else ("yellow" if retention > 0.3 else "red")
            table.add_row(age_str, str(mem.access_count), f"[{style}]{ret_pct}[/{style}]",
                          mem.content[:60])

        console.print(table)

    @app.command()
    def consolidate(
        limit: int = typer.Option(50, "--limit", "-l"),
    ):
        """Consolidate related memories into summaries using LLM."""
        from engram.consolidation.engine import ConsolidationEngine

        cfg = get_config()
        engine = ConsolidationEngine(
            _get_episodic(), model=cfg.llm.model, config=cfg.consolidation,
        )
        new_ids = run_async(engine.consolidate(limit=limit))
        if not new_ids:
            console.print("[dim]No clusters found to consolidate.[/dim]")
        else:
            console.print(f"[green]Consolidated into {len(new_ids)} summary memories.[/green]")

    @app.command("session-start")
    def session_start():
        """Start a new memory session."""
        from engram.session.store import SessionStore
        cfg = get_config()
        store = SessionStore(cfg.session.sessions_dir)
        s = store.start(namespace=_resolve_namespace() or "default")
        console.print(f"[green]Session started[/green] (id={s.id[:8]})")

    @app.command("session-end")
    def session_end():
        """End the current session."""
        from engram.session.store import SessionStore
        cfg = get_config()
        store = SessionStore(cfg.session.sessions_dir)
        s = store.end()
        if s:
            console.print(f"[green]Session ended[/green] (id={s.id[:8]})")
        else:
            console.print("[yellow]No active session.[/yellow]")

    @app.command()
    def tui():
        """Launch the Engram Terminal UI for browsing memories interactively."""
        try:
            from engram.tui.app import EngramTUI
        except ImportError:
            console.print("[red]TUI requires textual. Install with: pip install engram[tui][/red]")
            raise typer.Exit(1)

        from engram.session.store import SessionStore
        cfg = get_config()
        episodic = _get_episodic()
        session_store = SessionStore(cfg.session.sessions_dir)
        tui_app = EngramTUI(episodic, session_store)
        tui_app.run()

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
