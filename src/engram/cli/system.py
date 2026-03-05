"""CLI commands for system operations: status, dump, watch, serve, ingest, cleanup, summarize."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from engram.utils import run_async
from engram.cli.ingest import do_ingest, do_ingest_messages
from engram.cli.factories import make_factories

logger = logging.getLogger("engram")
console = Console()


def _load_env_file() -> None:
    """Load .env file from project root or ~/.engram/.env into os.environ.

    Ensures daemon children inherit API keys (GEMINI_API_KEY etc.) after fork.
    API keys are always overwritten (they rotate); other vars only set if missing.
    """
    import os
    # Keys that should always be refreshed from .env (they expire/rotate)
    _FORCE_OVERRIDE = {"GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACK", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    candidates = [
        Path.home() / ".engram" / ".env",
        Path(__file__).resolve().parents[3] / ".env",  # project root
    ]
    for env_path in candidates:
        if env_path.is_file():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip("'\"")
                    if key and (key in _FORCE_OVERRIDE or key not in os.environ):
                        os.environ[key] = val


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register system commands on the main Typer app."""

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    # Build lazy store factories
    _factories = make_factories(get_config, get_namespace)
    _get_episodic = _factories["get_episodic"]
    _get_graph = _factories["get_graph"]
    _get_engine = _factories["get_engine"]
    _get_extractor = _factories["get_extractor"]

    # Register health command via dedicated module
    from engram.cli.commands.health_cmd import register_health
    register_health(app, get_config, _get_episodic, _get_graph)

    @app.command()
    def status():
        """Show memory stats for both stores."""
        # If server is running, query via HTTP to avoid embedded Qdrant lock conflict
        cfg = get_config()
        if cfg.episodic.mode == "embedded":
            from engram.cli.daemon_cmd import _is_running
            running, _ = _is_running()
            if running:
                try:
                    import httpx
                    resp = httpx.get(f"http://{cfg.serve.host}:{cfg.serve.port}/api/v1/status", timeout=5)
                    data = resp.json()
                    console.print("[bold]Episodic Memory[/bold]")
                    console.print(f"  Memories: {data.get('episodic', {}).get('count', 0)}")
                    ns = data.get('episodic', {}).get('namespace')
                    if ns:
                        console.print(f"  Namespace: {ns}")
                    console.print("[bold]Semantic Memory[/bold]")
                    console.print(f"  Nodes: {data.get('semantic', {}).get('node_count', 0)}")
                    console.print(f"  Edges: {data.get('semantic', {}).get('edge_count', 0)}")
                    for t, c in data.get('semantic', {}).get('node_types', {}).items():
                        console.print(f"    {t}: {c}")
                    return
                except Exception:
                    pass  # Fallback to direct access

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
    def dump(format: str = typer.Option("table", "--format", "-f", help="Output format: table|json")):
        """Export all memory data (episodic + semantic)."""
        from rich.table import Table

        episodic = _get_episodic()
        graph = _get_graph()

        if format == "json":
            memories = run_async(episodic.get_recent(n=500))
            data = {
                "episodic": [
                    {
                        "id": m.id, "content": m.content,
                        "memory_type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                        "priority": m.priority, "tags": m.tags, "entities": m.entities,
                        "timestamp": str(m.timestamp), "confidence": getattr(m, "confidence", 1.0),
                        "source": getattr(m, "source", ""),
                    }
                    for m in memories
                ],
                "semantic": run_async(graph.dump()),
            }
            console.print_json(json.dumps(data, default=str))
            return

        # Rich table format (default fallback)
        memories = run_async(episodic.get_recent(n=500))
        sem = run_async(graph.dump())

        # Episodic table
        ep_table = Table(title=f"Episodic Memories ({len(memories)})")
        ep_table.add_column("#", style="dim", width=3)
        ep_table.add_column("Time", style="cyan", width=16)
        ep_table.add_column("Type", style="green", width=12)
        ep_table.add_column("P", justify="right", width=2)
        ep_table.add_column("Content", max_width=70)
        ep_table.add_column("Tags/Entities", style="dim", max_width=30)
        for i, m in enumerate(memories, 1):
            ts = m.timestamp.strftime("%m-%d %H:%M") if hasattr(m.timestamp, "strftime") else str(m.timestamp)[:16]
            mt = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
            extra = []
            if m.tags:
                extra.append(f"t:{','.join(m.tags[:3])}")
            if m.entities:
                extra.append(f"e:{','.join(m.entities[:3])}")
            ep_table.add_row(str(i), ts, mt, str(m.priority), m.content[:70], " ".join(extra))
        console.print(ep_table)

        # Nodes table
        nodes = sem.get("nodes", [])
        n_table = Table(title=f"Semantic Nodes ({len(nodes)})")
        n_table.add_column("#", style="dim", width=3)
        n_table.add_column("Type", style="green", width=14)
        n_table.add_column("Name", style="cyan")
        n_table.add_column("Attributes", style="dim")
        for i, n in enumerate(nodes, 1):
            attrs = ", ".join(f"{k}={v}" for k, v in n.get("attributes", {}).items()) if n.get("attributes") else ""
            # Support both direct model dump (type/name) and vis.js format (group/label/id)
            ntype = n.get("type", "") or n.get("group", "")
            nname = n.get("name", "") or n.get("label", "")
            n_table.add_row(str(i), ntype, nname, attrs)
        console.print(n_table)

        # Edges table
        edges = sem.get("edges", [])
        e_table = Table(title=f"Semantic Edges ({len(edges)})")
        e_table.add_column("#", style="dim", width=3)
        e_table.add_column("From", style="yellow")
        e_table.add_column("Relation", style="green")
        e_table.add_column("To", style="cyan")
        e_table.add_column("Weight", style="dim", width=6)
        for i, e in enumerate(edges, 1):
            # Support both direct model dump and vis.js format
            efrom = e.get("from_node", "") or e.get("from", "")
            erel = e.get("relation", "") or e.get("label", "")
            eto = e.get("to_node", "") or e.get("to", "")
            e_table.add_row(str(i), efrom, erel, eto, str(e.get("weight", 1.0)))
        console.print(e_table)

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
        result = run_async(do_ingest(file, dry_run, _get_extractor, _get_graph, _get_episodic))
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
        # Load .env so daemon child inherits API keys after fork
        _load_env_file()
        # Resolve LLM api_key (auto-refresh Anthropic OAuth tokens)
        from engram.config import apply_llm_api_key
        apply_llm_api_key(get_config())
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

        # If serve.proxy_to is set, proxy ingest via HTTP instead of local Qdrant
        proxy_url = getattr(cfg.serve, 'proxy_to', None) or os.environ.get('ENGRAM_PROXY_URL')
        if proxy_url:
            from engram.capture.http_ingest import http_ingest_messages
            async def ingest_messages(messages, source: str = ""):
                return await http_ingest_messages(messages, proxy_url, source=source)
        else:
            async def ingest_messages(messages, source: str = ""):
                return await do_ingest_messages(messages, _get_extractor, _get_graph, _get_episodic, source=source)

        if daemon:
            daemonize()

        async def _run_watchers():
            tasks = []
            # Inbox watcher (always)
            inbox = InboxWatcher(cfg.capture.inbox, ingest_messages, cfg.capture.poll_interval)
            tasks.append(asyncio.create_task(inbox.start()))
            console.print("[dim]Inbox watcher started[/dim]")

            # Session watchers (OpenClaw + Claude Code)
            from engram.capture.session_watcher import SessionWatcher

            if cfg.capture.openclaw.enabled:
                oc = SessionWatcher(
                    cfg.capture.openclaw.sessions_dir, ingest_messages,
                    label="OpenClaw", recursive=False,
                )
                tasks.append(asyncio.create_task(oc.start()))
                console.print("[dim]OpenClaw watcher started[/dim]")

            if cfg.capture.claude_code.enabled:
                cc = SessionWatcher(
                    cfg.capture.claude_code.sessions_dir, ingest_messages,
                    label="ClaudeCode", recursive=True,
                )
                tasks.append(asyncio.create_task(cc.start()))
                console.print("[dim]Claude Code watcher started[/dim]")

            # Memory scheduler (consolidation + cleanup + decay report)
            from engram.scheduler import create_default_scheduler
            consolidation_engine = None
            if cfg.consolidation.enabled:
                from engram.consolidation.engine import ConsolidationEngine
                consolidation_engine = ConsolidationEngine(
                    _get_episodic(), model=cfg.llm.model, config=cfg.consolidation,
                    disable_thinking=cfg.llm.disable_thinking,
                )
            scheduler = create_default_scheduler(_get_episodic(), consolidation_engine, config=cfg.scheduler)
            scheduler.start()
            console.print("[dim]Memory scheduler started[/dim]")

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
            disable_thinking=cfg.llm.disable_thinking,
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

    @app.command("queue-status")
    def queue_status():
        """Show embedding queue status (pending retries after embedding API failure)."""
        from engram.episodic.embedding_queue import get_embedding_queue
        q = get_embedding_queue()
        stats = q.queue_status()
        pending = stats["pending"]
        done = stats["done"]
        failed = stats["failed"]
        style = "green" if pending == 0 else "yellow"
        console.print(f"[bold]Embedding Queue:[/bold] [{style}]{pending} pending[/{style}]")
        console.print(f"  Processed (done): {done}")
        console.print(f"  Permanently failed: {failed}")
        if stats["oldest_pending"]:
            console.print(f"  Oldest pending: {stats['oldest_pending']}")
        if stats["last_error"]:
            console.print(f"  Last error: [red]{stats['last_error']}[/red]")
        if pending == 0:
            console.print("[dim]Queue empty — all embeddings are current.[/dim]")

    @app.command("resource-status")
    def resource_status():
        """Show resource tier, LLM availability, and embedding health."""
        from engram.resource_tier import get_resource_monitor
        monitor = get_resource_monitor()
        status = monitor.status()
        tier = status["tier"]
        style = {"full": "green", "standard": "yellow", "basic": "red", "readonly": "red bold"}.get(tier, "white")
        console.print(f"[bold]Resource Tier:[/bold] [{style}]{tier}[/{style}]")
        console.print(f"  Recent failures: {status['recent_failures']}/{status['failure_threshold']}")
        console.print(f"  Since last success: {status['seconds_since_last_success']}s")
        if status["forced"]:
            console.print(f"  [yellow]Forced tier: {status['forced']}[/yellow]")
        # Show embedding queue health
        try:
            from engram.episodic.embedding_queue import get_embedding_queue
            q_stats = get_embedding_queue().queue_status()
            pending = q_stats["pending"]
            embed_style = "green" if pending == 0 else ("yellow" if pending < 100 else "red")
            console.print(f"[bold]Embedding Health:[/bold] [{embed_style}]{pending} queued[/{embed_style}]"
                          f" ({q_stats['failed']} failed permanently)")
        except Exception:
            pass

    @app.command("constitution-status")
    def constitution_status():
        """Show constitution status and hash."""
        from engram.constitution import load_constitution, compute_constitution_hash, get_constitution_path
        content = load_constitution()
        hash_val = compute_constitution_hash(content)
        path = get_constitution_path()
        console.print("[bold]Data Constitution[/bold]")
        console.print(f"  Path: {path}")
        console.print(f"  Exists: {path.exists()}")
        console.print(f"  Hash: {hash_val}")
        console.print("  Laws: 3 (namespace isolation, no fabrication, audit rights)")

    @app.command("scheduler-status")
    def scheduler_status():
        """Show memory scheduler task status."""
        from engram.scheduler import create_default_scheduler

        cfg = get_config()
        consolidation_engine = None
        if cfg.consolidation.enabled:
            from engram.consolidation.engine import ConsolidationEngine
            consolidation_engine = ConsolidationEngine(
                _get_episodic(), model=cfg.llm.model, config=cfg.consolidation,
                disable_thinking=cfg.llm.disable_thinking,
            )

        scheduler = create_default_scheduler(_get_episodic(), consolidation_engine, config=cfg.scheduler)
        tasks = scheduler.status()
        if not tasks:
            console.print("[dim]No scheduled tasks.[/dim]")
            return

        from rich.table import Table
        table = Table(title="Memory Scheduler Tasks")
        table.add_column("Task", style="cyan")
        table.add_column("Interval", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("LLM?", justify="center")
        table.add_column("Last Error")

        for t in tasks:
            hours = t["interval_seconds"] / 3600
            interval = f"{hours:.0f}h" if hours >= 1 else f"{t['interval_seconds']}s"
            table.add_row(
                t["name"], interval, str(t["run_count"]),
                "yes" if t["requires_llm"] else "no",
                t["last_error"] or "[dim]none[/dim]",
            )
        console.print(table)

    @app.command()
    def graph(
        port: int = typer.Option(8100, "--port", "-p", help="Server port"),
    ):
        """Open graph visualization in browser."""
        import webbrowser
        url = f"http://localhost:{port}/graph"
        webbrowser.open(url)
        console.print(f"Opening graph UI at [cyan]{url}[/cyan]")
        console.print("[dim]Make sure 'engram serve' is running.[/dim]")

    @app.command()
    def serve(
        port: Optional[int] = typer.Option(None, "--port", "-p"),
        host: Optional[str] = typer.Option(None, "--host"),
    ):
        """Start HTTP webhook server (foreground). Use 'engram start' for background."""
        # Stop any existing background daemon to release embedded Qdrant lock
        from engram.cli.daemon_cmd import _is_running, _PID_FILE
        from engram.capture.watcher import is_daemon_running, stop_daemon
        import os, signal as _signal

        # Stop serve daemon
        running, pid = _is_running()
        if running:
            console.print(f"[yellow]Stopping background server (PID={pid})...[/yellow]")
            try:
                os.kill(pid, _signal.SIGTERM)
                _PID_FILE.unlink(missing_ok=True)
                import time; time.sleep(1)
            except OSError:
                _PID_FILE.unlink(missing_ok=True)

        # Stop watcher daemon (also holds embedded Qdrant lock)
        if is_daemon_running():
            console.print("[yellow]Stopping background watcher...[/yellow]")
            stop_daemon()
            import time; time.sleep(1)

        console.print("[dim]Tip: Use 'engram start' to run in background instead.[/dim]")

        _load_env_file()
        from engram.capture.server import run_server
        from engram.config import apply_llm_api_key

        cfg = get_config()
        apply_llm_api_key(cfg)
        if port:
            cfg.serve.port = port
        if host:
            cfg.serve.host = host

        from engram.telemetry import setup_telemetry
        setup_telemetry(cfg)

        async def ingest_messages(messages, source: str = ""):
            return await do_ingest_messages(messages, _get_extractor, _get_graph, _get_episodic, source=source)

        run_server(_get_episodic(), _get_graph(), _get_engine(), cfg, ingest_messages)
