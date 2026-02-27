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


def _load_env_file() -> None:
    """Load .env file from project root or ~/.engram/.env into os.environ.

    Ensures daemon children inherit API keys (GEMINI_API_KEY etc.) after fork.
    Existing env vars are NOT overwritten.
    """
    import os
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
                    if key and key not in os.environ:
                        os.environ[key] = val


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
            guard_enabled=cfg.ingestion.poisoning_guard,
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
            recall_config=cfg.recall_pipeline,
        )

    def _get_extractor():
        from engram.capture.extractor import EntityExtractor
        from engram.schema.loader import load_schema
        cfg = get_config()
        schema = load_schema(cfg.semantic.schema_name)
        return EntityExtractor(model=cfg.llm.model, schema=schema)

    @app.command()
    def health(
        quick: bool = typer.Option(False, "--quick", "-q", help="Skip LLM/embedding checks"),
        features: bool = typer.Option(False, "--features", "-f", help="Show full feature registry (all ~350 features)"),
        all_checks: bool = typer.Option(False, "--all", "-a", help="Show runtime health + full feature registry"),
        category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter registry by category name"),
        grep: Optional[str] = typer.Option(None, "--grep", "-g", help="Filter registry by text search"),
    ):
        """Full system health check across all subsystems."""
        from engram.health import (
            full_health_check, check_api_keys, check_disk, check_fts5, check_watcher,
            check_feature_flags, ComponentHealth,
        )
        from engram.health.feature_registry import build_full_registry
        from rich.table import Table

        cfg = get_config()

        # --- Full feature registry renderer ---
        def _render_registry(cat_filter: Optional[str] = None, grep_filter: Optional[str] = None):
            registry = build_full_registry(cfg)

            # Apply filters
            if cat_filter:
                lo = cat_filter.lower()
                registry = [e for e in registry
                            if lo in e.category.lower() or lo in e.subcategory.lower()]
            if grep_filter:
                lo = grep_filter.lower()
                registry = [e for e in registry
                            if lo in e.name.lower() or lo in e.subcategory.lower()
                            or lo in e.config_path.lower() or lo in e.env_var.lower()
                            or lo in e.category.lower() or lo in e.status.lower()]

            # Count boolean flags
            bool_flags = [e for e in registry if e.category == "Config Boolean Flags"]
            flag_on = sum(1 for e in bool_flags if e.status == "enabled")
            flag_off = sum(1 for e in bool_flags if e.status == "disabled")

            console.print(
                f"\n[bold]Feature Registry:[/bold] {len(registry)} features"
                + (f" ({len(bool_flags)} flags: {flag_on} on / {flag_off} off)" if bool_flags else "")
            )
            if cat_filter:
                console.print(f"[dim]  filter: category={cat_filter!r}[/dim]")
            if grep_filter:
                console.print(f"[dim]  filter: grep={grep_filter!r}[/dim]")
            console.print()

            # Group by category
            from collections import defaultdict
            grouped: dict[str, list] = defaultdict(list)
            for entry in registry:
                grouped[entry.category].append(entry)

            status_styles = {"enabled": "[green]enabled[/green]", "disabled": "[red]disabled[/red]",
                             "always-on": "[cyan]always-on[/cyan]"}

            for cat, entries in grouped.items():
                console.print(f"[bold]--- {cat} ({len(entries)}) ---[/bold]")
                tbl = Table(show_header=True, show_edge=False, pad_edge=False)
                tbl.add_column("Subcategory", style="dim", width=16)
                tbl.add_column("Feature / Parameter", width=40)
                tbl.add_column("Status / Value", width=20)
                tbl.add_column("Config / Env Var")
                for e in entries:
                    status_cell = status_styles.get(e.status, e.status)
                    cfg_col = e.config_path or ""
                    if e.env_var:
                        cfg_col += f"\n[dim]{e.env_var}[/dim]" if cfg_col else f"[dim]{e.env_var}[/dim]"
                    tbl.add_row(e.subcategory, e.name, status_cell, cfg_col)
                console.print(tbl)
                console.print()

        # --features: show full registry only, skip runtime checks
        if features and not all_checks:
            _render_registry(category, grep)
            return

        # --- Legacy feature-flags only render (used by --all) ---
        def _render_flags():
            flags = check_feature_flags(cfg)
            enabled_count = sum(1 for f in flags if f.enabled)
            console.print(f"\n[bold]Feature Flags:[/bold] {enabled_count} enabled / {len(flags)} total\n")
            tbl = Table(show_header=True)
            tbl.add_column("Category", style="dim", width=12)
            tbl.add_column("Feature", width=24)
            tbl.add_column("Status", width=10)
            tbl.add_column("Env Var")
            for fl in flags:
                status_cell = "[green]✓ on[/green]" if fl.enabled else "[red]✗ off[/red]"
                tbl.add_row(fl.category, fl.name, status_cell, fl.env_var)
            console.print(tbl)

        # Runtime health check
        episodic = None
        graph = None
        try:
            episodic = _get_episodic()
        except Exception as e:
            console.print(f"[yellow]Could not init episodic store:[/yellow] {e}")
        try:
            graph = _get_graph()
        except Exception as e:
            console.print(f"[yellow]Could not init semantic graph:[/yellow] {e}")

        if quick:
            async def _quick():
                results = {}
                api = check_api_keys()
                results[api.name] = api
                for coro in [check_disk(), check_fts5(), check_watcher()]:
                    r = await coro
                    results[r.name] = r
                if episodic:
                    from engram.health import check_chromadb
                    r = await check_chromadb(episodic)
                    results[r.name] = r
                if graph:
                    from engram.health import check_semantic
                    r = await check_semantic(graph)
                    results[r.name] = r
                return results

            comps = run_async(_quick())
            overall = "healthy"
            for c in comps.values():
                if c.status == "unhealthy":
                    overall = "unhealthy"
                    break
                if c.status == "degraded" and overall == "healthy":
                    overall = "degraded"
        else:
            result = run_async(full_health_check(episodic, graph, cfg.llm.model, config=cfg))
            overall = result["status"]
            comps = {}
            for name, data in result["components"].items():
                comps[name] = ComponentHealth(
                    name=name,
                    status=data.get("status", "unknown"),
                    latency_ms=data.get("latency_ms", 0),
                    details={k: v for k, v in data.items() if k not in ("status", "latency_ms", "error")},
                    error=data.get("error", ""),
                )

        # Render runtime health table
        style_map = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}
        overall_style = style_map.get(overall, "white")
        console.print(f"\n[bold]Engram Health:[/bold] [{overall_style}]{overall.upper()}[/{overall_style}]\n")

        table = Table(show_header=True)
        table.add_column("Component", style="cyan", width=14)
        table.add_column("Status", width=10)
        table.add_column("Latency", justify="right", width=8)
        table.add_column("Details")

        for comp in comps.values():
            s = style_map.get(comp.status, "white")
            latency = f"{comp.latency_ms:.0f}ms" if comp.latency_ms else "—"
            details_parts = []
            if comp.details:
                details_parts.append(", ".join(f"{k}={v}" for k, v in comp.details.items()))
            if comp.error:
                details_parts.append(f"[red]{comp.error}[/red]")
            table.add_row(comp.name, f"[{s}]{comp.status}[/{s}]", latency, " | ".join(details_parts) or "—")

        console.print(table)

        # Feature summary line (default) or full registry (--all)
        flags = check_feature_flags(cfg)
        enabled_count = sum(1 for f in flags if f.enabled)
        if all_checks:
            _render_registry(category, grep)
        else:
            console.print(f"\n[dim]Features: {enabled_count}/{len(flags)} flags enabled — run [bold]engram health --features[/bold] for full registry[/dim]")

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
        # Load .env so daemon child inherits API keys after fork
        _load_env_file()
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
                )
            scheduler = create_default_scheduler(_get_episodic(), consolidation_engine)
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

    @app.command("resource-status")
    def resource_status():
        """Show resource tier and LLM availability status."""
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
            )

        scheduler = create_default_scheduler(_get_episodic(), consolidation_engine)
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
        """Start HTTP webhook server."""
        from engram.capture.server import run_server

        cfg = get_config()
        if port:
            cfg.serve.port = port
        if host:
            cfg.serve.host = host

        from engram.telemetry import setup_telemetry
        setup_telemetry(cfg)

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
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if content:
            ctx = messages[max(0, i - 2): min(len(messages), i + 3)]
            per_content = extractor.filter_entities_for_content(content, entity_names, context_messages=ctx)
            await episodic.remember(content, entities=per_content)

    return IngestResult(
        episodic_count=len(messages),
        semantic_nodes=len(result.nodes),
        semantic_edges=len(result.edges),
    )


async def _do_ingest_messages(messages: list[dict], get_extractor, get_graph, get_episodic) -> IngestResult:
    """Ingest messages (called by watcher/server).

    Episodic storage runs first (no LLM needed).
    Entity extraction runs separately — failures don't block memory capture.
    """
    episodic = get_episodic()

    # Step 1: Store raw messages into episodic memory immediately (no LLM)
    episodic_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            await episodic.remember(content)
            episodic_count += 1

    # Step 2: Entity extraction + semantic graph (best-effort, LLM-dependent)
    semantic_nodes = 0
    semantic_edges = 0
    try:
        extractor = get_extractor()
        graph = get_graph()
        result = await extractor.extract_entities(messages)
        for node in result.nodes:
            await graph.add_node(node)
        for edge in result.edges:
            await graph.add_edge(edge)
        semantic_nodes = len(result.nodes)
        semantic_edges = len(result.edges)

        # Enrich episodic memories with entity tags (backfill)
        entity_names = [n.name for n in result.nodes]
        if entity_names:
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if content:
                    ctx = messages[max(0, i - 2): min(len(messages), i + 3)]
                    per_content = extractor.filter_entities_for_content(
                        content, entity_names, context_messages=ctx,
                    )
                    if per_content:
                        await episodic.remember(content, entities=per_content)
    except Exception as e:
        logger.warning("Entity extraction skipped (messages already stored): %s", e)

    return IngestResult(
        episodic_count=episodic_count,
        semantic_nodes=semantic_nodes,
        semantic_edges=semantic_edges,
    )
