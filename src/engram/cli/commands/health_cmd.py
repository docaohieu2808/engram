"""Health command implementation extracted from cli/system.py.

Registers the ``health`` Typer command including the feature registry renderer,
the legacy flags renderer, and the quick/full runtime check paths.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

console = Console()


def register_health(app: typer.Typer, get_config, get_episodic, get_graph) -> None:
    """Attach the ``health`` command to *app*."""

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
        from engram.utils import run_async
        from rich.table import Table

        cfg = get_config()

        def _render_registry(cat_filter: Optional[str] = None, grep_filter: Optional[str] = None):
            """Render the full feature registry table, optionally filtered."""
            registry = build_full_registry(cfg)

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

            from collections import defaultdict
            grouped: dict[str, list] = defaultdict(list)
            for entry in registry:
                grouped[entry.category].append(entry)

            status_styles = {
                "enabled": "[green]enabled[/green]",
                "disabled": "[red]disabled[/red]",
                "always-on": "[cyan]always-on[/cyan]",
            }

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

        def _render_flags():
            """Render legacy feature flags summary table (used by --all)."""
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
            episodic = get_episodic()
        except Exception as e:
            console.print(f"[yellow]Could not init episodic store:[/yellow] {e}")
        try:
            graph = get_graph()
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
            console.print(
                f"\n[dim]Features: {enabled_count}/{len(flags)} flags enabled"
                f" — run [bold]engram health --features[/bold] for full registry[/dim]"
            )
