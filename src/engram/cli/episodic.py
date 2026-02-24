"""CLI commands for episodic memory: remember and recall."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import typer
from rich.console import Console

from engram.models import MemoryType
from engram.utils import run_async

console = Console()


def _parse_duration(duration_str: str) -> datetime:
    """Parse duration string like '7d', '24h', '30m' → future datetime.

    Supported units: d (days), h (hours), m (minutes), s (seconds).
    """
    pattern = re.compile(r"^(\d+)([dhms])$")
    match = pattern.match(duration_str.strip().lower())
    if not match:
        raise typer.BadParameter(
            f"Invalid duration '{duration_str}'. Use format like '7d', '24h', '30m', '60s'."
        )
    value, unit = int(match.group(1)), match.group(2)
    delta_map = {"d": timedelta(days=value), "h": timedelta(hours=value),
                 "m": timedelta(minutes=value), "s": timedelta(seconds=value)}
    return datetime.now() + delta_map[unit]


def _get_episodic(get_config, namespace: str | None = None):
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(cfg.episodic, cfg.embedding, namespace=namespace)


def _get_semantic(get_config):
    from engram.semantic.graph import SemanticGraph
    cfg = get_config()
    return SemanticGraph(cfg.semantic)


def register(app: typer.Typer, get_config, get_namespace=None) -> None:
    """Register episodic commands on the main Typer app."""

    def _resolve_namespace():
        return get_namespace() if get_namespace else None

    @app.command()
    def remember(
        content: str = typer.Argument(..., help="Content to remember"),
        type: str = typer.Option("fact", "--type", "-t", help="Memory type"),
        priority: int = typer.Option(5, "--priority", "-p", help="Priority (1-10)"),
        tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags, e.g. 'deploy,prod'"),
        expires: Optional[str] = typer.Option(None, "--expires", help="Expiry duration, e.g. '7d', '24h', '30m'"),
    ):
        """Store a memory in episodic store."""
        store = _get_episodic(get_config, _resolve_namespace())
        mem_type = MemoryType(type)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        expires_at = _parse_duration(expires) if expires else None

        mem_id = run_async(store.remember(
            content,
            memory_type=mem_type,
            priority=priority,
            tags=tag_list,
            expires_at=expires_at,
        ))
        suffix = ""
        if tag_list:
            suffix += f", tags=[{', '.join(tag_list)}]"
        if expires_at:
            suffix += f", expires={expires_at.strftime('%Y-%m-%d %H:%M')}"
        console.print(f"[green]Remembered[/green] (id={mem_id[:8]}..., type={type}{suffix})")

    @app.command()
    def recall(
        query: str = typer.Argument(..., help="Search query"),
        limit: int = typer.Option(5, "--limit", "-l"),
        type: Optional[str] = typer.Option(None, "--type", "-t"),
        tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags to filter by"),
    ):
        """Search episodic memories."""
        store = _get_episodic(get_config, _resolve_namespace())
        filters = {}
        if type:
            filters["memory_type"] = type

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        results = run_async(store.search(
            query,
            limit=limit,
            filters=filters if filters else None,
            tags=tag_list,
        ))

        # Also search semantic graph for matching entities
        graph = _get_semantic(get_config)
        graph_nodes = run_async(graph.query(query))
        if graph_nodes:
            for node in graph_nodes[:3]:
                attrs = ", ".join(f"{k}={v}" for k, v in node.attributes.items()) if node.attributes else ""
                console.print(f"[yellow][graph][/yellow] {node.type}:{node.name}" + (f" ({attrs})" if attrs else ""))
                # Show related edges
                related = run_async(graph.get_related(node.key))
                for edge in related.get("edges", [])[:5]:
                    console.print(f"  [dim]{edge.from_node} --{edge.relation}--> {edge.to_node}[/dim]")

        if not results and not graph_nodes:
            console.print("[dim]No memories found.[/dim]")
            return

        for mem in results:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            console.print(f"[cyan][{ts}][/cyan] ({mem.memory_type.value}) {mem.content}")
            if mem.entities:
                console.print(f"  [dim]entities: {', '.join(mem.entities)}[/dim]")
            if mem.tags:
                console.print(f"  [dim]tags: {', '.join(mem.tags)}[/dim]")
