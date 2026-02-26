"""CLI commands for episodic memory: remember and recall."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
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
    return datetime.now(timezone.utc) + delta_map[unit]


def _get_episodic(get_config, namespace: str | None = None):
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(
        cfg.episodic,
        cfg.embedding,
        namespace=namespace,
        guard_enabled=cfg.ingestion.poisoning_guard,
    )


def _get_semantic(get_config):
    from engram.semantic import create_graph
    cfg = get_config()
    return create_graph(cfg.semantic)


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
        topic_key: Optional[str] = typer.Option(None, "--topic-key", help="Unique key; updates existing memory if same key exists"),
    ):
        """Store a memory in episodic store."""
        if len(content) > 10_000:
            console.print("[red]Error: Content too long (max 10,000 chars)[/red]")
            raise typer.Exit(1)
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
            topic_key=topic_key,
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
        no_federation: bool = typer.Option(False, "--no-federation", help="Skip federated provider search"),
        timeout: float = typer.Option(10.0, "--timeout", help="Federated search timeout in seconds"),
        resolve_entities: bool = typer.Option(False, "--resolve-entities", help="Resolve pronouns in query before searching"),
        resolve_temporal: bool = typer.Option(False, "--resolve-temporal", help="Resolve temporal references in query before searching"),
    ):
        """Search episodic memories and federated providers."""
        # Entity/temporal resolution pre-processing
        if resolve_entities or resolve_temporal:
            from engram.recall.entity_resolver import resolve as do_resolve
            resolved = run_async(do_resolve(
                query,
                context=None,
                resolve_temporal_refs=resolve_temporal,
                resolve_pronoun_refs=resolve_entities,
            ))
            if resolved.resolved != query:
                console.print(f"[dim]Resolved query:[/dim] {resolved.resolved}")
                query = resolved.resolved

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
                related = run_async(graph.get_related([node.name]))
                for edge in related.get(node.name, {}).get("edges", [])[:5]:
                    console.print(f"  [dim]{edge.from_node} --{edge.relation}--> {edge.to_node}[/dim]")

        # Federated search across external providers
        provider_results = []
        if not no_federation:
            from engram.providers.registry import ProviderRegistry
            from engram.providers.router import federated_search
            cfg = get_config()
            registry = ProviderRegistry()
            registry.load_from_config(cfg)
            providers = registry.get_active()
            if providers:
                provider_results = run_async(federated_search(query, providers, limit=limit, timeout_seconds=timeout))
                for r in provider_results:
                    console.print(f"[magenta]\\[{r.source}][/magenta] {r.content[:300]}")

        if not results and not graph_nodes and not provider_results:
            console.print("[dim]No memories found.[/dim]")
            return

        for mem in results:
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            console.print(f"[cyan][{ts}][/cyan] ({mem.memory_type.value}) {mem.content}")
            if mem.entities:
                console.print(f"  [dim]entities: {', '.join(mem.entities)}[/dim]")
            if mem.tags:
                console.print(f"  [dim]tags: {', '.join(mem.tags)}[/dim]")
