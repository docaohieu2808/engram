"""CLI commands for semantic graph: add/remove node/edge, query."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from engram.utils import run_async

console = Console()


def _get_graph(get_config):
    from engram.semantic import create_graph
    cfg = get_config()
    return create_graph(cfg.semantic)


def _get_episodic(get_config):
    from engram.episodic.store import EpisodicStore
    cfg = get_config()
    return EpisodicStore(cfg.episodic, cfg.embedding, namespace="default")


def register(app: typer.Typer, add_app: typer.Typer, remove_app: typer.Typer, get_config) -> None:
    """Register semantic commands on add/remove sub-apps and main app."""

    @add_app.command("node")
    def add_node(
        name: str = typer.Argument(..., help="Node name"),
        type: str = typer.Option(..., "--type", "-t", help="Node type"),
    ):
        """Add a node to semantic graph."""
        from engram.models import SemanticNode
        graph = _get_graph(get_config)
        node = SemanticNode(type=type, name=name)
        is_new = run_async(graph.add_node(node))
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
        graph = _get_graph(get_config)
        edge = SemanticEdge(from_node=from_node, to_node=to_node, relation=relation)
        is_new = run_async(graph.add_edge(edge))
        action = "Added" if is_new else "Updated"
        console.print(f"[green]{action}[/green] edge {edge.key}")

    @remove_app.command("node")
    def remove_node(key: str = typer.Argument(..., help="Node key (Type:Name)")):
        """Remove a node from semantic graph."""
        graph = _get_graph(get_config)
        removed = run_async(graph.remove_node(key))
        if removed:
            console.print(f"[red]Removed[/red] node {key}")
        else:
            console.print(f"[yellow]Not found:[/yellow] {key}")

    @remove_app.command("edge")
    def remove_edge(key: str = typer.Argument(..., help="Edge key")):
        """Remove an edge from semantic graph."""
        graph = _get_graph(get_config)
        removed = run_async(graph.remove_edge(key))
        if removed:
            console.print(f"[red]Removed[/red] edge {key}")
        else:
            console.print(f"[yellow]Not found:[/yellow] {key}")

    @app.command("autolink-orphans")
    def autolink_orphans(
        recent: int = typer.Option(1000, "--recent", help="How many recent episodic memories to scan"),
        min_co_mentions: int = typer.Option(3, "--min-co-mentions", help="Minimum co-mentions to suggest a link"),
        apply: bool = typer.Option(False, "--apply", help="Apply suggestions instead of dry-run"),
        limit: int = typer.Option(50, "--limit", help="Max suggestions to apply"),
        include_person: bool = typer.Option(False, "--include-person", help="Include Person nodes in auto-linking (off by default for safety)"),
    ):
        """Suggest/apply links for orphan nodes from episodic co-mentions."""
        from engram.semantic.orphan_linker import apply_suggestions, suggest_orphan_links

        graph = _get_graph(get_config)
        episodic = _get_episodic(get_config)

        suggestions = run_async(suggest_orphan_links(graph, episodic, recent=recent, min_co_mentions=min_co_mentions))

        if not include_person and suggestions:
            nodes = run_async(graph.get_nodes())
            by_key = {n.key: n for n in nodes}
            suggestions = [
                s for s in suggestions
                if by_key.get(s.orphan_key) and by_key.get(s.target_key)
                and by_key[s.orphan_key].type != "Person"
                and by_key[s.target_key].type != "Person"
            ]

        if not suggestions:
            console.print("[dim]No orphan link suggestions found.[/dim]")
            return

        table = Table(title=f"Orphan Link Suggestions ({len(suggestions)})")
        table.add_column("#", style="dim")
        table.add_column("Orphan", style="yellow")
        table.add_column("Target", style="cyan")
        table.add_column("Relation", style="green")
        table.add_column("Co-mentions", style="magenta")
        for i, s in enumerate(suggestions[: min(len(suggestions), 30)], 1):
            table.add_row(str(i), s.orphan_key, s.target_key, s.relation, str(s.co_mentions))
        console.print(table)

        if apply:
            applied = run_async(apply_suggestions(graph, suggestions, limit=limit))
            console.print(f"[green]Applied[/green] {applied} orphan links")
        else:
            console.print("[dim]Dry-run only. Use --apply to persist links.[/dim]")

    @app.command()
    def query(
        keyword: str = typer.Argument(None, help="Search keyword"),
        type: Optional[str] = typer.Option(None, "--type", "-t"),
        related_to: Optional[str] = typer.Option(None, "--related-to", "-r"),
        format: str = typer.Option("table", "--format", "-f", help="Output format: table|json"),
    ):
        """Query semantic graph."""
        graph = _get_graph(get_config)

        if related_to:
            results = run_async(graph.get_related([related_to], depth=2))
            if format == "json":
                console.print_json(json.dumps(results, default=str))
            else:
                console.print(f"[bold]Related to: {related_to}[/bold]")
                _print_related(results)
            return

        if keyword:
            nodes = run_async(graph.query(keyword, type=type))
        else:
            nodes = run_async(graph.get_nodes(type=type))

        if format == "json":
            console.print_json(json.dumps([n.model_dump() for n in nodes], default=str))
        else:
            _print_nodes_table(nodes)


def _print_nodes_table(nodes) -> None:
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


def _print_related(results) -> None:
    if not results:
        console.print("[dim]No related entities found.[/dim]")
        return
    for entity, data in results.items():
        console.print(f"\n[bold cyan]{entity}[/bold cyan]")
        if isinstance(data, dict):
            for key, value in data.items():
                console.print(f"  {key}: {value}")
