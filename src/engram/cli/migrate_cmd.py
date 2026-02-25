"""CLI command for migrating legacy memory exports into engram."""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from rich.console import Console

from engram.models import MemoryType
from engram.utils import run_async

console = Console()

_ENTITY_PREFIXES = {
    "Server": "Server", "Technology": "Technology", "Person": "Person",
    "Project": "Project", "Environment": "Environment", "Script": "Script",
    "Service": "Service",
}


def register(app: typer.Typer, get_config) -> None:
    """Register migrate command on the main Typer app."""

    def _get_graph():
        from engram.semantic import create_graph
        cfg = get_config()
        return create_graph(cfg.semantic)

    def _get_episodic():
        from engram.episodic.store import EpisodicStore
        cfg = get_config()
        return EpisodicStore(cfg.episodic, cfg.embedding)

    @app.command()
    def migrate(
        file: Path = typer.Argument(..., help="JSON file to import (legacy memory export)"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Preview without importing"),
    ):
        """Import data from legacy memory JSON exports.

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

        nodes_to_add, edges_to_add, episodic_texts = _parse_messages(messages)

        console.print(
            f"[bold]Parsed:[/bold] {len(nodes_to_add)} nodes, "
            f"{len(edges_to_add)} edges, {len(episodic_texts)} episodic"
        )

        if dry_run:
            _print_dry_run(nodes_to_add, edges_to_add)
            return

        added_nodes, added_edges = _import_graph(nodes_to_add, edges_to_add, _get_graph)
        episodic_count = _import_episodic(episodic_texts, _get_episodic)

        console.print(
            f"[green]Imported:[/green] {added_nodes} new nodes, "
            f"{added_edges} new edges, {episodic_count} episodic memories"
        )


def _parse_messages(messages):
    """Parse messages into nodes, edges, and episodic texts."""
    nodes_to_add: list[tuple[str, str, dict]] = []
    edges_to_add: list[tuple[str, str, str]] = []
    episodic_texts: list[str] = []

    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if not content or not content.strip():
            continue

        parsed = False
        if content.startswith("Relationship:"):
            match = re.match(r"Relationship:\s*(.+?)\s*--(\w+)-->\s*(.+)", content)
            if match:
                from_name, relation, to_name = match.groups()
                edges_to_add.append((from_name.strip(), to_name.strip(), relation.strip()))
                parsed = True

        if not parsed:
            for prefix, node_type in _ENTITY_PREFIXES.items():
                if content.startswith(f"{prefix}:"):
                    rest = content[len(prefix) + 1:].strip()
                    name, attrs = rest, {}
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

    return nodes_to_add, edges_to_add, episodic_texts


def _print_dry_run(nodes_to_add, edges_to_add) -> None:
    for ntype, name, attrs in nodes_to_add[:10]:
        console.print(f"  [cyan]Node:[/cyan] {ntype}:{name} {attrs if attrs else ''}")
    for fk, tk, rel in edges_to_add[:10]:
        console.print(f"  [green]Edge:[/green] {fk} --{rel}--> {tk}")
    if len(nodes_to_add) > 10:
        console.print(f"  ... and {len(nodes_to_add) - 10} more nodes")
    if len(edges_to_add) > 10:
        console.print(f"  ... and {len(edges_to_add) - 10} more edges")


def _import_graph(nodes_to_add, edges_to_add, get_graph) -> tuple[int, int]:
    from engram.models import SemanticEdge, SemanticNode
    graph = get_graph()
    added_nodes = sum(
        1 for ntype, name, attrs in nodes_to_add
        if run_async(graph.add_node(SemanticNode(type=ntype, name=name, attributes=attrs)))
    )
    all_nodes = run_async(graph.get_nodes())
    name_to_key = {n.name: n.key for n in all_nodes}
    added_edges = 0
    for from_name, to_name, relation in edges_to_add:
        from_key = name_to_key.get(from_name)
        to_key = name_to_key.get(to_name)
        if not from_key or not to_key:
            console.print(
                f"[yellow]Skipped edge: {from_name} --{relation}--> {to_name} "
                f"(node(s) not found)[/yellow]"
            )
            continue
        edge = SemanticEdge(from_node=from_key, to_node=to_key, relation=relation)
        if run_async(graph.add_edge(edge)):
            added_edges += 1
    return added_nodes, added_edges


def _import_episodic(episodic_texts: list[str], get_episodic) -> int:
    if not episodic_texts:
        return 0
    store = get_episodic()
    seen: set[str] = set()
    count = 0
    for text in episodic_texts:
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        run_async(store.remember(text, memory_type=MemoryType.CONTEXT, priority=4))
        count += 1
    return count
