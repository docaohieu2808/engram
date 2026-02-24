"""CLI commands for config and schema management."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def register(
    config_app: typer.Typer,
    schema_app: typer.Typer,
    get_config,
    save_config,
    get_config_value,
    set_config_value,
) -> None:
    """Register config and schema commands on their respective sub-apps."""

    # --- Config commands ---

    @config_app.command("show")
    def config_show():
        """Show current configuration."""
        cfg = get_config()
        console.print_json(cfg.model_dump_json(indent=2))

    @config_app.command("set")
    def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)):
        """Set a config value (dot notation: llm.model)."""
        set_config_value(key, value)
        console.print(f"[green]Set[/green] {key} = {value}")

    @config_app.command("get")
    def config_get(key: str = typer.Argument(...)):
        """Get a config value."""
        cfg = get_config()
        val = get_config_value(cfg, key)
        console.print(f"{key} = {val}")

    # --- Schema commands ---

    @schema_app.command("show")
    def schema_show():
        """Show current schema."""
        from engram.schema.loader import load_schema, schema_to_prompt
        cfg = get_config()
        schema = load_schema(cfg.semantic.schema_name)
        console.print(schema_to_prompt(schema))

    @schema_app.command("init")
    def schema_init(template: str = typer.Option("devops", "--template", "-t")):
        """Initialize schema from template."""
        from engram.schema.loader import load_schema
        load_schema(template)
        cfg = get_config()
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
