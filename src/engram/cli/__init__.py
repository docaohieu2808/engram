"""engram CLI - Memory traces for AI agents."""

from __future__ import annotations

import logging
from typing import Optional

import typer

from engram.config import (
    Config,
    get_config_value,
    load_config,
    save_config,
    set_config_value,
)
from engram.logging_setup import setup_logging

# Bootstrap logging from config (respects ENGRAM_LOG_FORMAT / ENGRAM_LOG_LEVEL)
setup_logging(load_config())

app = typer.Typer(name="engram", help="Memory traces for AI agents - Think like human")
add_app = typer.Typer(help="Add nodes/edges to semantic memory")
remove_app = typer.Typer(help="Remove nodes/edges from semantic memory")
schema_app = typer.Typer(help="Manage semantic schemas")
config_app = typer.Typer(help="Manage configuration")
auth_app = typer.Typer(help="Manage API keys and authentication")

app.add_typer(add_app, name="add")
app.add_typer(remove_app, name="remove")
app.add_typer(schema_app, name="schema")
app.add_typer(config_app, name="config")
app.add_typer(auth_app, name="auth")

_config: Config | None = None
_namespace: Optional[str] = None


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_namespace() -> Optional[str]:
    return _namespace


def _set_config_value(key: str, value: str) -> Config:
    global _config
    _config = set_config_value(key, value)
    return _config


@app.callback()
def main(
    namespace: Optional[str] = typer.Option(
        None, "--namespace", "-n",
        help="Memory namespace (isolates ChromaDB collection). Default: 'default'.",
        envvar="ENGRAM_NAMESPACE",
    ),
):
    """engram - Memory traces for AI agents."""
    global _namespace
    _namespace = namespace


# Register commands from sub-modules
from engram.cli import episodic as _episodic_mod  # noqa: E402
from engram.cli import semantic as _semantic_mod  # noqa: E402
from engram.cli import reasoning as _reasoning_mod  # noqa: E402
from engram.cli import system as _system_mod  # noqa: E402
from engram.cli import migrate_cmd as _migrate_mod  # noqa: E402
from engram.cli import config_cmd as _config_cmd_mod  # noqa: E402
from engram.cli import auth_cmd as _auth_cmd_mod  # noqa: E402

_episodic_mod.register(app, _get_config, get_namespace=_get_namespace)
_semantic_mod.register(app, add_app, remove_app, _get_config)
_reasoning_mod.register(app, _get_config)
_system_mod.register(app, _get_config, get_namespace=_get_namespace)
_migrate_mod.register(app, _get_config)
_config_cmd_mod.register(
    config_app,
    schema_app,
    _get_config,
    save_config,
    get_config_value,
    _set_config_value,
)
_auth_cmd_mod.register(auth_app, _get_config)

if __name__ == "__main__":
    app()
