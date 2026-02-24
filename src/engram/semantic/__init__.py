"""Semantic memory - graph DB for entities and relationships."""

from __future__ import annotations

from engram.semantic.graph import SemanticGraph

__all__ = ["SemanticGraph", "create_graph"]


def create_graph(config: "engram.config.SemanticConfig") -> SemanticGraph:  # type: ignore[name-defined]
    """Factory: pick backend from config.provider and return a SemanticGraph."""
    from engram.config import SemanticConfig  # local import avoids circular deps

    if not isinstance(config, SemanticConfig):
        raise TypeError(f"Expected SemanticConfig, got {type(config)}")

    if config.provider == "postgresql":
        from engram.semantic.pg_backend import PostgresBackend
        backend = PostgresBackend(
            dsn=config.dsn,
            pool_min=config.pool_min,
            pool_max=config.pool_max,
        )
    else:
        # Default: SQLite
        from engram.semantic.sqlite_backend import SqliteBackend
        backend = SqliteBackend(config.path)

    return SemanticGraph(backend)
