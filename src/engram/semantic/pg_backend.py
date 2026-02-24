"""PostgreSQL backend for semantic graph storage — asyncpg connection pool."""

from __future__ import annotations

import logging

logger = logging.getLogger("engram")

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment]
    _ASYNCPG_AVAILABLE = False

# DDL executed on initialize()
_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    key        TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    name       TEXT NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'
)
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    key      TEXT PRIMARY KEY,
    from_key TEXT NOT NULL,
    to_key   TEXT NOT NULL,
    relation TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)",
    "CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)",
    "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_key)",
    "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_key)",
]


class PostgresBackend:
    """asyncpg-backed graph storage with connection pooling."""

    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10) -> None:
        if not _ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is not installed. Run: pip install asyncpg")
        # Never log the DSN — it may contain credentials.
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: asyncpg.Pool | None = None  # type: ignore[name-defined]

    async def initialize(self) -> None:
        """Create connection pool and schema tables/indexes."""
        self._pool = await asyncpg.create_pool(  # type: ignore[union-attr]
            self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_NODES)
            await conn.execute(_CREATE_EDGES)
            for idx_sql in _CREATE_INDEXES:
                await conn.execute(idx_sql)

    def _pool_or_raise(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized — call initialize() first")
        return self._pool

    async def load_nodes(self) -> list[tuple[str, str, str, str]]:
        """Return all nodes as (key, type, name, attrs_json) tuples."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT key, type, name, attributes::text FROM nodes")
        return [(r["key"], r["type"], r["name"], r["attributes"]) for r in rows]

    async def load_edges(self) -> list[tuple[str, str, str, str]]:
        """Return all edges as (key, from_key, to_key, relation) tuples."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT key, from_key, to_key, relation FROM edges")
        return [(r["key"], r["from_key"], r["to_key"], r["relation"]) for r in rows]

    async def save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        """Upsert a single node."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO nodes (key, type, name, attributes)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (key) DO UPDATE
                    SET type=EXCLUDED.type, name=EXCLUDED.name, attributes=EXCLUDED.attributes
                """,
                key, type, name, attrs_json,
            )

    async def save_edge(self, key: str, from_key: str, to_key: str, relation: str) -> None:
        """Upsert a single edge."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO edges (key, from_key, to_key, relation)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (key) DO UPDATE
                    SET from_key=EXCLUDED.from_key, to_key=EXCLUDED.to_key,
                        relation=EXCLUDED.relation
                """,
                key, from_key, to_key, relation,
            )

    async def save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple nodes via COPY + ON CONFLICT in a single transaction."""
        if not rows:
            return
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO nodes (key, type, name, attributes)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (key) DO UPDATE
                        SET type=EXCLUDED.type, name=EXCLUDED.name,
                            attributes=EXCLUDED.attributes
                    """,
                    rows,
                )

    async def save_edges_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple edges in a single transaction."""
        if not rows:
            return
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO edges (key, from_key, to_key, relation)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (key) DO UPDATE
                        SET from_key=EXCLUDED.from_key, to_key=EXCLUDED.to_key,
                            relation=EXCLUDED.relation
                    """,
                    rows,
                )

    async def delete_node(self, key: str) -> None:
        """Delete a node by key."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM nodes WHERE key=$1", key)

    async def delete_edges_for_node(self, key: str) -> None:
        """Delete all edges connected to a node (from or to)."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM edges WHERE from_key=$1 OR to_key=$1", key
            )

    async def delete_edge(self, key: str) -> None:
        """Delete an edge by key."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM edges WHERE key=$1", key)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
