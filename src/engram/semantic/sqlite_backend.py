"""SQLite backend for semantic graph storage — synchronous sqlite3 wrapped as async."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteBackend:
    """SQLite-backed graph storage. Sync sqlite3 ops, wrapped with async interface."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        """Return cached connection, creating if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS nodes "
            "(key TEXT PRIMARY KEY, type TEXT, name TEXT, attributes TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS edges "
            "(key TEXT PRIMARY KEY, from_key TEXT, to_key TEXT, relation TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)")
        conn.commit()

    async def load_nodes(self) -> list[tuple[str, str, str, str]]:
        """Return all nodes as (key, type, name, attrs_json) tuples."""
        conn = self._connect()
        return list(conn.execute("SELECT key, type, name, attributes FROM nodes"))

    async def load_edges(self) -> list[tuple[str, str, str, str]]:
        """Return all edges as (key, from_key, to_key, relation) tuples."""
        conn = self._connect()
        return list(conn.execute("SELECT key, from_key, to_key, relation FROM edges"))

    async def save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        """Upsert a single node."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
            (key, type, name, attrs_json),
        )
        conn.commit()

    async def save_edge(self, key: str, from_key: str, to_key: str, relation: str) -> None:
        """Upsert a single edge."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
            (key, from_key, to_key, relation),
        )
        conn.commit()

    async def save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple nodes in one transaction."""
        if not rows:
            return
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
                rows,
            )

    async def save_edges_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple edges in one transaction."""
        if not rows:
            return
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
                rows,
            )

    async def delete_node(self, key: str) -> None:
        """Delete a node by key."""
        conn = self._connect()
        conn.execute("DELETE FROM nodes WHERE key=?", (key,))
        conn.commit()

    async def delete_edges_for_node(self, key: str) -> None:
        """Delete all edges connected to a node (from or to)."""
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE from_key=? OR to_key=?", (key, key))
        conn.commit()

    async def delete_edge(self, key: str) -> None:
        """Delete an edge by key."""
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE key=?", (key,))
        conn.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
