"""SQLite backend for semantic graph storage — synchronous sqlite3 wrapped as async."""

from __future__ import annotations

import asyncio
import sqlite3
from functools import partial
from pathlib import Path


class SqliteBackend:
    """SQLite-backed graph storage.

    Sync sqlite3 operations run in the default thread-pool executor via
    asyncio.get_running_loop().run_in_executor() to avoid blocking the event loop.
    Uses check_same_thread=False for safe cross-thread access.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        """Return cached connection (check_same_thread=False for executor use)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # --- Sync helpers (called from executor thread pool) ---

    def _sync_initialize(self) -> None:
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

    def _sync_load_nodes(self) -> list[tuple[str, str, str, str]]:
        conn = self._connect()
        return list(conn.execute("SELECT key, type, name, attributes FROM nodes"))

    def _sync_load_edges(self) -> list[tuple[str, str, str, str]]:
        conn = self._connect()
        return list(conn.execute("SELECT key, from_key, to_key, relation FROM edges"))

    def _sync_save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
            (key, type, name, attrs_json),
        )
        conn.commit()

    def _sync_save_edge(self, key: str, from_key: str, to_key: str, relation: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
            (key, from_key, to_key, relation),
        )
        conn.commit()

    def _sync_save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        if not rows:
            return
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO nodes (key, type, name, attributes) VALUES (?, ?, ?, ?)",
                rows,
            )

    def _sync_save_edges_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        if not rows:
            return
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO edges (key, from_key, to_key, relation) VALUES (?, ?, ?, ?)",
                rows,
            )

    def _sync_delete_node(self, key: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM nodes WHERE key=?", (key,))
        conn.commit()

    def _sync_delete_edges_for_node(self, key: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE from_key=? OR to_key=?", (key, key))
        conn.commit()

    def _sync_delete_edge(self, key: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE key=?", (key,))
        conn.commit()

    def _sync_close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Async public interface (runs sync ops in executor) ---

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        await asyncio.get_running_loop().run_in_executor(None, self._sync_initialize)

    async def load_nodes(self) -> list[tuple[str, str, str, str]]:
        """Return all nodes as (key, type, name, attrs_json) tuples."""
        return await asyncio.get_running_loop().run_in_executor(None, self._sync_load_nodes)

    async def load_edges(self) -> list[tuple[str, str, str, str]]:
        """Return all edges as (key, from_key, to_key, relation) tuples."""
        return await asyncio.get_running_loop().run_in_executor(None, self._sync_load_edges)

    async def save_node(self, key: str, type: str, name: str, attrs_json: str) -> None:
        """Upsert a single node."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_save_node, key, type, name, attrs_json)
        )

    async def save_edge(self, key: str, from_key: str, to_key: str, relation: str) -> None:
        """Upsert a single edge."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_save_edge, key, from_key, to_key, relation)
        )

    async def save_nodes_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple nodes in one transaction."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_save_nodes_batch, rows)
        )

    async def save_edges_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """Upsert multiple edges in one transaction."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_save_edges_batch, rows)
        )

    async def delete_node(self, key: str) -> None:
        """Delete a node by key."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_delete_node, key)
        )

    async def delete_edges_for_node(self, key: str) -> None:
        """Delete all edges connected to a node (from or to)."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_delete_edges_for_node, key)
        )

    async def delete_edge(self, key: str) -> None:
        """Delete an edge by key."""
        await asyncio.get_running_loop().run_in_executor(
            None, partial(self._sync_delete_edge, key)
        )

    async def close(self) -> None:
        """Close the SQLite connection."""
        await asyncio.get_running_loop().run_in_executor(None, self._sync_close)
