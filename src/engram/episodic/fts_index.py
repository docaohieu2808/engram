"""SQLite FTS5 full-text search index for episodic memories.

Provides exact keyword search as a complement to ChromaDB vector search.
DB is always-on at ~/.engram/fts_index.db; no extra config needed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("engram")

_DEFAULT_DB_PATH = "~/.engram/fts_index.db"


class FtsResult(NamedTuple):
    """Result from FTS5 search."""
    id: str
    snippet: str
    memory_type: str


class FtsIndex:
    """SQLite FTS5 index synced alongside ChromaDB episodic store.

    Thread-safe via WAL mode; all operations are synchronous (caller wraps
    in asyncio.to_thread where needed).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        resolved = Path(os.path.expanduser(db_path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(resolved)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        """Return a connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_table(self) -> None:
        """Create FTS5 virtual table if it doesn't exist."""
        conn = self._connect()
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                id UNINDEXED,
                content,
                memory_type UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        conn.commit()

    def insert(self, memory_id: str, content: str, memory_type: str) -> None:
        """Insert or replace a memory in the FTS index."""
        conn = self._connect()
        # Delete existing entry first to avoid duplicates on upsert
        conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        conn.execute(
            "INSERT INTO memories_fts(id, content, memory_type) VALUES (?, ?, ?)",
            (memory_id, content, memory_type),
        )
        conn.commit()

    def insert_batch(self, entries: list[tuple[str, str, str]]) -> None:
        """Batch insert (id, content, memory_type) tuples."""
        if not entries:
            return
        conn = self._connect()
        ids = [e[0] for e in entries]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM memories_fts WHERE id IN ({placeholders})", ids)
        conn.executemany(
            "INSERT INTO memories_fts(id, content, memory_type) VALUES (?, ?, ?)",
            entries,
        )
        conn.commit()

    def delete(self, memory_id: str) -> None:
        """Remove a memory from the FTS index."""
        conn = self._connect()
        conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        conn.commit()

    def search(self, query: str, limit: int = 10) -> list[FtsResult]:
        """Full-text search using FTS5 MATCH syntax.

        Returns list of FtsResult(id, snippet, memory_type) ordered by relevance.
        Returns empty list on invalid FTS query (e.g. lone operators).
        """
        if not query or not query.strip():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id,
                       snippet(memories_fts, 1, '<b>', '</b>', '...', 32),
                       memory_type
                FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 search error for query %r: %s", query, e)
            return []
        return [FtsResult(id=row[0], snippet=row[1], memory_type=row[2]) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
