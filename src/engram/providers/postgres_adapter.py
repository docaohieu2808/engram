"""PostgreSQL adapter for querying external databases as memory providers."""

from __future__ import annotations

import logging
import os
from typing import Any

from engram.providers.base import MemoryProvider, ProviderResult

logger = logging.getLogger("engram.providers.postgres")


class PostgresAdapter(MemoryProvider):
    """Executes parameterized SQL queries against a PostgreSQL database."""

    def __init__(
        self,
        name: str,
        dsn: str,
        search_query: str,
        **kwargs: Any,
    ):
        super().__init__(name=name, provider_type="postgres", **kwargs)
        self.dsn = os.path.expandvars(dsn)
        self.search_query = search_query
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=3)
        return self._pool

    async def search(self, query: str, limit: int = 5) -> list[ProviderResult]:
        pool = await self._get_pool()

        # Use asyncpg parameterized queries to prevent SQL injection
        # Replace {query} → $1 and {limit} → $2 placeholders
        sql = self.search_query.replace("{query}", "$1").replace("{limit}", "$2")
        params = [f"%{query}%", limit]

        if self.debug:
            logger.debug("[%s] SQL: %s params: %s", self.name, sql, params)

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for i, row in enumerate(rows[:limit]):
            # Use first column as content, rest as metadata
            columns = list(row.keys())
            content = str(row[columns[0]]) if columns else ""
            meta = {k: str(v) for k, v in dict(row).items() if k != columns[0]}

            results.append(ProviderResult(
                content=content,
                score=1.0 - (i * 0.1),
                source=self.name,
                metadata=meta,
            ))

        return results

    async def health(self) -> bool:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
