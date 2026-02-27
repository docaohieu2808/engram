"""ChromaDB embedded (PersistentClient) backend for episodic storage.

Wraps all synchronous ChromaDB collection calls in asyncio.to_thread()
so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("engram")

_INCLUDE_DEFAULT = ["documents", "metadatas", "embeddings"]


class ChromaDBBackend:
    """EpisodicBackend implementation using chromadb.PersistentClient (local file store)."""

    def __init__(self, db_path: str) -> None:
        resolved = str(Path(os.path.expanduser(db_path)))
        Path(resolved).mkdir(parents=True, exist_ok=True)
        import chromadb
        self._client = chromadb.PersistentClient(path=resolved)
        self._collection: Any = None

    async def initialize(self, namespace: str, embedding_dim: int | None = None) -> None:
        """Create or open the named collection (cosine similarity, no built-in embedder)."""
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=namespace,
            metadata={"hnsw:space": "cosine"},
        )

    def _col(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaDBBackend.initialize() must be called before use")
        return self._collection

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict,
    ) -> None:
        await asyncio.to_thread(
            self._col().add,
            ids=[id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        await asyncio.to_thread(
            self._col().upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def get(self, id: str) -> dict | None:
        result = await asyncio.to_thread(self._col().get, ids=[id])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": (result["documents"] or [""])[0],
            "metadata": (result["metadatas"] or [{}])[0],
        }

    async def get_many(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        if include is not None:
            kwargs["include"] = include
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        return await asyncio.to_thread(self._col().get, **kwargs)

    async def delete(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._col().delete, ids=ids)

    async def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where is not None:
            kwargs["where"] = where
        if include is not None:
            kwargs["include"] = include
        return await asyncio.to_thread(self._col().query, **kwargs)

    async def update(self, ids: list[str], metadatas: list[dict]) -> None:
        await asyncio.to_thread(self._col().update, ids=ids, metadatas=metadatas)

    async def update_with_embeddings(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        await asyncio.to_thread(
            self._col().update,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def count(self) -> int:
        return await asyncio.to_thread(self._col().count)

    async def peek(self, limit: int = 1) -> dict:
        return await asyncio.to_thread(self._col().peek, limit)

    async def close(self) -> None:
        """No explicit teardown needed for PersistentClient."""
        self._collection = None
