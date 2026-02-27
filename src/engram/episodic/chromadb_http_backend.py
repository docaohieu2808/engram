"""ChromaDB HTTP backend for episodic storage.

Uses chromadb.HttpClient to connect to a remote ChromaDB server.
Same interface as ChromaDBBackend (embedded) — only client construction differs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("engram")


class ChromaDBHttpBackend:
    """EpisodicBackend implementation using chromadb.HttpClient (remote server)."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        import chromadb
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection: Any = None

    async def initialize(self, namespace: str, embedding_dim: int | None = None) -> None:
        """Create or open the named collection on the remote ChromaDB server."""
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=namespace,
            metadata={"hnsw:space": "cosine"},
        )

    def _col(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaDBHttpBackend.initialize() must be called before use")
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
        """No explicit teardown needed for HttpClient."""
        self._collection = None
