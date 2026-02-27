"""Shim backend wrapping a raw ChromaDB collection object.

Used only when tests inject a mock/real collection directly via
`store._collection = obj` without going through `EpisodicStore.__init__`.
This adapter translates async backend calls into synchronous collection calls,
preserving backward compatibility without modifying test files.
"""

from __future__ import annotations

import asyncio
from typing import Any


class _LegacyCollectionBackend:
    """Wraps a synchronous ChromaDB collection object as an async EpisodicBackend."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    async def initialize(self, namespace: str, embedding_dim: int | None = None) -> None:
        pass  # already initialised

    def _col(self) -> Any:
        return self._collection

    async def add(self, id: str, embedding: list[float], content: str, metadata: dict) -> None:
        await asyncio.to_thread(
            self._col().add,
            ids=[id], documents=[content], embeddings=[embedding], metadatas=[metadata],
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
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas,
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
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas,
        )

    async def count(self) -> int:
        return await asyncio.to_thread(self._col().count)

    async def peek(self, limit: int = 1) -> dict:
        return await asyncio.to_thread(self._col().peek, limit)

    async def close(self) -> None:
        pass
