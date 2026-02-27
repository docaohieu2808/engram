"""Abstract backend protocol for episodic vector storage.

Mirrors the pattern used in semantic/backend.py.
Concrete implementations: chromadb_backend.py, chromadb_http_backend.py
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EpisodicBackend(Protocol):
    """Protocol for episodic storage backends (embedded ChromaDB, HTTP ChromaDB, etc.)."""

    async def initialize(self, namespace: str, embedding_dim: int | None) -> None:
        """Create or open the collection for the given namespace."""
        ...

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict,
    ) -> None:
        """Insert a new memory document."""
        ...

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Batch upsert multiple memory documents."""
        ...

    async def get(self, id: str) -> dict | None:
        """Retrieve a single document by ID. Returns None if not found.

        Result dict has keys: id, document, metadata.
        """
        ...

    async def get_many(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """Retrieve multiple documents. Returns raw ChromaDB-style result dict."""
        ...

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID list."""
        ...

    async def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """Vector similarity query. Returns raw ChromaDB-style result dict."""
        ...

    async def update(self, ids: list[str], metadatas: list[dict]) -> None:
        """Update metadata for existing documents."""
        ...

    async def update_with_embeddings(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Update documents including embeddings (used for topic-key upsert)."""
        ...

    async def count(self) -> int:
        """Return total number of documents in the collection."""
        ...

    async def peek(self, limit: int = 1) -> dict:
        """Return a small sample of documents (used for dimension detection)."""
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...
