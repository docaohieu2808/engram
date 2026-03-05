"""Qdrant backend for episodic vector storage.

Implements EpisodicBackend protocol using qdrant-client.
Connects to a remote Qdrant server via HTTP/gRPC with API key auth.
Returns ChromaDB-style result dicts for compatibility with existing code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger("engram")


class QdrantBackend:
    """EpisodicBackend implementation using Qdrant vector database."""

    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient
        # Use url= to force HTTP REST (avoids gRPC SSL issues over WireGuard)
        self._client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key, timeout=30)
        self._collection_name: str = ""

    async def initialize(self, namespace: str, embedding_dim: int | None = None) -> None:
        """Create collection if it doesn't exist, ensure payload indexes."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        self._collection_name = namespace
        try:
            await asyncio.to_thread(self._client.get_collection, self._collection_name)
            logger.info("Qdrant collection '%s' exists", self._collection_name)
        except Exception:
            dim = embedding_dim or 3072  # default: gemini-embedding-001
            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self._collection_name, dim)
        # Ensure payload index on timestamp for order_by support (needs DATETIME for range)
        try:
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=self._collection_name,
                field_name="timestamp",
                field_schema=PayloadSchemaType.DATETIME,
            )
        except Exception:
            pass  # index already exists

    def _col(self) -> str:
        if not self._collection_name:
            raise RuntimeError("QdrantBackend.initialize() must be called before use")
        return self._collection_name

    async def add(self, id: str, embedding: list[float], content: str, metadata: dict) -> None:
        """Insert a single memory document."""
        from qdrant_client.models import PointStruct
        point = PointStruct(id=id, vector=embedding, payload={"document": content, **metadata})
        await asyncio.to_thread(self._client.upsert, collection_name=self._col(), points=[point])

    async def upsert(self, ids: list[str], embeddings: list[list[float]] | None, documents: list[str], metadatas: list[dict]) -> None:
        """Batch upsert multiple memory documents."""
        from qdrant_client.models import PointStruct
        # Filter out records with no embedding vector (Qdrant requires vectors)
        valid = [
            (ids[i], embeddings[i] if embeddings else None, documents[i], metadatas[i])
            for i in range(len(ids))
            if embeddings and embeddings[i] is not None
        ]
        if not valid:
            logger.warning("QdrantBackend.upsert: all embeddings are None — skipping upsert (%d items)", len(ids))
            return
        points = [
            PointStruct(id=vid, vector=vemb, payload={"document": vdoc, **vmeta})
            for vid, vemb, vdoc, vmeta in valid
        ]
        # Batch in chunks of 500
        for start in range(0, len(points), 500):
            await asyncio.to_thread(self._client.upsert, collection_name=self._col(), points=points[start:start + 500])

    async def get(self, id: str) -> dict | None:
        """Retrieve a single document by ID."""
        results = await asyncio.to_thread(self._client.retrieve, collection_name=self._col(), ids=[id], with_payload=True)
        if not results:
            return None
        point = results[0]
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "document": payload.pop("document", ""),
            "metadata": payload,
        }

    async def get_many(self, ids: list[str] | None = None, where: dict | None = None, include: list[str] | None = None, limit: int | None = None, offset: int | None = None) -> dict:
        """Retrieve multiple documents. Returns ChromaDB-style result dict.

        Uses order_by=timestamp desc to get most recent results.
        The offset parameter is ignored for Qdrant scroll (not compatible).
        """
        if ids is not None:
            points = await asyncio.to_thread(self._client.retrieve, collection_name=self._col(), ids=ids, with_payload=True, with_vectors="embeddings" in (include or []))
        else:
            from qdrant_client.models import Filter, OrderBy, Direction
            qdrant_filter = _build_qdrant_filter(where) if where else None
            try:
                points, _next_offset = await asyncio.to_thread(
                    self._client.scroll,
                    collection_name=self._col(), scroll_filter=qdrant_filter,
                    limit=limit or 100,
                    with_payload=True, with_vectors="embeddings" in (include or []),
                    order_by=OrderBy(key="timestamp", direction=Direction.DESC),
                )
            except Exception:
                # Fallback: no order_by (index may not exist yet)
                points, _next_offset = await asyncio.to_thread(
                    self._client.scroll,
                    collection_name=self._col(), scroll_filter=qdrant_filter,
                    limit=limit or 100,
                    with_payload=True, with_vectors="embeddings" in (include or []),
                )
        return _points_to_chroma_dict(points, include)

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID list."""
        from qdrant_client.models import PointIdsList
        await asyncio.to_thread(self._client.delete, collection_name=self._col(), points_selector=PointIdsList(points=ids))

    async def query(self, query_embeddings: list[list[float]], n_results: int, where: dict | None = None, include: list[str] | None = None) -> dict:
        """Vector similarity search. Returns ChromaDB-style nested result dict."""
        from qdrant_client.models import Filter
        qdrant_filter = _build_qdrant_filter(where) if where else None
        all_ids, all_docs, all_metas, all_dists, all_embeddings = [], [], [], [], []
        for qe in query_embeddings:
            _resp = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self._col(), query=qe, limit=n_results,
                query_filter=qdrant_filter, with_payload=True,
                with_vectors="embeddings" in (include or []),
            )
            results = _resp.points
            ids, docs, metas, dists, embeddings = [], [], [], [], []
            for hit in results:
                payload = dict(hit.payload or {})
                ids.append(str(hit.id))
                docs.append(payload.pop("document", ""))
                metas.append(payload)
                dists.append(1.0 - hit.score)  # cosine distance = 1 - similarity
                if hasattr(hit, "vector") and hit.vector:
                    embeddings.append(hit.vector)
            all_ids.append(ids)
            all_docs.append(docs)
            all_metas.append(metas)
            all_dists.append(dists)
            if embeddings:
                all_embeddings.append(embeddings)
        result: dict[str, Any] = {"ids": all_ids, "documents": all_docs, "metadatas": all_metas, "distances": all_dists}
        if all_embeddings:
            result["embeddings"] = all_embeddings
        return result

    async def update(self, ids: list[str], metadatas: list[dict]) -> None:
        """Update metadata for existing documents."""
        for i, doc_id in enumerate(ids):
            await asyncio.to_thread(self._client.set_payload, collection_name=self._col(), payload=metadatas[i], points=[doc_id])

    async def update_with_embeddings(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Update documents including embeddings."""
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=ids[i], vector=embeddings[i], payload={"document": documents[i], **metadatas[i]})
            for i in range(len(ids))
        ]
        await asyncio.to_thread(self._client.upsert, collection_name=self._col(), points=points)

    async def count(self) -> int:
        """Return total number of documents."""
        info = await asyncio.to_thread(self._client.get_collection, self._col())
        return info.points_count or 0

    async def peek(self, limit: int = 1) -> dict:
        """Return a small sample of documents."""
        points, _ = await asyncio.to_thread(self._client.scroll, collection_name=self._col(), limit=limit, with_payload=True, with_vectors=True)
        return _points_to_chroma_dict(points, ["documents", "metadatas", "embeddings"])

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await asyncio.to_thread(self._client.close)
        self._collection_name = ""


def _build_qdrant_filter(where: dict | None) -> Any:
    """Convert ChromaDB-style where filter to Qdrant Filter."""
    if not where:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    conditions = []
    for key, value in where.items():
        if key.startswith("$"):  # skip $and/$or operators for now
            continue
        if isinstance(value, dict):
            # handle {"$eq": val}, {"$ne": val}, etc.
            for op, val in value.items():
                if op == "$eq":
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None


def _points_to_chroma_dict(points: list, include: list[str] | None = None) -> dict:
    """Convert Qdrant points to ChromaDB-style result dict."""
    ids, docs, metas, embeddings = [], [], [], []
    for p in points:
        payload = dict(p.payload or {})
        ids.append(str(p.id))
        docs.append(payload.pop("document", ""))
        metas.append(payload)
        if hasattr(p, "vector") and p.vector:
            embeddings.append(p.vector)
    result: dict[str, Any] = {"ids": ids, "documents": docs, "metadatas": metas}
    if embeddings and include and "embeddings" in include:
        result["embeddings"] = embeddings
    return result
