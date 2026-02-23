"""ChromaDB-backed episodic memory store."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import litellm

from engram.config import EmbeddingConfig, EpisodicConfig
from engram.models import EpisodicMemory, MemoryType


class GeminiEmbeddingFunction:
    """Custom ChromaDB embedding function using Gemini API via litellm.

    Implements both __call__ and embed_query for ChromaDB >= 1.5 compatibility.
    Falls back to ChromaDB default embedding on API errors.
    """

    def __init__(self, model: str = "gemini/gemini-embedding-001"):
        self._model = model
        self._fallback = None

    def name(self) -> str:
        return "gemini_engram"

    def _get_fallback(self):
        if self._fallback is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._fallback = DefaultEmbeddingFunction()
        return self._fallback

    def _embed(self, input: list[str]) -> list[list[float]]:
        try:
            response = litellm.embedding(model=self._model, input=input)
            return [item["embedding"] for item in response.data]
        except Exception:
            return self._get_fallback()(input=input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB >= 1.5 uses this for query embeddings."""
        return self._embed(input)


class EpisodicStore:
    """ChromaDB-backed episodic memory store."""

    COLLECTION_NAME = "engram_episodic"

    def __init__(self, config: EpisodicConfig, embedding_config: EmbeddingConfig):
        import chromadb
        from pathlib import Path
        import os

        db_path = str(Path(os.path.expanduser(config.path)))
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        self._embedding_fn = GeminiEmbeddingFunction(
            model=f"{embedding_config.provider}/{embedding_config.model}"
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
        entities: list[str] | None = None,
    ) -> str:
        """Store a memory and return its ID."""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        entities = entities or []

        doc_metadata: dict[str, Any] = {
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "priority": priority,
            "timestamp": timestamp,
            "entities": ",".join(entities),
        }
        if metadata:
            doc_metadata.update(metadata)

        try:
            self._collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[doc_metadata],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to store memory: {e}") from e

        return memory_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicMemory]:
        """Search memories by semantic similarity with optional metadata filters."""
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(limit, self._collection.count() or 1),
            }
            if filters:
                kwargs["where"] = filters

            results = self._collection.query(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

        memories: list[EpisodicMemory] = []
        if not results["ids"] or not results["ids"][0]:
            return memories

        for i, mem_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            memories.append(_build_memory(mem_id, doc, meta))

        return memories

    async def get(self, id: str) -> EpisodicMemory | None:
        """Retrieve a single memory by ID."""
        try:
            result = self._collection.get(ids=[id])
        except Exception:
            return None

        if not result["ids"]:
            return None

        doc = result["documents"][0] if result["documents"] else ""
        meta = result["metadatas"][0] if result["metadatas"] else {}
        return _build_memory(result["ids"][0], doc, meta)

    async def delete(self, id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        try:
            self._collection.delete(ids=[id])
            return True
        except Exception:
            return False

    async def stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "count": count,
            "collection": self.COLLECTION_NAME,
        }


def _build_memory(mem_id: str, document: str, metadata: dict[str, Any]) -> EpisodicMemory:
    """Construct EpisodicMemory from ChromaDB result fields."""
    raw_type = metadata.get("memory_type", MemoryType.FACT.value)
    try:
        memory_type = MemoryType(raw_type)
    except ValueError:
        memory_type = MemoryType.FACT

    raw_ts = metadata.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now()
    except (ValueError, TypeError):
        timestamp = datetime.now()

    raw_entities = metadata.get("entities", "")
    entities = [e.strip() for e in raw_entities.split(",") if e.strip()] if raw_entities else []

    # Exclude internal fields from extra metadata
    extra = {k: v for k, v in metadata.items()
              if k not in ("memory_type", "priority", "timestamp", "entities")}

    return EpisodicMemory(
        id=mem_id,
        content=document,
        memory_type=memory_type,
        priority=int(metadata.get("priority", 5)),
        metadata=extra,
        entities=entities,
        timestamp=timestamp,
    )
