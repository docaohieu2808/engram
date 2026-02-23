"""Search helpers for episodic memory - temporal and filtered queries."""

from __future__ import annotations

from engram.episodic.store import EpisodicStore
from engram.models import EpisodicMemory


async def temporal_search(
    store: EpisodicStore,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5,
) -> list[EpisodicMemory]:
    """Search memories within an optional timestamp range.

    Args:
        store: EpisodicStore instance
        query: Semantic search query
        start_date: ISO datetime string (inclusive lower bound)
        end_date: ISO datetime string (inclusive upper bound)
        limit: Max results to return
    """
    filters: dict | None = None

    conditions: list[dict] = []
    if start_date:
        conditions.append({"timestamp": {"$gte": start_date}})
    if end_date:
        conditions.append({"timestamp": {"$lte": end_date}})

    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}

    return await store.search(query, limit=limit, filters=filters)


async def filtered_search(
    store: EpisodicStore,
    query: str,
    memory_type: str | None = None,
    entities: list[str] | None = None,
    limit: int = 5,
) -> list[EpisodicMemory]:
    """Search memories filtered by type and/or entities.

    Args:
        store: EpisodicStore instance
        query: Semantic search query
        memory_type: MemoryType value string (e.g. "fact", "decision")
        entities: List of entity names; memories must contain at least one
        limit: Max results to return
    """
    conditions: list[dict] = []

    if memory_type:
        conditions.append({"memory_type": {"$eq": memory_type}})

    # Entities are stored as comma-joined string; use $contains per entity
    if entities:
        entity_conditions = [{"entities": {"$contains": e}} for e in entities]
        if len(entity_conditions) == 1:
            conditions.extend(entity_conditions)
        else:
            conditions.append({"$or": entity_conditions})

    filters: dict | None = None
    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}

    return await store.search(query, limit=limit, filters=filters)
