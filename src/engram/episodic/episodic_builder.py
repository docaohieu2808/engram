"""Module-level builder helpers shared across episodic mixin modules.

Provides pure functions for JSON parsing, entity canonicalization,
collection name construction, and EpisodicMemory object construction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from engram.models import EpisodicMemory, MemoryType

logger = logging.getLogger("engram")


def _safe_json_list(val: str | None) -> list:
    """Parse a JSON string as list, returning [] on any failure."""
    if not val:
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _canonicalize_entities(entities: list[str] | None) -> list[str]:
    """Normalize + case-insensitive dedupe entity names.

    Keeps first meaningful casing and applies a small canonical map for
    frequent product names to avoid duplicates like Engram/engram.
    """
    if not entities:
        return []

    canonical_map = {
        "engram": "Engram",
        "openclaw": "OpenClaw",
        "claude code": "Claude Code",
    }

    out: list[str] = []
    seen: set[str] = set()
    for raw in entities:
        if not raw:
            continue
        e = str(raw).strip()
        if not e:
            continue

        low = e.casefold()
        e = canonical_map.get(low, e)
        key = e.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)

    return out


def _collection_name(namespace: str) -> str:
    """Build ChromaDB collection name from namespace."""
    return f"engram_{namespace}"


def _build_memory(mem_id: str, document: str, metadata: dict[str, Any]) -> EpisodicMemory:
    """Construct EpisodicMemory from ChromaDB result fields."""
    raw_type = metadata.get("memory_type", MemoryType.FACT.value)
    try:
        memory_type = MemoryType(raw_type)
    except ValueError:
        memory_type = MemoryType.FACT

    raw_ts = metadata.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        timestamp = datetime.now(timezone.utc)

    raw_entities = metadata.get("entities", "")
    if raw_entities:
        if raw_entities.startswith("["):
            # JSON array format (new)
            try:
                entities = json.loads(raw_entities)
            except (json.JSONDecodeError, ValueError):
                entities = []
        else:
            # CSV format (backward compat for old data)
            entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
    else:
        entities = []

    entities = _canonicalize_entities(entities)

    # Parse tags (JSON array)
    raw_tags = metadata.get("tags", "[]")
    try:
        tags = json.loads(raw_tags) if raw_tags else []
    except (json.JSONDecodeError, ValueError):
        tags = []

    # Parse expires_at
    raw_expires = metadata.get("expires_at")
    expires_at = None
    if raw_expires:
        try:
            expires_at = datetime.fromisoformat(raw_expires)
        except (ValueError, TypeError):
            pass

    # Parse decay/access fields
    access_count = int(metadata.get("access_count", 0))
    decay_rate = float(metadata.get("decay_rate", 0.1))
    raw_last_accessed = metadata.get("last_accessed")
    last_accessed = None
    if raw_last_accessed:
        try:
            last_accessed = datetime.fromisoformat(raw_last_accessed)
        except (ValueError, TypeError):
            pass

    # Parse consolidation fields
    consolidation_group = metadata.get("consolidation_group")
    consolidated_into = metadata.get("consolidated_into")

    # Parse topic key fields
    topic_key = metadata.get("topic_key")
    revision_count = int(metadata.get("revision_count", 0))

    # Parse feedback fields
    confidence = float(metadata.get("confidence", 1.0))
    negative_count = int(metadata.get("negative_count", 0))

    # Exclude internal fields from extra metadata
    source = metadata.get("source", "")

    _internal = {
        "memory_type", "priority", "timestamp", "entities", "tags", "expires_at",
        "access_count", "last_accessed", "decay_rate",
        "consolidation_group", "consolidated_into",
        "topic_key", "revision_count",
        "confidence", "negative_count", "source",
    }
    extra = {k: v for k, v in metadata.items() if k not in _internal}

    return EpisodicMemory(
        id=mem_id,
        content=document,
        memory_type=memory_type,
        priority=int(metadata.get("priority", 5)),
        metadata=extra,
        entities=entities,
        tags=tags,
        timestamp=timestamp,
        expires_at=expires_at,
        access_count=access_count,
        last_accessed=last_accessed,
        decay_rate=decay_rate,
        consolidation_group=consolidation_group,
        consolidated_into=consolidated_into,
        topic_key=topic_key,
        revision_count=revision_count,
        confidence=confidence,
        negative_count=negative_count,
        source=source,
    )
