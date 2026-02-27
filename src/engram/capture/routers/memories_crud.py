"""Routes: GET/PUT/DELETE /memories/{id}, GET /memories,
GET /memories/export, POST /memories/bulk-delete.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from engram.auth import get_auth_context
from engram.auth_models import AuthContext
from engram.errors import EngramError, ErrorCode
from engram.models import MemoryType
from engram.capture.server_helpers import require_admin, resolve_episodic, serialize_memory

router = APIRouter()


class BulkDeleteRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=1000)


@router.get("/memories")
async def list_memories(
    search: Optional[str] = None,
    memory_type: Optional[str] = None,
    priority_min: int = 1,
    priority_max: int = 10,
    confidence_min: float = 0.0,
    confidence_max: float = 1.0,
    tags: Optional[str] = None,
    offset: int = 0,
    limit: int = 20,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """List memories with pagination and filters."""
    state = request.app.state
    ep = resolve_episodic(state, auth)
    limit = min(limit, 100)
    offset = min(offset, 500)
    fetch_limit = min((offset + limit) * 3, 1000)
    if search:
        raw = await ep.search(search, limit=fetch_limit)
    else:
        raw = await ep.get_recent(n=fetch_limit)

    filtered = []
    mt_filter = set(memory_type.split(",")) if memory_type else None
    tag_filter = set(t.strip() for t in tags.split(",")) if tags else None
    for m in raw:
        mt_val = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
        if mt_filter and mt_val not in mt_filter:
            continue
        if not (priority_min <= m.priority <= priority_max):
            continue
        if not (confidence_min <= m.confidence <= confidence_max):
            continue
        if tag_filter and not tag_filter.intersection(set(m.tags)):
            continue
        filtered.append(m)

    paginated = filtered[offset:offset + limit]
    return {
        "status": "ok",
        "memories": [serialize_memory(m) for m in paginated],
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
    }


@router.get("/memories/export")
async def export_memories(
    memory_type: Optional[str] = None,
    limit: int = 1000,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Export memories as JSON."""
    state = request.app.state
    ep = resolve_episodic(state, auth)
    raw = await ep.get_recent(n=min(limit, 1000))
    if memory_type:
        mt_set = set(memory_type.split(","))
        raw = [m for m in raw if (m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)) in mt_set]
    return {"status": "ok", "memories": [serialize_memory(m) for m in raw], "count": len(raw)}


@router.post("/memories/bulk-delete")
async def bulk_delete_memories(
    body: BulkDeleteRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Delete multiple memories by IDs."""
    require_admin(auth)
    state = request.app.state
    ep = resolve_episodic(state, auth)
    deleted = []
    for mid in body.ids:
        if await ep.delete(mid):
            deleted.append(mid)
    return {"status": "ok", "deleted": deleted, "count": len(deleted)}


@router.get("/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Get a single memory by ID."""
    state = request.app.state
    ep = resolve_episodic(state, auth)
    mem = await ep.get(memory_id)
    if not mem:
        raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found")
    return {"status": "ok", "memory": serialize_memory(mem)}


@router.put("/memories/{memory_id}")
async def update_memory(
    memory_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Update memory fields (content, type, priority, tags, expires_at)."""
    state = request.app.state
    ep = resolve_episodic(state, auth)
    mem = await ep.get(memory_id)
    if not mem:
        raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found")

    body = await request.json()
    meta_update: dict[str, Any] = {}
    if "memory_type" in body:
        try:
            MemoryType(body["memory_type"])
        except ValueError:
            raise EngramError(ErrorCode.VALIDATION_ERROR, f"Invalid memory_type: {body['memory_type']}")
        meta_update["memory_type"] = body["memory_type"]
    if "priority" in body:
        p = int(body["priority"])
        if not 1 <= p <= 10:
            raise EngramError(ErrorCode.VALIDATION_ERROR, "Priority must be 1-10")
        meta_update["priority"] = p
    if "tags" in body:
        meta_update["tags"] = ",".join(body["tags"]) if body["tags"] else ""
    if "entities" in body:
        meta_update["entities"] = ",".join(body["entities"]) if body["entities"] else ""
    if "expires_at" in body:
        meta_update["expires_at"] = body["expires_at"] or ""

    if meta_update:
        ok = await ep.update_metadata(memory_id, meta_update)
        if not ok:
            raise EngramError(ErrorCode.INTERNAL, "Failed to update memory")

    updated = await ep.get(memory_id)
    return {"status": "ok", "memory": serialize_memory(updated) if updated else None}


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Delete a single memory. S-H1: ADMIN role required."""
    require_admin(auth)
    state = request.app.state
    ep = resolve_episodic(state, auth)
    ok = await ep.delete(memory_id)
    if not ok:
        raise EngramError(ErrorCode.NOT_FOUND, f"Memory {memory_id} not found or delete failed")
    return {"status": "ok", "deleted": memory_id}
