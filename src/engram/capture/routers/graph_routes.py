"""Routes: GET/POST/PUT/DELETE /graph/nodes, /graph/edges, GET /graph/data,
GET /feedback/history.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from engram.auth import get_auth_context
from engram.auth_models import AuthContext
from engram.errors import EngramError, ErrorCode
from engram.capture.server_helpers import require_admin, resolve_graph

router = APIRouter()


# --- Request models ---


class CreateNodeRequest(BaseModel):
    type: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class UpdateNodeRequest(BaseModel):
    attributes: dict[str, Any] = Field(default_factory=dict)


class CreateEdgeRequest(BaseModel):
    from_node: str = Field(..., min_length=1)
    to_node: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1)
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class DeleteEdgeRequest(BaseModel):
    key: str = Field(..., min_length=1)


# --- Route handlers ---


@router.get("/graph/data")
async def graph_data(
    request: Request,
    limit: int = 500,
    offset: int = 0,
    auth: AuthContext = Depends(get_auth_context),
):
    """Return paginated nodes and edges in vis-network format. Admin only."""
    require_admin(auth)
    if limit < 1 or limit > 5000:
        raise EngramError(ErrorCode.VALIDATION_ERROR, "limit must be between 1 and 5000")
    if offset < 0:
        raise EngramError(ErrorCode.VALIDATION_ERROR, "offset must be >= 0")

    state = request.app.state
    gr = await resolve_graph(state, auth)

    def _norm_name(name: str) -> str:
        s = (name or "").strip()
        if not s:
            return s
        return s[0].upper() + s[1:]

    def _norm_key(node_type: str, name: str) -> str:
        return f"{node_type}:{_norm_name(name)}"

    def _norm_key_from_key(key: str) -> str:
        if ":" not in key:
            return key
        t, n = key.split(":", 1)
        return _norm_key(t, n)

    nodes = await gr.get_nodes()
    color_map = {
        "Person": "#4CAF50",
        "Technology": "#2196F3",
        "Project": "#FF9800",
        "Service": "#9C27B0",
    }

    # Merge case-variant nodes for graph visualization (e.g. engram/Engram)
    merged_nodes: dict[str, dict[str, Any]] = {}
    for node in nodes:
        attrs = node.attributes or {}
        normalized_name = _norm_name(node.name)
        node_id = _norm_key(node.type, node.name)
        tooltip = f"{node.type}: {normalized_name}"
        if attrs:
            tooltip += "\n" + "\n".join(f"{k}: {v}" for k, v in attrs.items())

        existing = merged_nodes.get(node_id)
        if existing:
            merged_attrs = dict(existing.get("attributes") or {})
            merged_attrs.update(attrs)
            existing["attributes"] = merged_attrs
            existing["title"] = tooltip
        else:
            merged_nodes[node_id] = {
                "id": node_id,
                "label": normalized_name,
                "group": node.type,
                "color": color_map.get(node.type, "#607D8B"),
                "title": tooltip,
                "attributes": attrs,
            }

    vis_nodes_all = list(merged_nodes.values())

    all_edges = await gr.get_edges()
    vis_edges_all = []
    seen: set[tuple[str, str, str]] = set()
    for edge in all_edges:
        from_key = _norm_key_from_key(edge.from_node)
        to_key = _norm_key_from_key(edge.to_node)
        key = (from_key, to_key, edge.relation)
        if key not in seen:
            seen.add(key)
            vis_edges_all.append({
                "from": from_key,
                "to": to_key,
                "label": edge.relation,
                "arrows": "to",
                "weight": edge.weight,
                "attributes": edge.attributes,
            })

    total_nodes = len(vis_nodes_all)
    total_edges = len(vis_edges_all)
    vis_nodes = vis_nodes_all[offset: offset + limit]
    vis_edges = vis_edges_all[offset: offset + limit]

    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


@router.post("/graph/nodes")
async def create_node(
    body: CreateNodeRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Create a semantic graph node."""
    from engram.models import SemanticNode

    node = SemanticNode(type=body.type, name=body.name, attributes=body.attributes)
    state = request.app.state
    gr = await resolve_graph(state, auth)
    is_new = await gr.add_node(node)
    return {"status": "ok", "key": node.key, "created": is_new}


@router.put("/graph/nodes/{node_key:path}")
async def update_node(
    node_key: str,
    body: UpdateNodeRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Update node attributes."""
    from engram.models import SemanticNode

    state = request.app.state
    gr = await resolve_graph(state, auth)
    nodes = await gr.get_nodes()
    existing = next((n for n in nodes if n.key == node_key), None)
    if not existing:
        raise EngramError(ErrorCode.NOT_FOUND, f"Node {node_key} not found")
    updated = SemanticNode(
        type=existing.type,
        name=existing.name,
        attributes={**existing.attributes, **body.attributes},
    )
    await gr.add_node(updated)
    return {"status": "ok", "key": updated.key}


@router.delete("/graph/nodes/{node_key:path}")
async def delete_node(
    node_key: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Delete a semantic graph node and its connected edges. S-H1: ADMIN role required."""
    require_admin(auth)
    state = request.app.state
    gr = await resolve_graph(state, auth)
    ok = await gr.remove_node(node_key)
    if not ok:
        raise EngramError(ErrorCode.NOT_FOUND, f"Node {node_key} not found")
    return {"status": "ok", "deleted": node_key}


@router.post("/graph/edges")
async def create_edge(
    body: CreateEdgeRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Create a semantic graph edge."""
    from engram.models import SemanticEdge

    edge = SemanticEdge(
        from_node=body.from_node,
        to_node=body.to_node,
        relation=body.relation,
        weight=body.weight,
        attributes=body.attributes,
    )
    state = request.app.state
    gr = await resolve_graph(state, auth)
    is_new = await gr.add_edge(edge)
    return {"status": "ok", "key": edge.key, "created": is_new}


@router.delete("/graph/edges")
async def delete_edge(
    body: DeleteEdgeRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Delete a semantic graph edge by key. S-H1: ADMIN role required."""
    require_admin(auth)
    state = request.app.state
    gr = await resolve_graph(state, auth)
    ok = await gr.remove_edge(body.key)
    if not ok:
        raise EngramError(ErrorCode.NOT_FOUND, f"Edge {body.key} not found")
    return {"status": "ok", "deleted": body.key}


@router.get("/feedback/history")
async def feedback_history(
    last: int = 50,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Get recent feedback entries from audit log.
    S-H5: cap last to prevent OOM. S-C3: filter by tenant_id.
    """
    from engram.audit import get_audit

    last = min(last, 1000)
    audit = get_audit()
    entries = audit.read_recent(last * 3)
    feedback_entries = [
        e for e in entries
        if e.get("operation") == "modification"
        and e.get("mod_type") == "metadata_update"
        and "confidence" in str(e.get("after_value", ""))
        and e.get("tenant_id") == auth.tenant_id
    ]
    return {"status": "ok", "entries": feedback_entries[:last]}
