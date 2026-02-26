"""WebSocket message protocol models — commands, responses, errors, events."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WSCommand(BaseModel):
    """Client -> Server command message."""

    id: str = ""
    type: str  # remember|recall|think|feedback|query|ingest|status
    payload: dict[str, Any] = Field(default_factory=dict)


class WSResponse(BaseModel):
    """Server -> Client success response."""

    id: str = ""
    type: str = "response"
    status: str = "ok"
    data: dict[str, Any] = Field(default_factory=dict)


class WSError(BaseModel):
    """Server -> Client error response."""

    id: str = ""
    type: str = "error"
    code: str = "INTERNAL_ERROR"
    message: str = ""


class WSEvent(BaseModel):
    """Server -> Client push notification for memory changes."""

    type: str = "event"
    event: str  # memory_created|memory_updated|memory_deleted|feedback_recorded
    tenant_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
