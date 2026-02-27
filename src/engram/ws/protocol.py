"""WebSocket message protocol models — commands, responses, errors, events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Typed payload schemas per command ---

class RememberPayload(BaseModel):
    content: str
    memory_type: str = "fact"
    priority: int = 5
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RecallPayload(BaseModel):
    query: str = ""
    limit: int = 5


class ThinkPayload(BaseModel):
    question: str


class FeedbackPayload(BaseModel):
    memory_id: str
    feedback: str  # "correct" | "incorrect" | "helpful" | "not_helpful"


class QueryPayload(BaseModel):
    keyword: str = ""


class IngestPayload(BaseModel):
    messages: list[dict[str, Any]]


class StatusPayload(BaseModel):
    pass


# Map command type → payload model for validation
PAYLOAD_SCHEMAS: dict[str, type[BaseModel]] = {
    "remember": RememberPayload,
    "recall": RecallPayload,
    "think": ThinkPayload,
    "feedback": FeedbackPayload,
    "query": QueryPayload,
    "ingest": IngestPayload,
    "status": StatusPayload,
}


# --- Wire protocol models ---

class WSCommand(BaseModel):
    """Client -> Server command message."""

    id: str = ""
    type: str  # remember|recall|think|feedback|query|ingest|status
    payload: dict[str, Any] = Field(default_factory=dict)

    def validate_payload(self) -> BaseModel:
        """Validate payload against typed schema for this command type.

        Raises ValueError if command type unknown or payload invalid.
        """
        schema = PAYLOAD_SCHEMAS.get(self.type)
        if schema is None:
            raise ValueError(f"Unknown command type: {self.type}")
        return schema.model_validate(self.payload)


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
