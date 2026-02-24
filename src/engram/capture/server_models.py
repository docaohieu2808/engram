"""Request/Response Pydantic models for the engram HTTP API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from engram.models import MemoryType


class IngestRequest(BaseModel):
    messages: list[dict[str, Any]]


class RememberRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    memory_type: MemoryType = MemoryType.FACT
    priority: int = Field(default=5, ge=1, le=10)
    entities: list[str] = []
    tags: list[str] = []


class ThinkRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    keyword: str = Field(..., min_length=1)
    node_type: Optional[str] = None


class SummarizeRequest(BaseModel):
    count: int = Field(default=20, ge=1, le=1000)
    save: bool = False


class TokenRequest(BaseModel):
    sub: str
    role: str = "agent"
    tenant_id: str = "default"
    jwt_secret: str  # caller must provide secret to obtain token
