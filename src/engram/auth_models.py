"""Auth models for engram HTTP API — roles, API key records, token payloads."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    ADMIN = "admin"    # Full access including cleanup/summarize
    AGENT = "agent"    # Read + write memories
    READER = "reader"  # Read only


class APIKeyRecord(BaseModel):
    """Stored record for a hashed API key."""
    key_hash: str
    name: str
    role: Role
    tenant_id: str = "default"
    active: bool = True
    created_at: str = ""   # ISO timestamp when key was created
    expires_at: str = ""   # ISO timestamp when key expires (empty = no expiry)


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str          # subject: user/agent name
    role: Role
    tenant_id: str = "default"
    exp: int          # UNIX expiry timestamp


class AuthContext(BaseModel):
    """Injected into route handlers via FastAPI Depends."""
    tenant_id: str = "default"
    role: Role = Role.ADMIN
