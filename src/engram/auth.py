"""Authentication utilities: JWT encode/decode, API key management, FastAPI dependency."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import jwt
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from engram.auth_models import APIKeyRecord, AuthContext, Role, TokenPayload
from engram.config import load_config


# --- Paths ---

def _api_keys_path() -> Path:
    return Path.home() / ".engram" / "api_keys.json"


# --- API Key Management ---

def _load_api_keys() -> list[dict]:
    path = _api_keys_path()
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_api_keys(keys: list[dict]) -> None:
    path = _api_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(keys, f, indent=2)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(name: str, role: Role, tenant_id: str = "default") -> tuple[str, APIKeyRecord]:
    """Generate a new API key. Returns (plaintext_key, record). Key is only shown once."""
    key = secrets.token_urlsafe(32)
    record = APIKeyRecord(
        key_hash=_hash_key(key),
        name=name,
        role=role,
        tenant_id=tenant_id,
        active=True,
    )
    keys = _load_api_keys()
    keys.append(record.model_dump())
    _save_api_keys(keys)
    return key, record


def verify_api_key(key: str) -> Optional[APIKeyRecord]:
    """Look up an API key by its hash. Returns record if active, else None."""
    key_hash = _hash_key(key)
    for entry in _load_api_keys():
        if entry.get("key_hash") == key_hash and entry.get("active", True):
            return APIKeyRecord(**entry)
    return None


def list_api_keys() -> list[APIKeyRecord]:
    """Return all stored API key records."""
    return [APIKeyRecord(**e) for e in _load_api_keys()]


def revoke_api_key(name: str) -> bool:
    """Deactivate API key by name. Returns True if found and deactivated."""
    keys = _load_api_keys()
    changed = False
    for entry in keys:
        if entry.get("name") == name and entry.get("active", True):
            entry["active"] = False
            changed = True
    if changed:
        _save_api_keys(keys)
    return changed


# --- JWT ---

def create_jwt(payload: TokenPayload, secret: str) -> str:
    """Sign and return a JWT token string."""
    data = {
        "sub": payload.sub,
        "role": payload.role.value,
        "tenant_id": payload.tenant_id,
        "exp": payload.exp,
    }
    return jwt.encode(data, secret, algorithm="HS256")


def verify_jwt(token: str, secret: str) -> Optional[TokenPayload]:
    """Decode and verify JWT. Returns TokenPayload or None on any failure."""
    try:
        data = jwt.decode(token, secret, algorithms=["HS256"])
        return TokenPayload(
            sub=data["sub"],
            role=Role(data["role"]),
            tenant_id=data.get("tenant_id", "default"),
            exp=data["exp"],
        )
    except Exception:
        return None


# --- FastAPI Dependency ---

# Public routes that never require auth
_PUBLIC_PATHS = {"/health", "/api/v1/auth/token", "/auth/token"}

# ADMIN-only POST paths
_ADMIN_ONLY_PATHS = {"/cleanup", "/summarize"}

# Read-only paths (READER role allowed)
_READ_PATHS = {"/recall", "/query", "/status", "/health"}


def _require_role(path: str, method: str, role: Role) -> Optional[JSONResponse]:
    """Check RBAC. Returns 403 JSONResponse if not allowed, else None."""
    if method == "GET":
        # All roles can use GET endpoints
        return None
    # POST/DELETE etc.
    if path in _ADMIN_ONLY_PATHS and role != Role.ADMIN:
        return JSONResponse(status_code=403, content={"detail": "Admin role required"})
    if role == Role.READER:
        return JSONResponse(status_code=403, content={"detail": "Reader role cannot write"})
    return None


async def get_auth_context(request: Request) -> AuthContext:
    """FastAPI dependency: extract and validate auth from request headers.

    When auth is disabled (config.auth.enabled=False), returns default AuthContext.
    Checks Authorization: Bearer <jwt> first, then X-API-Key header.
    """
    config = load_config()
    path = request.url.path
    method = request.method

    # Public paths — always pass through
    if path in _PUBLIC_PATHS:
        return AuthContext()

    # Auth disabled → backward compat, allow everything
    if not config.auth.enabled:
        return AuthContext()

    # Try JWT bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = verify_jwt(token, config.auth.jwt_secret)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        ctx = AuthContext(tenant_id=payload.tenant_id, role=payload.role)
        if _require_role(path, method, ctx.role):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return ctx

    # Try API key
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        record = verify_api_key(api_key)
        if record is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        ctx = AuthContext(tenant_id=record.tenant_id, role=record.role)
        if _require_role(path, method, ctx.role):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return ctx

    # No credentials provided
    raise HTTPException(status_code=401, detail="Authentication required")
