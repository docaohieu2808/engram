"""Authentication utilities: JWT encode/decode, API key management, FastAPI dependency."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import HTTPException, Request

from engram.auth_models import APIKeyRecord, AuthContext, Role, TokenPayload
from engram.config import Config, load_config
from engram.errors import EngramError, ErrorCode


# --- Module-level config cache (H2) ---

_config: Config | None = None


def init_auth(config: Config) -> None:
    """Cache config at startup so get_auth_context() does not reload it per request.

    H3 fix: validate jwt_secret minimum length (32 chars) when auth is enabled.
    Raises ValueError early at startup so misconfiguration is caught immediately.
    """
    global _config
    if config.auth.enabled and len(config.auth.jwt_secret) < 32:
        raise ValueError(
            "auth.jwt_secret must be at least 32 characters when auth is enabled. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    _config = config


def _get_config() -> Config:
    """Return cached config or load once on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


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
    os.chmod(path, 0o600)  # Restrict to owner-only read/write


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(
    name: str,
    role: Role,
    tenant_id: str = "default",
    expires_days: Optional[int] = None,
) -> tuple[str, APIKeyRecord]:
    """Generate a new API key. Returns (plaintext_key, record). Key is only shown once.

    Args:
        expires_days: If set, key expires after this many days from now.
    """
    key = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = ""
    if expires_days is not None:
        expires_at = (now + timedelta(days=expires_days)).isoformat()
    record = APIKeyRecord(
        key_hash=_hash_key(key),
        name=name,
        role=role,
        tenant_id=tenant_id,
        active=True,
        created_at=now.isoformat(),
        expires_at=expires_at,
    )
    keys = _load_api_keys()
    keys.append(record.model_dump())
    _save_api_keys(keys)
    return key, record


def verify_api_key(key: str) -> Optional[APIKeyRecord]:
    """Look up an API key by its hash. Returns record if active and not expired, else None."""
    key_hash = _hash_key(key)
    now = datetime.now(timezone.utc)
    for entry in _load_api_keys():
        if entry.get("key_hash") != key_hash:
            continue
        if not entry.get("active", True):
            continue
        # Check expiry if set
        expires_at = entry.get("expires_at", "")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if now >= exp_dt:
                    return None  # Key expired
            except (ValueError, TypeError):
                pass  # Malformed expiry — treat as no expiry
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
    """Decode and verify JWT. Returns TokenPayload or None on any failure.

    PyJWT checks exp by default. We also add require=["exp"] so tokens
    without an exp claim are explicitly rejected.
    """
    try:
        data = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
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
_PUBLIC_PATHS = {"/health", "/health/ready", "/api/v1/auth/token", "/auth/token"}

# ADMIN-only POST paths (bare path, stripped of /api/v1 prefix for matching)
_ADMIN_ONLY_PATHS = {"/cleanup", "/summarize"}

# Read-only paths (READER role allowed, bare path)
_READ_PATHS = {"/recall", "/query", "/status", "/health"}

# API v1 prefix to strip before path-based RBAC checks
_API_V1_PREFIX = "/api/v1"


def _normalize_path(path: str) -> str:
    """Strip /api/v1 prefix so RBAC checks work for both prefixed and bare paths.

    H2 fix: RBAC path matching must handle /api/v1/ prefixed routes.
    """
    if path.startswith(_API_V1_PREFIX):
        return path[len(_API_V1_PREFIX):]
    return path


def _require_role(path: str, method: str, role: Role) -> None:
    """Check RBAC. Raises EngramError(FORBIDDEN) if not allowed.

    H2 fix: normalise path by stripping /api/v1 prefix before matching so that
    routes like /api/v1/cleanup correctly map to the _ADMIN_ONLY_PATHS set.
    """
    bare_path = _normalize_path(path)
    if method == "GET":
        # All roles can use GET endpoints
        return
    # POST/DELETE etc.
    if bare_path in _ADMIN_ONLY_PATHS and role != Role.ADMIN:
        raise EngramError(ErrorCode.FORBIDDEN, "Admin role required")
    if role == Role.READER:
        raise EngramError(ErrorCode.FORBIDDEN, "Reader role cannot write")


async def get_auth_context(request: Request) -> AuthContext:
    """FastAPI dependency: extract and validate auth from request headers.

    When auth is disabled (config.auth.enabled=False), returns default AuthContext.
    Checks Authorization: Bearer <jwt> first, then X-API-Key header.
    Uses module-level cached config to avoid per-request reload (H2).
    """
    config = _get_config()
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
        try:
            _require_role(path, method, ctx.role)
        except EngramError:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return ctx

    # Try API key
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        record = verify_api_key(api_key)
        if record is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        ctx = AuthContext(tenant_id=record.tenant_id, role=record.role)
        try:
            _require_role(path, method, ctx.role)
        except EngramError:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return ctx

    # No credentials provided
    raise HTTPException(status_code=401, detail="Authentication required")
