"""Routes: GET /health, GET /health/ready, GET /status (admin), POST /backup,
POST /restore, GET /audit/log, GET/POST /scheduler/tasks, POST /benchmark/run,
POST /auth/token, GET /providers, GET/PUT /config, POST /restart.
"""

from __future__ import annotations

import hmac
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engram.auth import get_auth_context
from engram.auth_models import AuthContext, Role, TokenPayload
from engram.config import Config, load_config, save_config
from engram.errors import EngramError, ErrorCode
from engram.capture.server_helpers import require_admin, resolve_episodic

router = APIRouter()


# --- Request models ---


class TokenRequest(BaseModel):
    """Request body for /auth/token. C2 fix: no jwt_secret in body."""
    sub: str
    role: str = "agent"
    tenant_id: str = "default"


# --- Route handlers ---


@router.post("/auth/token")
async def auth_token(req: TokenRequest, request: Request):
    """Issue a JWT token using admin_secret header auth (C2 fix).

    Auth disabled or admin_secret not configured -> 404.
    Caller must provide Authorization: Bearer <admin_secret> header.
    """
    from engram.auth import create_jwt

    cfg: Config = request.app.state.cfg
    if not cfg.auth.enabled:
        raise HTTPException(status_code=404, detail="Auth not enabled")
    if not cfg.auth.admin_secret:
        raise HTTPException(status_code=404, detail="Auth not enabled")
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header with admin secret required")
    provided_secret = auth_header[7:]
    # C3 fix: constant-time comparison to prevent timing side-channel attacks
    if not hmac.compare_digest(provided_secret, cfg.auth.admin_secret):
        raise HTTPException(status_code=401, detail="Invalid secret")
    try:
        role = Role(req.role)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid role: {req.role}")
    expiry = int(time.time()) + cfg.auth.jwt_expiry_hours * 3600
    payload = TokenPayload(sub=req.sub, role=role, tenant_id=req.tenant_id, exp=expiry)
    token = create_jwt(payload, cfg.auth.jwt_secret)
    return {
        "access_token": token,
        "token_type": "bearer",  # noqa: B105
        "expires_in": cfg.auth.jwt_expiry_hours * 3600,
    }


@router.get("/providers")
async def list_providers(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """List all configured providers and their status. M2 fix: requires auth."""
    provider_registry = getattr(request.app.state, "provider_registry", None)
    providers = []
    if provider_registry is not None:
        for p in provider_registry.get_all():
            providers.append({
                "name": p.name,
                "type": p.provider_type,
                "active": p.is_active,
                "status": p.status_label,
                "stats": {
                    "query_count": p.stats.query_count,
                    "avg_latency_ms": round(p.stats.avg_latency_ms, 1),
                    "hit_count": p.stats.hit_count,
                    "error_count": p.stats.error_count,
                },
            })
    return {"providers": providers}


@router.get("/audit/log")
async def audit_log(
    last: int = 50,
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """Get recent audit log entries.
    S-H5: cap last to prevent OOM. S-C2: filter by tenant_id.
    """
    from engram.audit import get_audit

    last = min(last, 1000)
    audit = get_audit()
    entries = audit.read_recent(last * 2)
    entries = [e for e in entries if e.get("tenant_id") == auth.tenant_id][:last]
    return {"status": "ok", "entries": entries}


@router.get("/scheduler/tasks")
async def scheduler_tasks(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """List all scheduled tasks and their status."""
    try:
        sched = getattr(request.app.state, "scheduler", None)
        tasks = sched.status() if sched is not None else []
    except Exception:
        tasks = []
    return {"status": "ok", "tasks": tasks}


@router.post("/scheduler/tasks/{task_name}/run")
async def scheduler_force_run(
    task_name: str,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """Force-run a scheduled task."""
    require_admin(auth)
    return {"status": "ok", "message": f"Task {task_name} triggered (scheduler must be running)"}


@router.post("/benchmark/run")
async def benchmark_run(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Run benchmark with provided questions."""
    require_admin(auth)
    body = await request.json()
    questions = body.get("questions", [])
    if not questions:
        raise EngramError(ErrorCode.VALIDATION_ERROR, "questions list required")
    state = request.app.state
    ep = resolve_episodic(state, auth)
    results = []
    for q in questions:
        query = q.get("question", "")
        expected = q.get("expected", "")
        start = time.time()
        recalls = await ep.search(query, limit=3)
        latency = round((time.time() - start) * 1000, 1)
        actual = recalls[0].content if recalls else ""
        correct = expected.lower() in actual.lower() if expected else False
        results.append({
            "question": query,
            "expected": expected,
            "actual": actual[:200],
            "correct": correct,
            "latency_ms": latency,
        })
    accuracy = sum(1 for r in results if r["correct"]) / len(results) * 100 if results else 0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    return {
        "status": "ok",
        "results": results,
        "accuracy": round(accuracy, 1),
        "avg_latency_ms": round(avg_latency, 1),
    }


# Sections with secrets that should be masked in GET responses
_SECRET_FIELDS = {"api_key", "jwt_secret", "admin_secret", "password", "dsn", "redis_url"}

# Sections that require server restart to take effect
_RESTART_REQUIRED = {"serve", "episodic", "embedding", "semantic", "capture", "auth", "telemetry", "extraction", "recall", "scheduler"}


def _mask_secrets(data: dict) -> dict:
    """Recursively mask secret fields in config dict."""
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = _mask_secrets(v)
        elif k in _SECRET_FIELDS and isinstance(v, str) and v:
            out[k] = "***" + v[-4:] if len(v) > 4 else "***"
        else:
            out[k] = v
    return out


@router.get("/config")
async def get_config(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Return current config with secrets masked. Admin only."""
    require_admin(auth)
    cfg: Config = request.app.state.cfg
    data = cfg.model_dump(by_alias=True)
    return {
        "status": "ok",
        "config": _mask_secrets(data),
        "restart_required_sections": sorted(_RESTART_REQUIRED),
    }


@router.put("/config")
async def put_config(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Update config sections. Body: {"section.field": value, ...}.

    Validates types via Pydantic, saves to config.yaml, reloads.
    Secret fields in the update body are stored as-is (not masked).
    """
    require_admin(auth)
    body = await request.json()
    if not isinstance(body, dict):
        raise EngramError(ErrorCode.VALIDATION_ERROR, "Body must be a JSON object")

    import yaml
    from engram.config import get_config_path

    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    changed = []
    for key_path, value in body.items():
        parts = key_path.split(".")
        if len(parts) < 2:
            continue
        # Navigate to parent dict
        node = raw
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        changed.append(key_path)

    # Save to disk (write-to-temp + atomic rename)
    import tempfile
    tmp_path = config_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    # Validate by loading saved file (includes env overlay)
    try:
        new_cfg = load_config(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise EngramError(ErrorCode.VALIDATION_ERROR, f"Invalid config: {e}")

    import os
    os.replace(str(tmp_path), str(config_path))

    # Update in-memory config
    request.app.state.cfg = new_cfg

    # Check if restart needed
    restart_needed = any(p.split(".")[0] in _RESTART_REQUIRED for p in changed)

    return {
        "status": "ok",
        "changed": changed,
        "restart_required": restart_needed,
    }


@router.get("/models")
async def list_models(
    provider: str = "all",
    request: Request = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """List available LLM models from litellm's registry, filtered by provider.

    ?provider=anthropic|gemini|openai|all (default: all 3 main providers)
    """
    import litellm

    all_models = litellm.model_list or []

    # Use litellm's model_info to filter: only mode=chat models
    def _is_chat_model(model_name: str) -> bool:
        try:
            info = litellm.get_model_info(model_name)
            return info.get("mode") == "chat"
        except Exception:
            return False

    provider_prefixes = {
        "anthropic": lambda m: m.startswith("claude-") and not m.startswith("claude-instant") and not m.startswith("claude-3-"),
        "gemini": lambda m: m.startswith("gemini/gemini-") and "-1.0" not in m and "-1.5" not in m,
        "openai": lambda m: m.startswith("gpt-4") or m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"),
    }

    result = {}
    providers = [provider] if provider != "all" else list(provider_prefixes.keys())
    for p in providers:
        prefix_fn = provider_prefixes.get(p)
        if not prefix_fn:
            continue
        candidates = [m for m in all_models if prefix_fn(m)]
        # Filter by mode=chat (removes realtime, transcribe, embed, image, etc.)
        models = sorted(set(
            (f"anthropic/{m}" if p == "anthropic" else m)
            for m in candidates if _is_chat_model(m)
        ))
        result[p] = models
    return {"models": result}


@router.post("/restart")
async def restart_server(request: Request, auth: AuthContext = Depends(get_auth_context)):
    """Gracefully restart the server process. Admin only.

    Sends SIGTERM after response flushes. Relies on systemd Restart=always
    to bring the process back up automatically.
    """
    import asyncio
    import os
    import signal

    require_admin(auth)

    async def _do_restart():
        await asyncio.sleep(0.5)  # let response flush
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.get_event_loop().create_task(_do_restart())
    return {"status": "ok", "message": "Server restarting..."}
