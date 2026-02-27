"""HTTP middleware for engram server."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from engram.logging_setup import correlation_id
from engram.rate_limiter import RateLimiter

logger = logging.getLogger("engram")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Read or generate X-Correlation-ID, set contextvar, echo in response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        correlation_id.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant sliding-window rate limiting. Skips when rate_limiter is None."""

    def __init__(self, app: Any, rate_limiter: RateLimiter | None = None, jwt_secret: str = "") -> None:  # noqa: B107 — empty string default is not a credential
        super().__init__(app)
        self._limiter = rate_limiter
        self._jwt_secret = jwt_secret

    def _extract_tenant_id(self, request: Request) -> str:
        """Extract tenant_id from JWT bearer token with signature verification.

        When jwt_secret is configured, decodes the token with full verification so
        an attacker cannot forge an arbitrary tenant_id to bypass rate limits.
        Falls back to client IP when no secret is configured or token is missing/invalid.
        Never reads X-Tenant-ID header — that header is unauthenticated and spoofable.
        """
        import jwt as _jwt

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and self._jwt_secret:
            token = auth_header[7:]
            try:
                data = _jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
                tenant_id = data.get("tenant_id") or data.get("sub")
                if tenant_id and isinstance(tenant_id, str):
                    return tenant_id
            except Exception as exc:
                logger.debug("rate_limit: JWT decode failed, falling back to IP: %s", exc)

        client_host = request.client.host if request.client else "anonymous"
        return client_host or "anonymous"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self._limiter is None:
            return await call_next(request)

        # C1 fix: use authenticated tenant from JWT, not spoofable X-Tenant-ID header
        tenant_id = self._extract_tenant_id(request)
        allowed, remaining, reset_at = await self._limiter.check(tenant_id)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "Too many requests"}},
                headers={
                    "X-RateLimit-Limit": str(self._limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(reset_at - int(time.time())),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response
