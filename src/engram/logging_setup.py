"""Structured logging setup for engram.

Provides JSON-line formatter, correlation ID context var, and a
`setup_logging()` function called at app startup.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.config import Config

# Context variable for per-request correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class _CorrelationIdFilter(logging.Filter):
    """Inject current correlation_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get("")  # type: ignore[attr-defined]
        return True


class StructuredFormatter(logging.Formatter):
    """Emit JSON log lines for machine-readable structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "")
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
        }
        if cid:
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config: "Config") -> None:
    """Configure root logger based on config.logging settings.

    When config.logging.format == 'json', use StructuredFormatter.
    Otherwise fall back to the plain text format used before.
    """
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level.upper(), logging.WARNING)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(_CorrelationIdFilter())

    if log_cfg.format.lower() == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("LiteLLM", "litellm", "chromadb", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current context."""
    cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid
