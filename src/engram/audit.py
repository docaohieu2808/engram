"""Audit logger for engram. Writes structured JSONL audit trail for memory operations.

Disabled by default. Enable via config: audit.enabled = true.
Output format: one JSON object per line (JSONL) at audit.path.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("engram")


class AuditLogger:
    """Append-only JSONL audit log for memory operations.

    Each entry records: timestamp, tenant_id, actor, operation, resource_id, details.
    When disabled (default), all methods are no-ops with zero overhead.
    """

    def __init__(self, enabled: bool = False, path: str = "~/.engram/audit.jsonl") -> None:
        self._enabled = enabled
        if enabled:
            self._path = Path(path).expanduser()
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(
        self,
        tenant_id: str,
        actor: str,
        operation: str,
        resource_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit entry. No-op when disabled."""
        if not self._enabled or self._path is None:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "actor": actor,
            "operation": operation,
            "resource_id": resource_id,
            "details": details or {},
        }

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("audit: failed to write entry to %s", self._path)


# Module-level singleton; replaced by setup_audit() at startup
_audit: AuditLogger = AuditLogger(enabled=False)


def setup_audit(enabled: bool, path: str) -> AuditLogger:
    """Create and register the global audit logger. Call once at startup."""
    global _audit
    _audit = AuditLogger(enabled=enabled, path=path)
    return _audit


def get_audit() -> AuditLogger:
    """Return the global audit logger (may be a no-op instance)."""
    return _audit
