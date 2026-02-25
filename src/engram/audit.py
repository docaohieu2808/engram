"""Audit logger for engram. Writes structured JSONL audit trail for memory operations.

Disabled by default. Enable via config: audit.enabled = true.
Output format: one JSON object per line (JSONL) at audit.path.

Supports two log types:
- General operations (remember, recall, think)
- Modification trail (remember, forget, modify) with before/after values
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("engram")

# Modification types for structured audit trail
MODIFICATION_TYPES = {
    "memory_create",    # New memory stored
    "memory_delete",    # Memory deleted
    "memory_update",    # Memory content/metadata updated (topic-key upsert)
    "metadata_update",  # Metadata-only change (feedback, consolidation)
    "config_change",    # Configuration modified
    "batch_create",     # Batch memory creation
    "cleanup_expired",  # Expired memories removed
}


class AuditLogger:
    """Append-only JSONL audit log for memory operations.

    Each entry records: timestamp, tenant_id, actor, operation, resource_id, details.
    Modification entries additionally include: mod_type, before_value, after_value, reversible.
    When disabled (default), all methods are no-ops with zero overhead.
    """

    def __init__(self, enabled: bool = False, path: str = "~/.engram/audit.jsonl") -> None:
        self._enabled = enabled
        self._file = None
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
            if self._file is None or self._file.closed:
                self._file = open(self._path, "a", encoding="utf-8")
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except OSError:
            logger.warning("audit: failed to write entry to %s", self._path)

    def log_modification(
        self,
        tenant_id: str,
        actor: str,
        mod_type: str,
        resource_id: str = "",
        before_value: Any = None,
        after_value: Any = None,
        reversible: bool = True,
        description: str = "",
    ) -> None:
        """Log a structured modification entry with before/after values.

        For creator oversight: every memory mutation is traceable.
        No-op when disabled.
        """
        if not self._enabled or self._path is None:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "actor": actor,
            "operation": "modification",
            "mod_type": mod_type,
            "resource_id": resource_id,
            "description": description,
            "before_value": _safe_serialize(before_value),
            "after_value": _safe_serialize(after_value),
            "reversible": reversible,
        }

        try:
            if self._file is None or self._file.closed:
                self._file = open(self._path, "a", encoding="utf-8")
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except OSError:
            logger.warning("audit: failed to write modification entry to %s", self._path)

    def read_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Read the last N entries from the audit log. Returns newest first."""
        if not self._enabled or self._path is None or not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            recent = lines[-n:] if len(lines) > n else lines
            recent.reverse()
            return [json.loads(line) for line in recent if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []


def _safe_serialize(value: Any, max_len: int = 5000) -> Any:
    """Safely serialize a value for audit log, truncating large strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value[:max_len] if len(value) > max_len else value
    if isinstance(value, dict):
        return {k: _safe_serialize(v, max_len) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v, max_len) for v in value]
    return value


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
