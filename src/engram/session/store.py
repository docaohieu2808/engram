"""Session lifecycle management. Sessions stored as JSON files in ~/.engram/sessions/."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Session:
    """A tracked memory session."""
    id: str
    started_at: str
    ended_at: str | None = None
    goal: str | None = None
    discoveries: list[str] = field(default_factory=list)
    accomplished: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    namespace: str = "default"


class SessionStore:
    """JSON-file backed session store under ~/.engram/sessions/."""

    def __init__(self, sessions_dir: str = "~/.engram/sessions"):
        self._dir = Path(sessions_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._active_path = self._dir / "active.json"

    def start(self, namespace: str = "default") -> Session:
        """Create a new session, persist as active.json."""
        session = Session(
            id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat(),
            namespace=namespace,
        )
        self._active_path.write_text(json.dumps(asdict(session)))
        return session

    def get_active(self) -> Session | None:
        """Return the currently active session or None."""
        if not self._active_path.exists():
            return None
        try:
            data = json.loads(self._active_path.read_text())
            return Session(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def get_active_id(self) -> str | None:
        """Return active session ID or None (for tagging memories)."""
        s = self.get_active()
        return s.id if s else None

    def end(
        self,
        goal: str | None = None,
        discoveries: list[str] | None = None,
        accomplished: list[str] | None = None,
        files: list[str] | None = None,
    ) -> Session | None:
        """Finalize the active session, archive it, return it."""
        session = self.get_active()
        if not session:
            return None
        session.ended_at = datetime.now(timezone.utc).isoformat()
        if goal:
            session.goal = goal
        if discoveries:
            session.discoveries = discoveries
        if accomplished:
            session.accomplished = accomplished
        if files:
            session.files_touched = files
        # Archive
        archive_path = self._dir / f"{session.id}.json"
        archive_path.write_text(json.dumps(asdict(session)))
        self._active_path.unlink(missing_ok=True)
        return session

    def get_recent(self, n: int = 5) -> list[Session]:
        """Return N most recent archived sessions sorted by started_at desc."""
        sessions = []
        files = sorted(
            (f for f in self._dir.glob("*.json") if f.name != "active.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[:n]:
            try:
                sessions.append(Session(**json.loads(f.read_text())))
            except (json.JSONDecodeError, TypeError):
                pass
        return sessions
