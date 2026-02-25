"""Sessions pane — session history with detail view."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static
from textual.widget import Widget


class SessionsPane(Widget):
    """Shows session history from SessionStore."""

    def __init__(self, session_store):
        super().__init__()
        self._sessions = session_store
        self._session_list = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            DataTable(id="sessions-table"),
            Static("", id="session-detail"),
        )

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("ID", "Started", "Goal", "Items")
        table.cursor_type = "row"
        if self._sessions is None:
            return
        self._session_list = self._sessions.get_recent(n=20)
        for s in self._session_list:
            table.add_row(
                s.id[:8],
                s.started_at[:16],
                (s.goal or "no goal")[:40],
                str(len(s.accomplished)),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._session_list):
            s = self._session_list[idx]
            lines = [
                f"[bold]Session {s.id[:8]}[/bold]",
                f"Started: {s.started_at}",
                f"Ended: {s.ended_at or 'active'}",
                f"Goal: {s.goal or 'none'}",
                "",
                "[bold]Accomplished:[/bold]",
            ]
            for a in s.accomplished:
                lines.append(f"  \u2713 {a}")
            if s.discoveries:
                lines.append("\n[bold]Discoveries:[/bold]")
                for d in s.discoveries:
                    lines.append(f"  \u2192 {d}")
            if s.files_touched:
                lines.append(f"\n[bold]Files:[/bold] {', '.join(s.files_touched)}")
            self.query_one("#session-detail", Static).update("\n".join(lines))
