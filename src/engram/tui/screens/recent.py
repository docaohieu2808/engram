"""Recent memories pane — chronological list with detail on selection."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static
from textual.widget import Widget

from engram.utils import run_async


class RecentPane(Widget):
    """Shows recent memories sorted by timestamp."""

    def __init__(self, episodic_store):
        super().__init__()
        self._episodic = episodic_store
        self._memories = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            DataTable(id="recent-list"),
            Static("", id="recent-detail"),
        )

    def on_mount(self) -> None:
        self._memories = run_async(self._episodic.get_recent(n=50))
        table = self.query_one("#recent-list", DataTable)
        table.add_columns("ID", "Time", "Type", "Tags", "Content")
        table.cursor_type = "row"
        for mem in self._memories:
            ts = mem.timestamp.strftime("%m-%d %H:%M")
            tags = ",".join(mem.tags[:2]) if mem.tags else ""
            table.add_row(mem.id[:8], ts, mem.memory_type.value, tags, mem.content[:60])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._memories):
            mem = self._memories[idx]
            detail = (
                f"[bold]{mem.id}[/bold]\n"
                f"Type: {mem.memory_type.value}  Priority: {mem.priority}  Access: {mem.access_count}\n"
                f"Tags: {', '.join(mem.tags) or 'none'}  Entities: {', '.join(mem.entities) or 'none'}\n\n"
                f"{mem.content}"
            )
            self.query_one("#recent-detail", Static).update(detail)
