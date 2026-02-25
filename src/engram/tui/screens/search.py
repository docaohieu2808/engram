"""Search pane — input field with results list and detail drill-down."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, DataTable, Static
from textual.widget import Widget

from engram.utils import run_async


class SearchPane(Widget):
    """Search memories with results and detail view."""

    def __init__(self, episodic_store):
        super().__init__()
        self._episodic = episodic_store
        self._results = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            Input(placeholder="Search memories... (press Enter)", id="search-input"),
            DataTable(id="search-results"),
            Static("", id="search-detail"),
        )

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.add_columns("ID", "Time", "Type", "Content")
        table.cursor_type = "row"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            return
        self._results = run_async(self._episodic.search(event.value, limit=20))
        table = self.query_one("#search-results", DataTable)
        table.clear()
        for mem in self._results:
            ts = mem.timestamp.strftime("%m-%d %H:%M")
            table.add_row(mem.id[:8], ts, mem.memory_type.value, mem.content[:80])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._results):
            mem = self._results[idx]
            detail = (
                f"[bold]ID:[/bold] {mem.id}\n"
                f"[bold]Type:[/bold] {mem.memory_type.value}  [bold]Priority:[/bold] {mem.priority}\n"
                f"[bold]Time:[/bold] {mem.timestamp.isoformat()}\n"
                f"[bold]Tags:[/bold] {', '.join(mem.tags) or 'none'}\n"
                f"[bold]Entities:[/bold] {', '.join(mem.entities) or 'none'}\n"
                f"[bold]Access:[/bold] {mem.access_count}\n\n"
                f"{mem.content}"
            )
            self.query_one("#search-detail", Static).update(detail)
