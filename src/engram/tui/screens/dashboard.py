"""Dashboard pane — memory stats and recent activity overview."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, DataTable
from textual.widget import Widget

from engram.utils import run_async


class DashboardPane(Widget):
    """Shows memory statistics and recent memories."""

    def __init__(self, episodic_store):
        super().__init__()
        self._episodic = episodic_store

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold]Memory Stats[/bold]", id="stats-header"),
            Static("Loading...", id="stats-content"),
            Static(""),
            Static("[bold]Recent Activity[/bold]"),
            DataTable(id="recent-table"),
        )

    def on_mount(self) -> None:
        # Load stats
        stats = run_async(self._episodic.stats())
        content = f"  Memories: {stats.get('count', 0)}  |  Namespace: {stats.get('namespace', 'default')}"
        if stats.get("embedding_dim"):
            content += f"  |  Embedding: {stats['embedding_dim']}d"
        self.query_one("#stats-content", Static).update(content)

        # Load recent memories into table
        table = self.query_one("#recent-table", DataTable)
        table.add_columns("Time", "Type", "Content")
        recent = run_async(self._episodic.get_recent(n=10))
        for mem in recent:
            ts = mem.timestamp.strftime("%m-%d %H:%M")
            table.add_row(ts, mem.memory_type.value, mem.content[:60])
