"""Engram TUI — Interactive terminal interface for memory exploration."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from engram.tui.screens.dashboard import DashboardPane
from engram.tui.screens.search import SearchPane
from engram.tui.screens.recent import RecentPane
from engram.tui.screens.sessions import SessionsPane


class EngramTUI(App):
    """Engram terminal interface for browsing and searching memories."""

    TITLE = "engram"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "switch_tab('dashboard')", "Dashboard", show=False),
        Binding("s", "switch_tab('search')", "Search", show=False),
        Binding("r", "switch_tab('recent')", "Recent", show=False),
        Binding("e", "switch_tab('sessions')", "Sessions", show=False),
    ]
    CSS = """
    TabbedContent {
        height: 1fr;
    }
    """

    def __init__(self, episodic_store, session_store=None):
        super().__init__()
        self._episodic = episodic_store
        self._sessions = session_store

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard [d]", id="dashboard"):
                yield DashboardPane(self._episodic)
            with TabPane("Search [s]", id="search"):
                yield SearchPane(self._episodic)
            with TabPane("Recent [r]", id="recent"):
                yield RecentPane(self._episodic)
            with TabPane("Sessions [e]", id="sessions"):
                yield SessionsPane(self._sessions)
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id
