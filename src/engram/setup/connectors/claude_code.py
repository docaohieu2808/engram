"""Connector for Claude Code (Anthropic) — detects and configures MCP server + rules."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engram.setup.connectors import _ALL_CONNECTORS
from engram.setup.connectors.base import ConfigureResult, DetectionResult
from engram.setup.connectors.mcp_json_mixin import McpJsonConnector

# Claude Code stores MCP config in ~/.claude.json under "mcpServers"
_CLAUDE_DIR = Path.home() / ".claude"
_MCP_CONFIG_PATH = Path.home() / ".claude.json"
_RULES_DIR = _CLAUDE_DIR / "rules"
_RULES_PATH = _RULES_DIR / "engram.md"

# Auto-recall rules injected into Claude Code's rules directory
_RULES_CONTENT = """\
# Engram Memory — Persistent AI Brain (MANDATORY)

## Absolute Rules
1. SESSION START: Call `engram_session_start()` then `engram_session_context()` to load context from previous sessions. NEVER start working without recalling context.
2. BEFORE ASKING USER: Call `engram_recall(query)` to check if information is already stored. Do NOT ask what is already known.
3. AFTER EVERY DECISION: Call `engram_remember(content, memory_type="decision", priority=7)` immediately.
4. AFTER EVERY BUG FIX: Call `engram_remember(content, memory_type="error", priority=7)` to avoid repeating mistakes.
5. SESSION END: Call `engram_session_summary(goal, discoveries, accomplished)` to save state.

## Auto-save (no user request needed)
| Event | Action |
|-------|--------|
| User states preference | `engram_remember(content, memory_type="preference", priority=6)` |
| Tech stack decision | `engram_remember(content, memory_type="decision", priority=8)` |
| Bug fix | `engram_remember(content, memory_type="error", priority=7)` |
| TODO mentioned | `engram_remember(content, memory_type="todo", priority=5)` |
| Important fact | `engram_remember(content, memory_type="fact", priority=6)` |
| Workflow/process | `engram_remember(content, memory_type="workflow", priority=6)` |
| Lesson learned | `engram_remember(content, memory_type="lesson", priority=7)` |

## Session workflow
- Session start → `engram_session_start()` → `engram_session_context(limit=5)`
- Need context → `engram_recall(query)` or `engram_ask(query)`
- Need reasoning → `engram_think(question)`
- Decision made → `engram_remember(content, memory_type="decision", priority=7)`
- Session end → `engram_session_summary(goal="...", discoveries=[...], accomplished=[...])`

## Tool selection
- **engram_ask** — Default, auto-routes to recall or think
- **engram_recall** — Direct lookup, semantic search
- **engram_think** — Complex questions needing reasoning (why/how/explain/compare)
- **engram_remember** — Store important information
- **engram_ingest** — Bulk ingest chat messages + extract entities
- **engram_feedback** — Rate memory accuracy to improve retrieval

## NEVER
- NEVER ask user what is already stored in memory
- NEVER skip recall context at session start
- NEVER forget to save important decisions
"""


class ClaudeCodeConnector(McpJsonConnector):
    """Connector for Anthropic Claude Code IDE integration.

    Claude Code uses ~/.claude.json with mcpServers wrapper.
    Each entry needs "type": "stdio" field.
    Also installs auto-recall rules at ~/.claude/rules/engram.md.
    """

    name = "claude-code"
    display_name = "Claude Code"
    tier = 1
    config_path = _MCP_CONFIG_PATH

    def detect(self) -> DetectionResult:
        """Detect Claude Code via binary or ~/.claude/ directory."""
        binary = shutil.which("claude")
        dir_exists = _CLAUDE_DIR.exists()

        if not binary and not dir_exists:
            return DetectionResult(
                installed=False,
                name=self.display_name,
                version=None,
                config_path=None,
                details="Neither 'claude' binary nor ~/.claude/ found",
            )

        version = _get_claude_version(binary)
        details = binary or str(_CLAUDE_DIR)

        return DetectionResult(
            installed=True,
            name=self.display_name,
            version=version,
            config_path=_MCP_CONFIG_PATH,
            details=details,
        )

    def configure(self, dry_run: bool = False) -> ConfigureResult:
        """Configure MCP entry + install auto-recall rules."""
        # 1. MCP config (parent class)
        mcp_result = super().configure(dry_run=dry_run)

        # 2. Auto-recall rules file
        rules_written = _install_rules(dry_run)

        # Combine results
        files = list(mcp_result.files_modified or [])
        if rules_written and not dry_run:
            files.append(_RULES_PATH)

        msg = mcp_result.message
        if rules_written:
            msg += f" + {'Would write' if dry_run else 'Wrote'} rules to {_RULES_PATH}"

        return ConfigureResult(
            success=mcp_result.success,
            message=msg,
            files_modified=files,
            backup_path=mcp_result.backup_path,
        )


def _install_rules(dry_run: bool = False) -> bool:
    """Install auto-recall rules to ~/.claude/rules/engram.md. Returns True if written."""
    if _RULES_PATH.exists():
        existing = _RULES_PATH.read_text(encoding="utf-8")
        if existing == _RULES_CONTENT:
            return False  # already up to date
    if dry_run:
        return True
    _RULES_DIR.mkdir(parents=True, exist_ok=True)
    _RULES_PATH.write_text(_RULES_CONTENT, encoding="utf-8")
    # Remove old name if present
    old_path = _RULES_DIR / "neural-memory.md"
    if old_path.exists():
        old_path.unlink()
    return True


def _get_claude_version(binary: str | None) -> str | None:
    """Try to retrieve Claude Code version string via CLI."""
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout or result.stderr).strip()
        return output or None
    except (OSError, subprocess.TimeoutExpired):
        return None


# Register in global connector registry
_ALL_CONNECTORS.append(ClaudeCodeConnector)
