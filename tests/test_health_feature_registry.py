"""Tests for the health feature registry covering all ~350 features."""

from __future__ import annotations

import pytest

from engram.config import Config
from engram.health.feature_registry import FeatureEntry, build_full_registry


@pytest.fixture
def default_cfg():
    return Config()


@pytest.fixture
def registry(default_cfg):
    return build_full_registry(default_cfg)


# --- Count tests ---

def test_registry_total_count(registry):
    """Registry must contain at least 300 features (allow minor variance)."""
    assert len(registry) >= 300, f"Expected >=300 entries, got {len(registry)}"


def test_registry_all_categories_present(registry):
    """All 12 categories must be represented."""
    expected = {
        "Config Boolean Flags",
        "Config Parameters",
        "CLI Commands",
        "HTTP Endpoints",
        "MCP Tools",
        "WebSocket",
        "Pipeline Stages",
        "Algorithms",
        "Middleware",
        "Integrations",
        "Data Model Values",
        "Runtime Features",
    }
    actual = {e.category for e in registry}
    missing = expected - actual
    assert not missing, f"Missing categories: {missing}"


def test_registry_boolean_flags_count(registry):
    """Boolean flags category must have exactly 24 entries."""
    flags = [e for e in registry if e.category == "Config Boolean Flags"]
    assert len(flags) == 24, f"Expected 24 boolean flags, got {len(flags)}"


def test_registry_cli_commands_count(registry):
    """CLI Commands category must have at least 40 entries."""
    cmds = [e for e in registry if e.category == "CLI Commands"]
    assert len(cmds) >= 40, f"Expected >=40 CLI commands, got {len(cmds)}"


def test_registry_http_endpoints_count(registry):
    """HTTP Endpoints category must have at least 30 entries."""
    eps = [e for e in registry if e.category == "HTTP Endpoints"]
    assert len(eps) >= 30, f"Expected >=30 HTTP endpoints, got {len(eps)}"


def test_registry_mcp_tools_count(registry):
    """MCP Tools category must have at least 18 entries."""
    tools = [e for e in registry if e.category == "MCP Tools"]
    assert len(tools) >= 18, f"Expected >=18 MCP tools, got {len(tools)}"


# --- FeatureEntry structure tests ---

def test_registry_entries_are_feature_entry_instances(registry):
    """All entries must be FeatureEntry dataclasses."""
    for e in registry:
        assert isinstance(e, FeatureEntry), f"Not a FeatureEntry: {e!r}"


def test_registry_entries_have_non_empty_name(registry):
    """Every entry must have a non-empty name."""
    empty_names = [e for e in registry if not e.name]
    assert not empty_names, f"{len(empty_names)} entries have empty name"


def test_registry_entries_have_valid_status(registry):
    """Boolean flags must have 'enabled'/'disabled' status; others have non-empty status."""
    for e in registry:
        assert e.status, f"Entry {e.name!r} has empty status"
        if e.category == "Config Boolean Flags":
            assert e.status in ("enabled", "disabled"), (
                f"Flag {e.name!r} has unexpected status {e.status!r}"
            )


# --- Filter tests ---

def test_registry_filter_by_category(default_cfg):
    """Filtering by category returns only matching entries."""
    full = build_full_registry(default_cfg)
    cli_entries = [e for e in full if "CLI" in e.category]
    assert cli_entries, "No CLI entries found"
    for e in cli_entries:
        assert "CLI" in e.category


def test_registry_grep_filter_redis(registry):
    """Text search for 'redis' returns relevant entries."""
    lo = "redis"
    matches = [
        e for e in registry
        if lo in e.name.lower() or lo in e.config_path.lower() or lo in e.env_var.lower()
    ]
    assert len(matches) >= 3, f"Expected >=3 redis-related entries, got {len(matches)}"


def test_registry_grep_filter_no_match(registry):
    """Text search for nonsense string returns empty list."""
    lo = "xyzzy_no_match_42"
    matches = [
        e for e in registry
        if lo in e.name.lower() or lo in e.config_path.lower() or lo in e.env_var.lower()
    ]
    assert matches == []


# --- Config reflection tests ---

def test_registry_params_reflect_config_values(default_cfg):
    """Config parameter entries must show actual config values (not None/empty)."""
    registry = build_full_registry(default_cfg)
    params = [e for e in registry if e.category == "Config Parameters"]
    assert params, "No config parameter entries found"
    # At minimum the episodic.path param should show a real path
    path_entries = [e for e in params if e.config_path == "episodic.path"]
    assert path_entries, "No episodic.path entry found"
    assert "engram" in path_entries[0].status.lower(), (
        f"Unexpected value for episodic.path: {path_entries[0].status!r}"
    )


def test_registry_boolean_flag_default_state(default_cfg):
    """Validate a sample of known default flag states."""
    registry = build_full_registry(default_cfg)
    flags = {e.name: e.status for e in registry if e.category == "Config Boolean Flags"}

    # These are enabled by default in Config
    assert flags.get("Memory Decay") == "enabled"
    assert flags.get("FTS5 Full-Text Search") == "enabled"
    assert flags.get("Recall Pipeline") == "enabled"

    # These are disabled by default
    assert flags.get("Redis Cache") == "disabled"
    assert flags.get("Authentication/JWT") == "disabled"
    assert flags.get("OpenTelemetry") == "disabled"


def test_registry_websocket_count(registry):
    """WebSocket category must have 11 entries (7 commands + 4 events)."""
    ws = [e for e in registry if e.category == "WebSocket"]
    assert len(ws) == 11, f"Expected 11 WebSocket entries, got {len(ws)}"


def test_registry_pipeline_stages_count(registry):
    """Pipeline Stages category must have exactly 13 entries."""
    stages = [e for e in registry if e.category == "Pipeline Stages"]
    assert len(stages) == 13, f"Expected 13 pipeline stages, got {len(stages)}"


def test_registry_data_model_values_count(registry):
    """Data Model Values category must have exactly 37 entries."""
    dmv = [e for e in registry if e.category == "Data Model Values"]
    assert len(dmv) == 37, f"Expected 37 data model values, got {len(dmv)}"
