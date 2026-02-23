"""Tests for engram config loading, saving, and dot-notation access."""

from __future__ import annotations

import os

import pytest

from engram.config import (
    Config,
    get_config_value,
    load_config,
    save_config,
    set_config_value,
)


def test_load_default_config(tmp_path):
    """Non-existent config path returns Config() defaults."""
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config == Config()


def test_expand_env_vars(tmp_path, monkeypatch):
    """${VAR} in config values is expanded from environment."""
    monkeypatch.setenv("TEST_API_KEY", "my-secret-key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  api_key: ${TEST_API_KEY}\n")
    config = load_config(config_file)
    assert config.llm.api_key == "my-secret-key"


def test_save_and_load_config(tmp_path):
    """Round-trip: save then load returns identical config."""
    config = Config()
    config.llm.model = "openai/gpt-4o"
    config.serve.port = 9999

    config_file = tmp_path / "config.yaml"
    save_config(config, config_file)
    loaded = load_config(config_file)

    assert loaded.llm.model == "openai/gpt-4o"
    assert loaded.serve.port == 9999


def test_get_set_config_value(tmp_path, monkeypatch):
    """Dot-notation get/set reads and writes nested config values."""
    # Point config path to tmp dir so set_config_value doesn't touch ~/.engram
    config_file = tmp_path / "config.yaml"
    monkeypatch.setattr("engram.config.get_config_path", lambda: config_file)

    updated = set_config_value("llm.model", "anthropic/claude-3-opus")
    assert get_config_value(updated, "llm.model") == "anthropic/claude-3-opus"
    assert get_config_value(updated, "serve.port") == 8765  # default unchanged
