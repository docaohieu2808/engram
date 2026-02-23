"""Configuration system for engram. YAML-based with env var expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# --- Config Models ---


class EpisodicConfig(BaseModel):
    provider: str = "chromadb"
    path: str = "~/.engram/episodic"


class EmbeddingConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-embedding-001"


class SemanticConfig(BaseModel):
    provider: str = "sqlite"
    path: str = "~/.engram/semantic.db"
    schema_name: str = Field(default="devops", alias="schema")

    model_config = {"populate_by_name": True}


class CaptureConfig(BaseModel):
    enabled: bool = True
    inbox: str = "~/.engram/inbox/"
    poll_interval: int = 5


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini/gemini-2.0-flash"
    api_key: str = "${GEMINI_API_KEY}"


class ServeConfig(BaseModel):
    port: int = 8765
    host: str = "127.0.0.1"


class Config(BaseModel):
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)


# --- Helpers ---

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def get_config_dir() -> Path:
    """Get or create engram config directory."""
    config_dir = Path.home() / ".engram"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    return get_config_dir() / "config.yaml"


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} in strings."""
    if isinstance(data, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), data)
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(v) for v in data]
    return data


def _expand_path(path: str) -> Path:
    """Expand ~ and env vars in path string."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML, expanding env vars."""
    config_path = path or get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        data = _expand_env_vars(raw)
        return Config(**data)
    return Config()


def save_config(config: Config, path: Path | None = None) -> None:
    """Save config to YAML."""
    config_path = path or get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(by_alias=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_config_value(config: Config, key_path: str) -> Any:
    """Get nested config value via dot notation (e.g. 'llm.model')."""
    obj: Any = config
    for part in key_path.split("."):
        if isinstance(obj, BaseModel):
            obj = getattr(obj, part, None)
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def set_config_value(key_path: str, value: str) -> Config:
    """Set config value via dot notation, save, and return updated config."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Navigate and set
    parts = key_path.split(".")
    obj = raw
    for part in parts[:-1]:
        if part not in obj or not isinstance(obj[part], dict):
            obj[part] = {}
        obj = obj[part]
    obj[parts[-1]] = value

    # Save and reload
    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return load_config()
