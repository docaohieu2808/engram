"""Configuration system for engram. YAML-based with env var expansion and env var overlay."""

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
    namespace: str = "default"


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


class HooksConfig(BaseModel):
    """Webhook URLs fired after memory operations (fire-and-forget)."""
    on_remember: str | None = None  # POST {id, content, memory_type} after remember()
    on_think: str | None = None     # POST {question, answer} after think()


class LoggingConfig(BaseModel):
    """Logging configuration."""
    format: str = "text"   # "text" or "json"
    level: str = "WARNING"


class SecurityConfig(BaseModel):
    """Security configuration."""
    max_content_length: int = 10240  # bytes; default 10KB


class Config(BaseModel):
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


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


# Mapping of ENGRAM_* env var suffixes to (section, field) tuples.
# Extend this table when adding new config fields.
_ENV_VAR_MAP: dict[str, tuple[str, str]] = {
    "SERVE_PORT": ("serve", "port"),
    "SERVE_HOST": ("serve", "host"),
    "LLM_MODEL": ("llm", "model"),
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_API_KEY": ("llm", "api_key"),
    "EPISODIC_NAMESPACE": ("episodic", "namespace"),
    "EPISODIC_PATH": ("episodic", "path"),
    "EPISODIC_PROVIDER": ("episodic", "provider"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "EMBEDDING_PROVIDER": ("embedding", "provider"),
    "SEMANTIC_PATH": ("semantic", "path"),
    "LOG_FORMAT": ("logging", "format"),
    "LOG_LEVEL": ("logging", "level"),
    "SECURITY_MAX_CONTENT_LENGTH": ("security", "max_content_length"),
}

# Section model classes for type inference
_SECTION_MODELS: dict[str, type[BaseModel]] = {}


def _get_section_models() -> dict[str, type[BaseModel]]:
    """Lazily build section model map after all classes are defined."""
    return {
        "serve": ServeConfig,
        "llm": LLMConfig,
        "episodic": EpisodicConfig,
        "embedding": EmbeddingConfig,
        "semantic": SemanticConfig,
        "logging": LoggingConfig,
        "security": SecurityConfig,
        "capture": CaptureConfig,
        "hooks": HooksConfig,
    }


def _apply_env_overlay(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ENGRAM_* environment variables on top of YAML data dict.

    Converts values to the correct type based on Pydantic field annotations.
    Secret fields (api_key) are applied but never logged.
    """
    section_models = _get_section_models()

    for env_suffix, (section, field) in _ENV_VAR_MAP.items():
        env_key = f"ENGRAM_{env_suffix}"
        raw_val = os.environ.get(env_key)
        if raw_val is None:
            continue

        # Determine target type from Pydantic model annotation
        model_cls = section_models.get(section)
        target_type: type = str
        if model_cls is not None:
            field_info = model_cls.model_fields.get(field)
            if field_info is not None:
                ann = field_info.annotation
                if ann is int:
                    target_type = int
                elif ann is bool:
                    target_type = bool
                elif ann is float:
                    target_type = float

        # Cast value
        try:
            if target_type is bool:
                typed_val: Any = raw_val.lower() in ("1", "true", "yes")
            else:
                typed_val = target_type(raw_val)
        except (ValueError, TypeError):
            typed_val = raw_val  # fall back to string; Pydantic will validate

        # Merge into data dict
        if section not in data or not isinstance(data[section], dict):
            data[section] = {}
        data[section][field] = typed_val

    return data


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML, expanding env vars, then applying ENGRAM_* env overlay."""
    config_path = path or get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        data = _expand_env_vars(raw)
    else:
        data = {}
    data = _apply_env_overlay(data)
    return Config(**data)


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
