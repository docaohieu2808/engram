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
    decay_rate: float = 0.1
    decay_enabled: bool = True


class EmbeddingConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-embedding-001"


class SemanticConfig(BaseModel):
    provider: str = "sqlite"
    path: str = "~/.engram/semantic.db"
    schema_name: str = Field(default="devops", alias="schema")
    # PostgreSQL settings (used when provider="postgresql")
    dsn: str = "${ENGRAM_SEMANTIC_DSN}"
    pool_min: int = 5
    pool_max: int = 20

    model_config = {"populate_by_name": True}


class OpenClawCaptureConfig(BaseModel):
    """OpenClaw realtime session capture via inotify/watchdog."""
    enabled: bool = True
    sessions_dir: str = "~/.openclaw/agents/main/sessions"


class CaptureConfig(BaseModel):
    enabled: bool = True
    inbox: str = "~/.engram/inbox/"
    poll_interval: int = 5
    openclaw: OpenClawCaptureConfig = Field(default_factory=OpenClawCaptureConfig)


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


class AuthConfig(BaseModel):
    """HTTP API authentication configuration. Disabled by default for backward compat."""
    enabled: bool = False       # Toggle auth (false = no auth required)
    jwt_secret: str = ""        # Required when enabled; min 32 chars recommended
    jwt_expiry_hours: int = 24  # Token lifetime
    admin_secret: str = ""      # Admin bootstrap token for /auth/token endpoint


class TelemetryConfig(BaseModel):
    """OpenTelemetry configuration. Disabled by default; requires telemetry extra."""
    enabled: bool = False
    otlp_endpoint: str = ""
    sample_rate: float = 0.1
    service_name: str = "engram"


class AuditConfig(BaseModel):
    """Audit logging configuration. Disabled by default."""
    enabled: bool = False
    backend: str = "file"
    path: str = "~/.engram/audit.jsonl"


class CacheConfig(BaseModel):
    """Redis cache configuration. Disabled by default."""
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    recall_ttl: int = 300   # seconds; TTL for /recall results
    think_ttl: int = 900    # seconds; TTL for /think results
    query_ttl: int = 300    # seconds; TTL for /query results


class RateLimitConfig(BaseModel):
    """Sliding-window rate limit configuration. Disabled by default."""
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    requests_per_minute: int = 60
    burst: int = 10


class ScoringConfig(BaseModel):
    """Activation-based recall scoring weights. Must sum to ~1.0."""
    similarity_weight: float = 0.5
    retention_weight: float = 0.2
    recency_weight: float = 0.15
    frequency_weight: float = 0.15


class ConsolidationConfig(BaseModel):
    """Memory consolidation settings."""
    enabled: bool = False
    min_cluster_size: int = 3
    similarity_threshold: float = 0.3


class ProviderEntry(BaseModel):
    """Configuration for a single external memory provider."""
    name: str
    type: str  # rest, file, postgres, mcp
    enabled: bool = True
    debug: bool = False
    max_consecutive_errors: int = 5
    timeout_seconds: float = 3.0
    # REST adapter fields
    url: str = ""
    search_endpoint: str = ""
    search_method: str = "POST"
    search_body: str = ""
    result_path: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    # REST auth fields (JWT auto-login)
    auth_login_endpoint: str = ""  # e.g. /api/v1/auth/login
    auth_username: str = ""
    auth_password: str = ""
    # File adapter fields
    path: str = ""
    pattern: str = "*.md"
    # Postgres adapter fields
    dsn: str = ""
    search_query: str = ""
    # MCP adapter fields
    command: str = ""
    tool_name: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class DiscoveryConfig(BaseModel):
    """Auto-discovery configuration for finding external memory services."""
    local: bool = True
    hosts: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)


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
    auth: AuthConfig = Field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    providers: list[ProviderEntry] = Field(default_factory=list)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)


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
    "SEMANTIC_PROVIDER": ("semantic", "provider"),
    "SEMANTIC_DSN": ("semantic", "dsn"),
    "SEMANTIC_POOL_MIN": ("semantic", "pool_min"),
    "SEMANTIC_POOL_MAX": ("semantic", "pool_max"),
    "LOG_FORMAT": ("logging", "format"),
    "LOG_LEVEL": ("logging", "level"),
    "SECURITY_MAX_CONTENT_LENGTH": ("security", "max_content_length"),
    "AUTH_ENABLED": ("auth", "enabled"),
    "AUTH_JWT_SECRET": ("auth", "jwt_secret"),
    "AUTH_JWT_EXPIRY_HOURS": ("auth", "jwt_expiry_hours"),
    "AUTH_ADMIN_SECRET": ("auth", "admin_secret"),
    "TELEMETRY_ENABLED": ("telemetry", "enabled"),
    "TELEMETRY_OTLP_ENDPOINT": ("telemetry", "otlp_endpoint"),
    "TELEMETRY_SAMPLE_RATE": ("telemetry", "sample_rate"),
    "TELEMETRY_SERVICE_NAME": ("telemetry", "service_name"),
    "AUDIT_ENABLED": ("audit", "enabled"),
    "AUDIT_PATH": ("audit", "path"),
    "AUDIT_BACKEND": ("audit", "backend"),
    "CACHE_ENABLED": ("cache", "enabled"),
    "CACHE_REDIS_URL": ("cache", "redis_url"),
    "CACHE_RECALL_TTL": ("cache", "recall_ttl"),
    "CACHE_THINK_TTL": ("cache", "think_ttl"),
    "CACHE_QUERY_TTL": ("cache", "query_ttl"),
    "RATE_LIMIT_ENABLED": ("rate_limit", "enabled"),
    "RATE_LIMIT_REDIS_URL": ("rate_limit", "redis_url"),
    "RATE_LIMIT_REQUESTS_PER_MINUTE": ("rate_limit", "requests_per_minute"),
    "RATE_LIMIT_BURST": ("rate_limit", "burst"),
    "CAPTURE_OPENCLAW_ENABLED": ("capture.openclaw", "enabled"),
    "CAPTURE_OPENCLAW_SESSIONS_DIR": ("capture.openclaw", "sessions_dir"),
    "EPISODIC_DECAY_RATE": ("episodic", "decay_rate"),
    "EPISODIC_DECAY_ENABLED": ("episodic", "decay_enabled"),
    "SCORING_SIMILARITY_WEIGHT": ("scoring", "similarity_weight"),
    "SCORING_RETENTION_WEIGHT": ("scoring", "retention_weight"),
    "SCORING_RECENCY_WEIGHT": ("scoring", "recency_weight"),
    "SCORING_FREQUENCY_WEIGHT": ("scoring", "frequency_weight"),
    "CONSOLIDATION_ENABLED": ("consolidation", "enabled"),
    "CONSOLIDATION_MIN_CLUSTER_SIZE": ("consolidation", "min_cluster_size"),
    "CONSOLIDATION_SIMILARITY_THRESHOLD": ("consolidation", "similarity_threshold"),
}

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
        "auth": AuthConfig,
        "telemetry": TelemetryConfig,
        "audit": AuditConfig,
        "cache": CacheConfig,
        "rate_limit": RateLimitConfig,
        "scoring": ScoringConfig,
        "consolidation": ConsolidationConfig,
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
