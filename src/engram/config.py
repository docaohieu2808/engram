"""Configuration system for engram. YAML-based with env var expansion and env var overlay."""

from __future__ import annotations

__all__ = [
    "Config", "load_config", "save_config", "get_config_value", "set_config_value",
    "ExtractionConfig", "RecallConfig", "SchedulerConfig",
]

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
    dedup_enabled: bool = True
    dedup_threshold: float = 0.85  # cosine similarity; 1.0 = identical
    # Backend transport: "embedded" (local PersistentClient) or "http" (remote HttpClient)
    mode: str = "embedded"
    host: str = "localhost"
    port: int = 8000
    fts_db_path: str = "~/.engram/fts_index.db"
    fts_enabled: bool = True


class EmbeddingConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-embedding-001"
    # Primary API key for embedding provider (env var ref like ${GEMINI_API_KEY} or literal)
    api_key: str = "${GEMINI_API_KEY}"
    # Key rotation strategy: "failover" (primary first, fallback on error)
    # or "round-robin" (rotate evenly across keys to spread quota)
    key_strategy: str = "failover"


class SemanticConfig(BaseModel):
    provider: str = "sqlite"
    path: str = "~/.engram/semantic.db"
    schema_name: str = Field(default="devops", alias="schema")
    # PostgreSQL settings (used when provider="postgresql")
    dsn: str = "${ENGRAM_SEMANTIC_DSN}"
    pool_min: int = 5
    pool_max: int = 20
    max_nodes: int = 50_000  # Safety cap; log warning if exceeded

    model_config = {"populate_by_name": True}


class OpenClawCaptureConfig(BaseModel):
    """OpenClaw realtime session capture via inotify/watchdog."""
    enabled: bool = True
    sessions_dir: str = "~/.openclaw/agents/main/sessions"


class ClaudeCodeCaptureConfig(BaseModel):
    """Claude Code session capture — watches ~/.claude/projects/ for JSONL sessions."""
    enabled: bool = True
    sessions_dir: str = "~/.claude/projects"


class CaptureConfig(BaseModel):
    enabled: bool = True
    inbox: str = "~/.engram/inbox/"
    poll_interval: int = 5
    openclaw: OpenClawCaptureConfig = Field(default_factory=OpenClawCaptureConfig)
    claude_code: ClaudeCodeCaptureConfig = Field(default_factory=ClaudeCodeCaptureConfig)


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "anthropic/claude-sonnet-4-6"
    api_key: str = "${ANTHROPIC_API_KEY}"
    # Set true for thinking models (e.g. gemini-2.5-flash) to disable thinking tokens
    disable_thinking: bool = False


class ServeConfig(BaseModel):
    port: int = 8765
    host: str = "127.0.0.1"


class HooksConfig(BaseModel):
    """Webhook URLs fired after memory operations (fire-and-forget)."""
    on_remember: str | None = None  # POST {id, content, memory_type} after remember()
    on_think: str | None = None     # POST {question, answer} after think()
    webhook_timeout_seconds: float = 5.0


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
    # fail_open: when Redis is down, allow requests (less secure, more available)
    fail_open: bool = False


class ScoringConfig(BaseModel):
    """Activation-based recall scoring weights. Must sum to ~1.0."""
    similarity_weight: float = 0.5
    retention_weight: float = 0.2
    recency_weight: float = 0.15
    frequency_weight: float = 0.15


class ExtractionConfig(BaseModel):
    """LLM entity/memory extraction settings."""
    llm_model: str = ""  # Empty = use llm.model
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    chunk_size: int = 50
    user_msg_max_len: int = 2000
    assistant_msg_max_len: int = 3000


class RecallConfig(BaseModel):
    """Recall pipeline tuning parameters."""
    search_limit: int = 15
    entity_search_limit: int = 10
    provider_search_limit: int = 5
    entity_graph_depth: int = 2
    entity_boost_score: float = 0.55
    semantic_edge_score: float = 0.5
    entity_co_mention_score: float = 0.4
    keyword_exact_match_score: float = 0.6
    fuzzy_match_score: float = 0.3
    fusion_similarity_weight: float = 0.6
    fusion_retention_weight: float = 0.4
    entity_resolution_context_window: int = 10
    entity_resolution_max_len: int = 3000
    fusion_entry_max_chars: int = 200
    format_for_llm_max_chars: int = 2000
    federated_search_timeout: float = 10.0


class SchedulerConfig(BaseModel):
    """Background scheduler intervals and limits."""
    consolidate_interval_seconds: int = 21600   # 6 hours
    cleanup_interval_seconds: int = 86400       # 24 hours
    decay_report_interval_seconds: int = 86400  # 24 hours
    queue_drain_interval_seconds: int = 60
    tick_interval_seconds: int = 60
    task_timeout_seconds: int = 300
    decay_access_multiplier: float = 0.1


class ConsolidationConfig(BaseModel):
    """Memory consolidation settings."""
    enabled: bool = False
    min_cluster_size: int = 3
    similarity_threshold: float = 0.3
    auto_trigger_threshold: int = 20  # messages before auto-consolidation
    llm_model: str = ""  # Empty = use llm.model


class RetrievalAuditConfig(BaseModel):
    """Retrieval audit logging configuration."""
    enabled: bool = False
    path: str = "~/.engram/retrieval_audit.jsonl"
    max_file_bytes: int = 10_485_760  # 10MB
    max_backups: int = 3


class ResolutionConfig(BaseModel):
    """Entity and temporal resolution settings for recall pipeline."""
    enabled: bool = True
    resolve_pronouns: bool = True
    resolve_temporal: bool = True
    context_window: int = 10  # messages to look back for pronoun resolution
    llm_model: str = ""  # Empty = use llm.model


class RecallPipelineConfig(BaseModel):
    """Recall pipeline orchestration settings."""
    enabled: bool = True
    parallel_search: bool = True
    fusion_top_k: int = 10
    fallback_threshold: float = 0.3  # keyword fallback if best score < this


class FeedbackConfig(BaseModel):
    """User feedback loop settings for memory confidence adjustment."""
    enabled: bool = True
    positive_boost: float = 0.15      # confidence increase on positive feedback
    negative_penalty: float = 0.2     # confidence decrease on negative feedback
    auto_delete_threshold: int = 3    # negative count before auto-delete check
    min_confidence_for_delete: float = 0.15  # confidence floor for auto-delete


class IngestionConfig(BaseModel):
    """Content ingestion settings."""
    poisoning_guard: bool = True
    auto_memory: bool = True


class EventBusConfig(BaseModel):
    """Event bus configuration. Disabled by default (in-process memory bus)."""
    enabled: bool = False
    backend: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379/0"


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


class SessionConfig(BaseModel):
    """Session lifecycle configuration."""
    sessions_dir: str = "~/.engram/sessions"


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
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    recall_pipeline: RecallPipelineConfig = Field(default_factory=RecallPipelineConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval_audit: RetrievalAuditConfig = Field(default_factory=RetrievalAuditConfig)
    providers: list[ProviderEntry] = Field(default_factory=list)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)


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
    "EMBEDDING_KEY_STRATEGY": ("embedding", "key_strategy"),
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
    "CAPTURE_ENABLED": ("capture", "enabled"),
    "CAPTURE_INBOX": ("capture", "inbox"),
    "CAPTURE_POLL_INTERVAL": ("capture", "poll_interval"),
    "CAPTURE_OPENCLAW_ENABLED": ("capture.openclaw", "enabled"),
    "CAPTURE_OPENCLAW_SESSIONS_DIR": ("capture.openclaw", "sessions_dir"),
    "CAPTURE_CLAUDE_CODE_ENABLED": ("capture.claude_code", "enabled"),
    "CAPTURE_CLAUDE_CODE_SESSIONS_DIR": ("capture.claude_code", "sessions_dir"),
    "EPISODIC_DECAY_RATE": ("episodic", "decay_rate"),
    "EPISODIC_DECAY_ENABLED": ("episodic", "decay_enabled"),
    "EPISODIC_DEDUP_ENABLED": ("episodic", "dedup_enabled"),
    "EPISODIC_DEDUP_THRESHOLD": ("episodic", "dedup_threshold"),
    "EPISODIC_MODE": ("episodic", "mode"),
    "EPISODIC_HOST": ("episodic", "host"),
    "EPISODIC_PORT": ("episodic", "port"),
    "EPISODIC_FTS_DB_PATH": ("episodic", "fts_db_path"),
    "EPISODIC_FTS_ENABLED": ("episodic", "fts_enabled"),
    "SCORING_SIMILARITY_WEIGHT": ("scoring", "similarity_weight"),
    "SCORING_RETENTION_WEIGHT": ("scoring", "retention_weight"),
    "SCORING_RECENCY_WEIGHT": ("scoring", "recency_weight"),
    "SCORING_FREQUENCY_WEIGHT": ("scoring", "frequency_weight"),
    "CONSOLIDATION_ENABLED": ("consolidation", "enabled"),
    "CONSOLIDATION_MIN_CLUSTER_SIZE": ("consolidation", "min_cluster_size"),
    "CONSOLIDATION_SIMILARITY_THRESHOLD": ("consolidation", "similarity_threshold"),
    "SESSION_SESSIONS_DIR": ("session", "sessions_dir"),
    "RESOLUTION_ENABLED": ("resolution", "enabled"),
    "RESOLUTION_RESOLVE_PRONOUNS": ("resolution", "resolve_pronouns"),
    "RESOLUTION_RESOLVE_TEMPORAL": ("resolution", "resolve_temporal"),
    "RESOLUTION_CONTEXT_WINDOW": ("resolution", "context_window"),
    "RESOLUTION_LLM_MODEL": ("resolution", "llm_model"),
    "RECALL_PIPELINE_ENABLED": ("recall_pipeline", "enabled"),
    "RECALL_PIPELINE_PARALLEL_SEARCH": ("recall_pipeline", "parallel_search"),
    "RECALL_PIPELINE_FUSION_TOP_K": ("recall_pipeline", "fusion_top_k"),
    "RECALL_PIPELINE_FALLBACK_THRESHOLD": ("recall_pipeline", "fallback_threshold"),
    "FEEDBACK_ENABLED": ("feedback", "enabled"),
    "FEEDBACK_POSITIVE_BOOST": ("feedback", "positive_boost"),
    "FEEDBACK_NEGATIVE_PENALTY": ("feedback", "negative_penalty"),
    "FEEDBACK_AUTO_DELETE_THRESHOLD": ("feedback", "auto_delete_threshold"),
    "FEEDBACK_MIN_CONFIDENCE_FOR_DELETE": ("feedback", "min_confidence_for_delete"),
    "INGESTION_POISONING_GUARD": ("ingestion", "poisoning_guard"),
    "INGESTION_AUTO_MEMORY": ("ingestion", "auto_memory"),
    "CONSOLIDATION_AUTO_TRIGGER_THRESHOLD": ("consolidation", "auto_trigger_threshold"),
    "CONSOLIDATION_LLM_MODEL": ("consolidation", "llm_model"),
    "EXTRACTION_LLM_MODEL": ("extraction", "llm_model"),
    "EXTRACTION_TEMPERATURE": ("extraction", "temperature"),
    "EXTRACTION_MAX_RETRIES": ("extraction", "max_retries"),
    "EXTRACTION_CHUNK_SIZE": ("extraction", "chunk_size"),
    "EXTRACTION_USER_MSG_MAX_LEN": ("extraction", "user_msg_max_len"),
    "EXTRACTION_ASSISTANT_MSG_MAX_LEN": ("extraction", "assistant_msg_max_len"),
    "RECALL_SEARCH_LIMIT": ("recall", "search_limit"),
    "RECALL_ENTITY_SEARCH_LIMIT": ("recall", "entity_search_limit"),
    "RECALL_PROVIDER_SEARCH_LIMIT": ("recall", "provider_search_limit"),
    "RECALL_ENTITY_GRAPH_DEPTH": ("recall", "entity_graph_depth"),
    "RECALL_ENTITY_BOOST_SCORE": ("recall", "entity_boost_score"),
    "RECALL_FUSION_SIMILARITY_WEIGHT": ("recall", "fusion_similarity_weight"),
    "RECALL_FUSION_RETENTION_WEIGHT": ("recall", "fusion_retention_weight"),
    "RECALL_FEDERATED_SEARCH_TIMEOUT": ("recall", "federated_search_timeout"),
    "SCHEDULER_TICK_INTERVAL": ("scheduler", "tick_interval_seconds"),
    "SCHEDULER_TASK_TIMEOUT": ("scheduler", "task_timeout_seconds"),
    "SCHEDULER_CONSOLIDATE_INTERVAL": ("scheduler", "consolidate_interval_seconds"),
    "SCHEDULER_CLEANUP_INTERVAL": ("scheduler", "cleanup_interval_seconds"),
    "HOOKS_WEBHOOK_TIMEOUT": ("hooks", "webhook_timeout_seconds"),
    "RETRIEVAL_AUDIT_ENABLED": ("retrieval_audit", "enabled"),
    "RETRIEVAL_AUDIT_PATH": ("retrieval_audit", "path"),
    "RETRIEVAL_AUDIT_MAX_FILE_BYTES": ("retrieval_audit", "max_file_bytes"),
    "RETRIEVAL_AUDIT_MAX_BACKUPS": ("retrieval_audit", "max_backups"),
    "EVENT_BUS_ENABLED": ("event_bus", "enabled"),
    "EVENT_BUS_BACKEND": ("event_bus", "backend"),
    "EVENT_BUS_REDIS_URL": ("event_bus", "redis_url"),
}

def _get_section_models() -> dict[str, type[BaseModel]]:
    """Lazily build section model map after all classes are defined.

    Keys may be dotted paths (e.g. "capture.openclaw") for nested sections.
    """
    return {
        "serve": ServeConfig,
        "llm": LLMConfig,
        "episodic": EpisodicConfig,
        "embedding": EmbeddingConfig,
        "semantic": SemanticConfig,
        "logging": LoggingConfig,
        "security": SecurityConfig,
        "capture": CaptureConfig,
        "capture.openclaw": OpenClawCaptureConfig,
        "capture.claude_code": ClaudeCodeCaptureConfig,
        "hooks": HooksConfig,
        "auth": AuthConfig,
        "telemetry": TelemetryConfig,
        "audit": AuditConfig,
        "cache": CacheConfig,
        "rate_limit": RateLimitConfig,
        "scoring": ScoringConfig,
        "extraction": ExtractionConfig,
        "recall": RecallConfig,
        "scheduler": SchedulerConfig,
        "consolidation": ConsolidationConfig,
        "resolution": ResolutionConfig,
        "recall_pipeline": RecallPipelineConfig,
        "feedback": FeedbackConfig,
        "ingestion": IngestionConfig,
        "retrieval_audit": RetrievalAuditConfig,
        "session": SessionConfig,
        "event_bus": EventBusConfig,
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

        # Merge into data dict — support dotted section paths (e.g. "capture.openclaw")
        parts = section.split(".")
        node = data
        for part in parts:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[field] = typed_val

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


_ENV_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_anthropic_token(api_key: str) -> str:
    """For Anthropic, auto-read fresh OAuth token from ~/.claude/.credentials.json."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if creds_path.exists():
        try:
            import json
            creds = json.loads(creds_path.read_text())
            fresh_token = creds.get("claudeAiOauth", {}).get("accessToken", "")
            if fresh_token:
                return fresh_token
        except Exception:
            pass
    return api_key


def _apply_provider_key(provider: str, api_key: str) -> None:
    """Set resolved API key in os.environ for litellm to pick up."""
    if not api_key or api_key.startswith("${"):
        return
    if provider == "anthropic":
        api_key = _resolve_anthropic_token(api_key)
    env_key = _ENV_KEY_MAP.get(provider.lower())
    if env_key:
        os.environ[env_key] = api_key


def apply_llm_api_key(config: Config) -> None:
    """Resolve LLM + embedding api_keys and set in os.environ for litellm."""
    # LLM key
    _apply_provider_key(config.llm.provider, config.llm.api_key)
    # Embedding key (may differ from LLM provider)
    _apply_provider_key(config.embedding.provider, config.embedding.api_key)


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
