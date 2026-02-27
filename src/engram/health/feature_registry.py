"""Feature registry builder — aggregates ALL ~350 features from config + hardcoded lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.health.feature_registry_static import (
    ALGORITHMS,
    CLI_COMMANDS,
    DATA_MODEL_VALUES,
    HTTP_ENDPOINTS,
    INTEGRATIONS,
    MIDDLEWARE,
    MCP_TOOLS,
    PIPELINE_STAGES,
    RUNTIME_FEATURES,
    WS_COMMANDS,
    WS_EVENTS,
)

if TYPE_CHECKING:
    from engram.config import Config


@dataclass
class FeatureEntry:
    name: str           # "POST /api/v1/remember"
    category: str       # "HTTP Endpoint", "CLI Command", etc.
    subcategory: str    # "Memory", "Admin", "Graph", etc.
    status: str         # "enabled", "disabled", "always-on", or current value
    config_path: str    # "cache.enabled" or "" for always-on
    env_var: str        # "ENGRAM_CACHE_ENABLED" or ""


# (name, config_path, category, env_var) — boolean flags only
_BOOL_FLAGS: list[tuple[str, str, str, str]] = [
    # Storage
    ("Memory Decay",          "episodic.decay_enabled",          "Storage",    "ENGRAM_EPISODIC_DECAY_ENABLED"),
    ("Semantic Dedup",        "episodic.dedup_enabled",          "Storage",    "ENGRAM_EPISODIC_DEDUP_ENABLED"),
    ("FTS5 Full-Text Search", "episodic.fts_enabled",            "Storage",    "ENGRAM_EPISODIC_FTS_ENABLED"),
    # Pipeline
    ("Recall Pipeline",       "recall_pipeline.enabled",         "Pipeline",   "ENGRAM_RECALL_PIPELINE_ENABLED"),
    ("Parallel Search",       "recall_pipeline.parallel_search", "Pipeline",   "ENGRAM_RECALL_PIPELINE_PARALLEL_SEARCH"),
    ("Entity Resolution",     "resolution.enabled",              "Pipeline",   "ENGRAM_RESOLUTION_ENABLED"),
    ("Pronoun Resolution",    "resolution.resolve_pronouns",     "Pipeline",   "ENGRAM_RESOLUTION_RESOLVE_PRONOUNS"),
    ("Temporal Resolution",   "resolution.resolve_temporal",     "Pipeline",   "ENGRAM_RESOLUTION_RESOLVE_TEMPORAL"),
    ("Feedback Loop",         "feedback.enabled",                "Pipeline",   "ENGRAM_FEEDBACK_ENABLED"),
    ("Poisoning Guard",       "ingestion.poisoning_guard",       "Pipeline",   "ENGRAM_INGESTION_POISONING_GUARD"),
    ("Auto Memory Detection", "ingestion.auto_memory",           "Pipeline",   "ENGRAM_INGESTION_AUTO_MEMORY"),
    ("Memory Consolidation",  "consolidation.enabled",           "Pipeline",   "ENGRAM_CONSOLIDATION_ENABLED"),
    # Enterprise
    ("Authentication/JWT",    "auth.enabled",                    "Enterprise", "ENGRAM_AUTH_ENABLED"),
    ("Redis Cache",           "cache.enabled",                   "Enterprise", "ENGRAM_CACHE_ENABLED"),
    ("Rate Limiting",         "rate_limit.enabled",              "Enterprise", "ENGRAM_RATE_LIMIT_ENABLED"),
    ("Rate Limit Fail-Open",  "rate_limit.fail_open",            "Enterprise", "ENGRAM_RATE_LIMIT_FAIL_OPEN"),
    ("Audit Trail",           "audit.enabled",                   "Enterprise", "ENGRAM_AUDIT_ENABLED"),
    ("OpenTelemetry",         "telemetry.enabled",               "Enterprise", "ENGRAM_TELEMETRY_ENABLED"),
    ("Retrieval Audit",       "retrieval_audit.enabled",         "Enterprise", "ENGRAM_RETRIEVAL_AUDIT_ENABLED"),
    ("Event Bus",             "event_bus.enabled",               "Enterprise", "ENGRAM_EVENT_BUS_ENABLED"),
    # Capture
    ("Capture (global)",      "capture.enabled",                 "Capture",    "ENGRAM_CAPTURE_ENABLED"),
    ("OpenClaw Watcher",      "capture.openclaw.enabled",        "Capture",    "ENGRAM_CAPTURE_OPENCLAW_ENABLED"),
    ("Claude Code Watcher",   "capture.claude_code.enabled",     "Capture",    "ENGRAM_CAPTURE_CLAUDE_CODE_ENABLED"),
    # Discovery
    ("Local Discovery",       "discovery.local",                 "Discovery",  "ENGRAM_DISCOVERY_LOCAL"),
]

# (name, config_path, subcategory, env_var_suffix) — non-boolean params
# env_var is derived automatically from config_path when not specified
_PARAM_ENTRIES: list[tuple[str, str, str]] = [
    # EpisodicConfig
    ("episodic.path",              "path",            "Episodic"),
    ("episodic.mode",              "mode",            "Episodic"),
    ("episodic.host",              "host",            "Episodic"),
    ("episodic.port",              "port",            "Episodic"),
    ("episodic.namespace",         "namespace",       "Episodic"),
    ("episodic.decay_rate",        "decay_rate",      "Episodic"),
    ("episodic.dedup_threshold",   "dedup_threshold", "Episodic"),
    ("episodic.fts_db_path",       "fts_db_path",     "Episodic"),
    # EmbeddingConfig
    ("embedding.provider",         "provider",        "Embedding"),
    ("embedding.model",            "model",           "Embedding"),
    ("embedding.key_strategy",     "key_strategy",    "Embedding"),
    # SemanticConfig
    ("semantic.provider",          "provider",        "Semantic"),
    ("semantic.path",              "path",            "Semantic"),
    ("semantic.dsn",               "dsn",             "Semantic"),
    ("semantic.pool_min",          "pool_min",        "Semantic"),
    ("semantic.pool_max",          "pool_max",        "Semantic"),
    ("semantic.max_nodes",         "max_nodes",       "Semantic"),
    ("semantic.schema_name",       "schema_name",     "Semantic"),
    # LLMConfig
    ("llm.provider",               "provider",        "LLM"),
    ("llm.model",                  "model",           "LLM"),
    # ServeConfig
    ("serve.port",                 "port",            "Serve"),
    ("serve.host",                 "host",            "Serve"),
    # AuthConfig
    ("auth.jwt_expiry_hours",      "jwt_expiry_hours","Auth"),
    # CacheConfig
    ("cache.redis_url",            "redis_url",       "Cache"),
    ("cache.recall_ttl",           "recall_ttl",      "Cache"),
    ("cache.think_ttl",            "think_ttl",       "Cache"),
    ("cache.query_ttl",            "query_ttl",       "Cache"),
    # RateLimitConfig
    ("rate_limit.redis_url",       "redis_url",       "RateLimit"),
    ("rate_limit.requests_per_minute", "requests_per_minute", "RateLimit"),
    ("rate_limit.burst",           "burst",           "RateLimit"),
    # CaptureConfig
    ("capture.inbox",              "inbox",           "Capture"),
    ("capture.poll_interval",      "poll_interval",   "Capture"),
    ("capture.openclaw.sessions_dir",   "sessions_dir","Capture"),
    ("capture.claude_code.sessions_dir","sessions_dir","Capture"),
    # ScoringConfig
    ("scoring.similarity_weight",  "similarity_weight","Scoring"),
    ("scoring.retention_weight",   "retention_weight", "Scoring"),
    ("scoring.recency_weight",     "recency_weight",   "Scoring"),
    ("scoring.frequency_weight",   "frequency_weight", "Scoring"),
    # ConsolidationConfig
    ("consolidation.min_cluster_size",    "min_cluster_size",    "Consolidation"),
    ("consolidation.similarity_threshold","similarity_threshold","Consolidation"),
    ("consolidation.auto_trigger_threshold","auto_trigger_threshold","Consolidation"),
    ("consolidation.llm_model",    "llm_model",       "Consolidation"),
    # ResolutionConfig
    ("resolution.context_window",  "context_window",  "Resolution"),
    ("resolution.llm_model",       "llm_model",       "Resolution"),
    # RecallPipelineConfig
    ("recall_pipeline.fusion_top_k","fusion_top_k",   "RecallPipeline"),
    ("recall_pipeline.fallback_threshold","fallback_threshold","RecallPipeline"),
    # FeedbackConfig
    ("feedback.positive_boost",    "positive_boost",  "Feedback"),
    ("feedback.negative_penalty",  "negative_penalty","Feedback"),
    ("feedback.auto_delete_threshold","auto_delete_threshold","Feedback"),
    ("feedback.min_confidence_for_delete","min_confidence_for_delete","Feedback"),
    # RetrievalAuditConfig
    ("retrieval_audit.path",       "path",            "RetrievalAudit"),
    # SessionConfig
    ("session.sessions_dir",       "sessions_dir",    "Session"),
    # EventBusConfig
    ("event_bus.backend",          "backend",         "EventBus"),
    ("event_bus.redis_url",        "redis_url",       "EventBus"),
    # TelemetryConfig
    ("telemetry.otlp_endpoint",    "otlp_endpoint",   "Telemetry"),
    ("telemetry.sample_rate",      "sample_rate",     "Telemetry"),
    ("telemetry.service_name",     "service_name",    "Telemetry"),
    # AuditConfig
    ("audit.backend",              "backend",         "Audit"),
    ("audit.path",                 "path",            "Audit"),
    # SecurityConfig
    ("security.max_content_length","max_content_length","Security"),
    # LoggingConfig
    ("logging.format",             "format",          "Logging"),
    ("logging.level",              "level",           "Logging"),
    # HooksConfig
    ("hooks.on_remember",          "on_remember",     "Hooks"),
    ("hooks.on_think",             "on_think",        "Hooks"),
    # DiscoveryConfig
    ("discovery.hosts",            "hosts",           "Discovery"),
    ("discovery.endpoints",        "endpoints",       "Discovery"),
]


def _get_nested(obj, path: str):
    """Traverse dot-path on pydantic models / dicts."""
    for part in path.split("."):
        if obj is None:
            return None
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def _path_to_env_var(path: str) -> str:
    """Convert 'cache.redis_url' → 'ENGRAM_CACHE_REDIS_URL'."""
    return "ENGRAM_" + path.replace(".", "_").upper()


def _build_bool_flags(config: "Config") -> list[FeatureEntry]:
    entries = []
    for name, path, subcat, env_var in _BOOL_FLAGS:
        val = _get_nested(config, path)
        enabled = bool(val) if val is not None else False
        entries.append(FeatureEntry(
            name=name,
            category="Config Boolean Flags",
            subcategory=subcat,
            status="enabled" if enabled else "disabled",
            config_path=path,
            env_var=env_var,
        ))
    return entries


def _build_params(config: "Config") -> list[FeatureEntry]:
    entries = []
    for path, _field, subcat in _PARAM_ENTRIES:
        val = _get_nested(config, path)
        if val is None:
            str_val = "(not set)"
        elif isinstance(val, list):
            str_val = f"[{len(val)} items]"
        elif val == "":
            str_val = "(empty)"
        else:
            str_val = str(val)
        entries.append(FeatureEntry(
            name=path,
            category="Config Parameters",
            subcategory=subcat,
            status=str_val,
            config_path=path,
            env_var=_path_to_env_var(path),
        ))
    return entries


def build_full_registry(config: "Config") -> list[FeatureEntry]:
    """Build the complete feature registry (~350 entries) from config + static lists."""
    entries: list[FeatureEntry] = []

    # 1. Config Boolean Flags (26)
    entries.extend(_build_bool_flags(config))

    # 2. Config Parameters (65+)
    entries.extend(_build_params(config))

    # 3. CLI Commands
    for cmd, subcat in CLI_COMMANDS:
        entries.append(FeatureEntry(cmd, "CLI Commands", subcat, "always-on", "", ""))

    # 4. HTTP API Endpoints
    for method, path, subcat in HTTP_ENDPOINTS:
        entries.append(FeatureEntry(f"{method} {path}", "HTTP Endpoints", subcat, "always-on", "", ""))

    # 5. MCP Tools
    for tool, subcat in MCP_TOOLS:
        entries.append(FeatureEntry(tool, "MCP Tools", subcat, "always-on", "", ""))

    # 6. WebSocket Commands + Events
    for cmd, subcat in WS_COMMANDS:
        entries.append(FeatureEntry(cmd, "WebSocket", subcat, "always-on", "", ""))
    for evt, subcat in WS_EVENTS:
        entries.append(FeatureEntry(evt, "WebSocket", subcat, "always-on", "", ""))

    # 7. Pipeline Stages
    for stage, subcat in PIPELINE_STAGES:
        entries.append(FeatureEntry(stage, "Pipeline Stages", subcat, "always-on", "", ""))

    # 8. Algorithms
    for algo, subcat in ALGORITHMS:
        entries.append(FeatureEntry(algo, "Algorithms", subcat, "always-on", "", ""))

    # 9. Middleware
    for mw, subcat in MIDDLEWARE:
        entries.append(FeatureEntry(mw, "Middleware", subcat, "always-on", "", ""))

    # 10. Integrations
    for integ, subcat in INTEGRATIONS:
        entries.append(FeatureEntry(integ, "Integrations", subcat, "always-on", "", ""))

    # 11. Data Model Values
    for val_name, subcat in DATA_MODEL_VALUES:
        entries.append(FeatureEntry(val_name, "Data Model Values", subcat, "always-on", "", ""))

    # 12. Runtime Features
    for feat, subcat in RUNTIME_FEATURES:
        entries.append(FeatureEntry(feat, "Runtime Features", subcat, "always-on", "", ""))

    return entries
