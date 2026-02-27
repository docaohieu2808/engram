"""Feature flag status checker — reports on/off state of all 24 boolean config flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.config import Config


@dataclass
class FeatureStatus:
    name: str           # Human-readable feature name
    config_path: str    # Dot-path to config field (e.g. "cache.enabled")
    enabled: bool       # Current on/off state
    category: str       # Group: Storage / Pipeline / Enterprise / Capture
    env_var: str        # ENGRAM_* env var that controls it


# Registry of all boolean feature flags: (name, config_path, category, env_var)
_FEATURE_REGISTRY: list[tuple[str, str, str, str]] = [
    # Storage
    ("Memory Decay",          "episodic.decay_enabled",        "Storage",    "ENGRAM_EPISODIC_DECAY_ENABLED"),
    ("Semantic Dedup",        "episodic.dedup_enabled",        "Storage",    "ENGRAM_EPISODIC_DEDUP_ENABLED"),
    ("FTS5 Full-Text Search", "episodic.fts_enabled",          "Storage",    "ENGRAM_EPISODIC_FTS_ENABLED"),
    # Pipeline
    ("Recall Pipeline",       "recall_pipeline.enabled",       "Pipeline",   "ENGRAM_RECALL_PIPELINE_ENABLED"),
    ("Parallel Search",       "recall_pipeline.parallel_search", "Pipeline", "ENGRAM_RECALL_PIPELINE_PARALLEL_SEARCH"),
    ("Entity Resolution",     "resolution.enabled",            "Pipeline",   "ENGRAM_RESOLUTION_ENABLED"),
    ("Pronoun Resolution",    "resolution.resolve_pronouns",   "Pipeline",   "ENGRAM_RESOLUTION_RESOLVE_PRONOUNS"),
    ("Temporal Resolution",   "resolution.resolve_temporal",   "Pipeline",   "ENGRAM_RESOLUTION_RESOLVE_TEMPORAL"),
    ("Feedback Loop",         "feedback.enabled",              "Pipeline",   "ENGRAM_FEEDBACK_ENABLED"),
    ("Poisoning Guard",       "ingestion.poisoning_guard",     "Pipeline",   "ENGRAM_INGESTION_POISONING_GUARD"),
    ("Auto Memory Detection", "ingestion.auto_memory",         "Pipeline",   "ENGRAM_INGESTION_AUTO_MEMORY"),
    ("Memory Consolidation",  "consolidation.enabled",         "Pipeline",   "ENGRAM_CONSOLIDATION_ENABLED"),
    # Enterprise
    ("Authentication/JWT",    "auth.enabled",                  "Enterprise", "ENGRAM_AUTH_ENABLED"),
    ("Redis Cache",           "cache.enabled",                 "Enterprise", "ENGRAM_CACHE_ENABLED"),
    ("Rate Limiting",         "rate_limit.enabled",            "Enterprise", "ENGRAM_RATE_LIMIT_ENABLED"),
    ("Audit Trail",           "audit.enabled",                 "Enterprise", "ENGRAM_AUDIT_ENABLED"),
    ("OpenTelemetry",         "telemetry.enabled",             "Enterprise", "ENGRAM_TELEMETRY_ENABLED"),
    ("Retrieval Audit",       "retrieval_audit.enabled",       "Enterprise", "ENGRAM_RETRIEVAL_AUDIT_ENABLED"),
    ("Event Bus",             "event_bus.enabled",             "Enterprise", "ENGRAM_EVENT_BUS_ENABLED"),
    # Capture
    ("Capture (global)",      "capture.enabled",               "Capture",    "ENGRAM_CAPTURE_ENABLED"),
    ("OpenClaw Watcher",      "capture.openclaw.enabled",      "Capture",    "ENGRAM_CAPTURE_OPENCLAW_ENABLED"),
    ("Claude Code Watcher",   "capture.claude_code.enabled",   "Capture",    "ENGRAM_CAPTURE_CLAUDE_CODE_ENABLED"),
    # Enterprise (additional)
    ("Rate Limit Fail-Open",  "rate_limit.fail_open",          "Enterprise", "ENGRAM_RATE_LIMIT_FAIL_OPEN"),
    # Discovery
    ("Local Discovery",       "discovery.local",               "Discovery",  "ENGRAM_DISCOVERY_LOCAL"),
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


def check_feature_flags(config: "Config") -> list[FeatureStatus]:
    """Return FeatureStatus for every registered boolean feature flag."""
    results: list[FeatureStatus] = []
    for name, path, category, env_var in _FEATURE_REGISTRY:
        val = _get_nested(config, path)
        enabled = bool(val) if val is not None else False
        results.append(FeatureStatus(
            name=name,
            config_path=path,
            enabled=enabled,
            category=category,
            env_var=env_var,
        ))
    return results
