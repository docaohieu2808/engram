"""Health check package for engram — backward-compatible re-exports."""

from engram.health.components import (
    ComponentHealth,
    check_api_keys,
    check_chromadb,
    check_constitution,
    check_disk,
    check_embedding,
    check_fts5,
    check_llm,
    check_redis,
    check_resource_tier,
    check_semantic,
    check_watcher,
)
from engram.health.feature_flags import FeatureStatus, check_feature_flags
from engram.health.feature_registry import FeatureEntry, build_full_registry
from engram.health.runner import _add_component, deep_check, full_health_check

__all__ = [
    # dataclasses
    "ComponentHealth",
    "FeatureStatus",
    "FeatureEntry",
    # component checks
    "check_api_keys",
    "check_chromadb",
    "check_constitution",
    "check_disk",
    "check_embedding",
    "check_fts5",
    "check_llm",
    "check_redis",
    "check_resource_tier",
    "check_semantic",
    "check_watcher",
    # feature flags
    "check_feature_flags",
    # feature registry
    "build_full_registry",
    # orchestration
    "_add_component",
    "deep_check",
    "full_health_check",
]
