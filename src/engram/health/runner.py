"""Health check orchestration: deep_check, full_health_check, _add_component."""

from __future__ import annotations

import asyncio
import logging

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

logger = logging.getLogger("engram")


def _add_component(components: dict, comp: ComponentHealth) -> None:
    """Add a ComponentHealth result to the components dict."""
    entry: dict = {"status": comp.status}
    if comp.latency_ms:
        entry["latency_ms"] = round(comp.latency_ms, 1)
    if comp.details:
        entry.update(comp.details)
    if comp.error:
        entry["error"] = comp.error
    components[comp.name] = entry


async def deep_check(episodic, graph) -> dict:
    """Run core component checks and return aggregated status."""
    results = await asyncio.gather(
        check_chromadb(episodic),
        check_semantic(graph),
        check_disk(),
        return_exceptions=True,
    )

    components: dict = {}
    overall = "healthy"

    for r in results:
        if isinstance(r, Exception):
            continue
        _add_component(components, r)
        if r.status == "unhealthy":
            overall = "unhealthy"

    return {"status": overall, "components": components}


async def full_health_check(episodic=None, graph=None, llm_model: str = "", config=None) -> dict:
    """Run comprehensive health check across all subsystems.

    Checks: chromadb, semantic, disk, api_keys, fts5, llm, embedding,
            watcher, redis (if cache/rate_limit enabled), constitution,
            resource_tier.
    """
    tasks = [check_disk(), check_fts5(), check_watcher()]

    # Sync checks
    api_keys_result = check_api_keys()
    constitution_result = check_constitution()
    resource_tier_result = check_resource_tier()

    if episodic is not None:
        tasks.append(check_chromadb(episodic))
    if graph is not None:
        tasks.append(check_semantic(graph))

    # LLM + embedding only if API keys present (avoids wasted time)
    if api_keys_result.status == "healthy":
        tasks.append(check_llm(llm_model))
        tasks.append(check_embedding())

    # Redis check if cache or rate_limit is enabled
    if config is not None:
        redis_url = None
        if getattr(config, "cache", None) and config.cache.enabled:
            redis_url = config.cache.redis_url
        elif getattr(config, "rate_limit", None) and config.rate_limit.enabled:
            redis_url = config.rate_limit.redis_url
        if redis_url:
            tasks.append(check_redis(redis_url))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    components: dict = {}
    overall = "healthy"

    # Add sync results — api_keys unhealthy → degraded only (service still partly functional)
    _add_component(components, api_keys_result)
    if api_keys_result.status == "unhealthy":
        overall = "degraded"
    for sync_result in (constitution_result, resource_tier_result):
        _add_component(components, sync_result)
        if sync_result.status == "unhealthy" and overall == "healthy":
            overall = "degraded"
        elif sync_result.status == "degraded" and overall == "healthy":
            overall = "degraded"

    for r in results:
        if isinstance(r, Exception):
            logger.warning("Health check exception: %s", r)
            continue
        _add_component(components, r)
        if r.status == "unhealthy":
            overall = "unhealthy"
        elif r.status == "degraded" and overall == "healthy":
            overall = "degraded"

    return {"status": overall, "components": components}
