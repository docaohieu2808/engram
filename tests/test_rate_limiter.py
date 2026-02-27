"""Tests for sliding-window rate limiter (rate_limiter.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.rate_limiter import RateLimiter


# --- Helpers ---

def _mock_redis(pipeline_results: list | None = None, ping_ok: bool = True):
    """Create a mock Redis with pipeline support."""
    redis = AsyncMock()
    redis.ping = AsyncMock() if ping_ok else AsyncMock(side_effect=ConnectionError("no redis"))
    redis.zrem = AsyncMock()

    pipe = AsyncMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=pipeline_results or [0, 0, 1, True])
    redis.pipeline = MagicMock(return_value=pipe)
    return redis, pipe


# --- Tests: initialization ---

def test_init_defaults():
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=100, burst=20)
    assert rl.requests_per_minute == 100
    assert rl.burst == 20
    assert rl._redis is None


# --- Tests: connect ---

@pytest.mark.asyncio
async def test_connect_success():
    rl = RateLimiter("redis://localhost:6379")
    redis_mock, _ = _mock_redis()
    with patch("engram.rate_limiter.Redis") as MockRedis:
        MockRedis.from_url.return_value = redis_mock
        with patch("engram.rate_limiter._REDIS_AVAILABLE", True):
            await rl.connect()
    assert rl._redis is not None


@pytest.mark.asyncio
async def test_connect_failure_graceful():
    rl = RateLimiter("redis://bad-host:6379")
    redis_mock, _ = _mock_redis(ping_ok=False)
    redis_mock.ping.side_effect = ConnectionError("refused")
    with patch("engram.rate_limiter.Redis") as MockRedis:
        MockRedis.from_url.return_value = redis_mock
        with patch("engram.rate_limiter._REDIS_AVAILABLE", True):
            await rl.connect()
    assert rl._redis is None  # graceful degradation


@pytest.mark.asyncio
async def test_connect_no_redis_module():
    """When redis package not installed, connect is a no-op."""
    rl = RateLimiter("redis://localhost:6379")
    with patch("engram.rate_limiter._REDIS_AVAILABLE", False):
        await rl.connect()
    assert rl._redis is None


# --- Tests: check — fail closed when redis unavailable (S-H4) ---

@pytest.mark.asyncio
async def test_check_no_redis_denies():
    """S-H4: when Redis unavailable, deny (fail closed)."""
    rl = RateLimiter("redis://localhost:6379")
    rl._redis = None
    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is False
    assert remaining == 0
    assert reset_at == 0


@pytest.mark.asyncio
async def test_fail_open_allows_when_redis_unavailable():
    """fail_open=True: when Redis unavailable, allow the request."""
    rl = RateLimiter("redis://localhost:6379", fail_open=True)
    rl._redis = None
    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is True
    assert remaining == 0
    assert reset_at == 0


@pytest.mark.asyncio
async def test_fail_closed_denies_when_redis_unavailable():
    """fail_open=False (default): when Redis unavailable, deny the request."""
    rl = RateLimiter("redis://localhost:6379", fail_open=False)
    rl._redis = None
    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is False
    assert remaining == 0
    assert reset_at == 0


# --- Tests: check — allowed ---

@pytest.mark.asyncio
async def test_check_allowed_under_limit():
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=60, burst=10)
    redis_mock, pipe = _mock_redis(pipeline_results=[0, 5, 1, True])  # count=5
    rl._redis = redis_mock

    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is True
    assert remaining == 64  # (60+10) - 5 - 1
    assert reset_at > 0


@pytest.mark.asyncio
async def test_check_allowed_first_request():
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=60, burst=10)
    redis_mock, _ = _mock_redis(pipeline_results=[0, 0, 1, True])  # count=0
    rl._redis = redis_mock

    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is True
    assert remaining == 69  # (60+10) - 0 - 1


# --- Tests: check — denied (over limit) ---

@pytest.mark.asyncio
async def test_check_denied_over_limit():
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=60, burst=10)
    redis_mock, _ = _mock_redis(pipeline_results=[0, 70, 1, True])  # count=70 >= limit(70)
    rl._redis = redis_mock

    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is False
    assert remaining == 0
    # Should have called zrem to remove the entry we just added
    redis_mock.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_check_denied_at_exact_limit():
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=5, burst=0)
    redis_mock, _ = _mock_redis(pipeline_results=[0, 5, 1, True])  # count=5 >= limit(5)
    rl._redis = redis_mock

    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is False


# --- Tests: check — pipeline error (S-H4 fail closed) ---

@pytest.mark.asyncio
async def test_check_pipeline_error_denies():
    """S-H4: Redis pipeline error → fail closed."""
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=60, burst=10)
    redis_mock, pipe = _mock_redis()
    pipe.execute.side_effect = RuntimeError("pipeline broken")
    rl._redis = redis_mock

    allowed, remaining, reset_at = await rl.check("tenant-1")
    assert allowed is False
    assert remaining == 0
    assert reset_at == 0


# --- Tests: tenant isolation ---

@pytest.mark.asyncio
async def test_check_uses_tenant_scoped_key():
    """Verify rate limit key includes tenant_id."""
    rl = RateLimiter("redis://localhost:6379")
    redis_mock, pipe = _mock_redis(pipeline_results=[0, 0, 1, True])
    rl._redis = redis_mock

    await rl.check("my-tenant")
    # Check that pipeline was called (key validation via behavior)
    pipe.zremrangebyscore.assert_called_once()
    call_args = pipe.zremrangebyscore.call_args
    assert "engram:ratelimit:my-tenant" == call_args[0][0]


# --- Tests: burst allowance ---

@pytest.mark.asyncio
async def test_burst_allows_over_base_rpm():
    """Burst allows requests beyond requests_per_minute up to rpm+burst."""
    rl = RateLimiter("redis://localhost:6379", requests_per_minute=10, burst=5)
    # count=12 which is > rpm(10) but < rpm+burst(15)
    redis_mock, _ = _mock_redis(pipeline_results=[0, 12, 1, True])
    rl._redis = redis_mock

    allowed, remaining, _ = await rl.check("t1")
    assert allowed is True
    assert remaining == 2  # (10+5) - 12 - 1
