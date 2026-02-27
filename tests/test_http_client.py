"""Tests for EngramHttpClient — mocked httpx, no live server required."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from engram.http_client import EngramHttpClient


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


async def _make_client(base_url: str = "http://localhost:8000", **kwargs) -> EngramHttpClient:
    """Create client with mocked httpx.AsyncClient internals."""
    return EngramHttpClient(base_url, **kwargs)


# ------------------------------------------------------------------ #
#  Initialization tests                                               #
# ------------------------------------------------------------------ #

def test_init_rejects_missing_scheme():
    with pytest.raises(ValueError, match="http://"):
        EngramHttpClient("localhost:8000")


def test_init_rejects_wrong_scheme():
    with pytest.raises(ValueError, match="http://"):
        EngramHttpClient("ftp://localhost:8000")


def test_init_accepts_http():
    c = EngramHttpClient("http://localhost:8000")
    assert c._base_url == "http://localhost:8000"


def test_init_accepts_https():
    c = EngramHttpClient("https://api.example.com/")
    assert c._base_url == "https://api.example.com"  # trailing slash stripped


def test_init_api_key_not_logged(caplog):
    """api_key must not appear in log output."""
    import logging
    with caplog.at_level(logging.DEBUG, logger="engram.http_client"):
        EngramHttpClient("http://localhost:8000", api_key="super-secret-key")
    assert "super-secret-key" not in caplog.text


# ------------------------------------------------------------------ #
#  Context manager tests                                              #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_async_context_manager_creates_and_closes_client():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance

        async with EngramHttpClient("http://localhost:8000") as client:
            assert client._client is mock_instance

        mock_instance.aclose.assert_called_once()
        assert client._client is None


@pytest.mark.asyncio
async def test_require_client_raises_without_context_manager():
    client = EngramHttpClient("http://localhost:8000")
    with pytest.raises(RuntimeError, match="context manager"):
        client._require_client()


# ------------------------------------------------------------------ #
#  remember() tests                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_remember_returns_id_on_success():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.return_value = _mock_response(200, {"status": "ok", "id": "mem-123"})

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.remember("Visited Paris today")

    assert result == "mem-123"


@pytest.mark.asyncio
async def test_remember_returns_none_on_http_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.side_effect = httpx.ConnectError("connection refused")

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.remember("Test content")

    assert result is None


@pytest.mark.asyncio
async def test_remember_returns_none_on_status_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.return_value = _mock_response(500, {"detail": "server error"})

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.remember("Test content")

    assert result is None


# ------------------------------------------------------------------ #
#  recall() tests                                                     #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_recall_returns_results_on_success():
    results_data = [{"content": "Visited Paris", "score": 0.9}]
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.get.return_value = _mock_response(200, {"status": "ok", "results": results_data})

        async with EngramHttpClient("http://localhost:8000") as client:
            results = await client.recall("Paris trip")

    assert results == results_data


@pytest.mark.asyncio
async def test_recall_returns_empty_list_on_http_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.get.side_effect = httpx.TimeoutException("timeout")

        async with EngramHttpClient("http://localhost:8000") as client:
            results = await client.recall("Paris trip")

    assert results == []


# ------------------------------------------------------------------ #
#  health() tests                                                     #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_health_returns_true_on_200():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.get.return_value = mock_response

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.health()

    assert result is True


@pytest.mark.asyncio
async def test_health_returns_false_on_non_200():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_instance.get.return_value = mock_response

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.health()

    assert result is False


@pytest.mark.asyncio
async def test_health_returns_false_on_connection_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.get.side_effect = httpx.ConnectError("refused")

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.health()

    assert result is False


# ------------------------------------------------------------------ #
#  think() tests                                                      #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_think_returns_answer_string():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.return_value = _mock_response(200, {"status": "ok", "answer": "You were in Paris."})

        async with EngramHttpClient("http://localhost:8000") as client:
            answer = await client.think("Where have I been?")

    assert answer == "You were in Paris."


@pytest.mark.asyncio
async def test_think_returns_empty_string_on_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.side_effect = httpx.ConnectError("refused")

        async with EngramHttpClient("http://localhost:8000") as client:
            answer = await client.think("Where have I been?")

    assert answer == ""


# ------------------------------------------------------------------ #
#  feedback() tests                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_feedback_returns_dict_on_success():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.return_value = _mock_response(200, {"status": "ok", "confidence_delta": 0.15})

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.feedback("mem-123", "positive")

    assert result.get("status") == "ok"


@pytest.mark.asyncio
async def test_feedback_returns_empty_dict_on_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.post.side_effect = httpx.ConnectError("refused")

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.feedback("mem-123", "negative")

    assert result == {}


# ------------------------------------------------------------------ #
#  graph_query() tests                                                #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_graph_query_returns_results_on_success():
    nodes = [{"type": "Person", "name": "Alice"}]
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.get.return_value = _mock_response(200, {"status": "ok", "results": nodes})

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.graph_query(keyword="Alice")

    assert result == nodes


@pytest.mark.asyncio
async def test_graph_query_returns_empty_list_on_error():
    with patch("engram.http_client.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        mock_instance.get.side_effect = httpx.ConnectError("refused")

        async with EngramHttpClient("http://localhost:8000") as client:
            result = await client.graph_query(keyword="Alice")

    assert result == []
