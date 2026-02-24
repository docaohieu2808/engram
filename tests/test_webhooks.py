"""Tests for webhook/hook firing behavior in engram/hooks.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from engram.hooks import fire_hook


# --- no-op tests ---

def test_fire_hook_noop_when_url_none():
    """Hook must not attempt any network call when URL is None."""
    with patch("urllib.request.urlopen") as mock_open:
        fire_hook(None, {"event": "remember", "id": "123"})
    mock_open.assert_not_called()


def test_fire_hook_noop_when_url_empty():
    """Hook must not attempt any network call when URL is empty string."""
    with patch("urllib.request.urlopen") as mock_open:
        fire_hook("", {"event": "remember", "id": "123"})
    mock_open.assert_not_called()


# --- payload tests ---

def test_fire_hook_sends_correct_json_payload():
    """Hook must POST the exact data dict as JSON."""
    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        return resp

    data = {"event": "remember", "id": "abc", "content": "hello"}
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fire_hook("http://example.com:9999/hook", data)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.get_header("Content-type") == "application/json"
    body = json.loads(req.data)
    assert body["event"] == "remember"
    assert body["id"] == "abc"
    assert body["content"] == "hello"


def test_fire_hook_uses_post_method():
    """Hook must use POST HTTP method."""
    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fire_hook("http://example.com:9999/hook", {"x": 1})

    assert captured_requests[0].get_method() == "POST"


# --- error resilience tests ---

def test_fire_hook_does_not_crash_on_connection_error():
    """Hook must swallow connection errors silently."""
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        # Must not raise
        fire_hook("http://example.com:9999/hook", {"event": "remember"})


def test_fire_hook_does_not_crash_on_timeout():
    """Hook must swallow timeout errors silently."""
    import socket
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        fire_hook("http://localhost:9999/hook", {"event": "think"})


def test_fire_hook_does_not_crash_on_generic_exception():
    """Hook must swallow any unexpected exception."""
    with patch("urllib.request.urlopen", side_effect=RuntimeError("unexpected")):
        fire_hook("http://example.com:9999/hook", {"event": "remember"})


# --- timeout behavior ---

def test_fire_hook_passes_timeout_to_urlopen():
    """Hook must pass a timeout so it never blocks indefinitely."""
    timeout_used = []

    def fake_urlopen(req, timeout=None):
        timeout_used.append(timeout)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fire_hook("http://example.com:9999/hook", {"event": "remember"})

    assert timeout_used and timeout_used[0] is not None
    assert timeout_used[0] <= 10  # must be bounded (hooks.py uses 5s)
