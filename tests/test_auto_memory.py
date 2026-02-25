"""Tests for auto-memory candidate detection."""

import pytest

from engram.capture.auto_memory import detect_candidates
from engram.models import MemoryType


class TestDetectCandidates:
    """Test auto-save candidate detection."""

    # Manual save
    def test_save_prefix(self):
        candidates = detect_candidates("Save: Trâm's birthday is March 15")
        assert len(candidates) == 1
        assert candidates[0].category == "manual"
        assert candidates[0].importance == 4
        assert candidates[0].content == "Trâm's birthday is March 15"

    def test_save_prefix_case_insensitive(self):
        candidates = detect_candidates("SAVE: important fact")
        assert len(candidates) == 1
        assert candidates[0].category == "manual"

    # Identity patterns
    def test_my_name_is(self):
        candidates = detect_candidates("my name is Hiếu")
        assert any(c.category == "identity" for c in candidates)

    def test_ten_toi_la(self):
        candidates = detect_candidates("tên tôi là Hiếu")
        assert any(c.category == "identity" for c in candidates)

    # Preference patterns
    def test_i_prefer(self):
        candidates = detect_candidates("I prefer using PostgreSQL over MySQL")
        assert any(c.category == "preference" for c in candidates)
        assert any(c.memory_type == MemoryType.PREFERENCE for c in candidates)

    def test_toi_thich(self):
        candidates = detect_candidates("tôi thích ăn sushi")
        assert any(c.category == "preference" for c in candidates)

    def test_i_hate(self):
        candidates = detect_candidates("I hate writing tests without mocks")
        assert any(c.category == "preference" for c in candidates)

    # Explicit memory requests
    def test_remember_that(self):
        candidates = detect_candidates("remember that the API uses port 8765")
        assert any(c.category == "explicit" for c in candidates)

    def test_nho_la(self):
        candidates = detect_candidates("nhớ là meeting lúc 3pm")
        assert any(c.category == "explicit" for c in candidates)

    # Behavioral patterns
    def test_i_always(self):
        candidates = detect_candidates("I always use dark mode")
        assert any(c.category == "pattern" for c in candidates)

    def test_i_never(self):
        candidates = detect_candidates("I never deploy on Fridays")
        assert any(c.category == "pattern" for c in candidates)

    # Decision patterns
    def test_i_decided(self):
        candidates = detect_candidates("I decided to use Redis for caching")
        assert any(c.category == "decision" for c in candidates)
        assert any(c.memory_type == MemoryType.DECISION for c in candidates)

    # Sensitive data — should return empty
    def test_password_blocked(self):
        candidates = detect_candidates("my password is hunter2")
        assert candidates == []

    def test_api_key_blocked(self):
        candidates = detect_candidates("Save: api_key = sk-12345")
        assert candidates == []

    def test_credit_card_blocked(self):
        candidates = detect_candidates("my credit card number is 4111-1111")
        assert candidates == []

    def test_ssh_key_blocked(self):
        candidates = detect_candidates("here's my ssh key: ssh-rsa AAAA")
        assert candidates == []

    # No candidates
    def test_normal_question(self):
        candidates = detect_candidates("What's the weather like?")
        assert candidates == []

    def test_empty_string(self):
        candidates = detect_candidates("")
        assert candidates == []

    def test_short_string(self):
        candidates = detect_candidates("hi")
        assert candidates == []

    def test_technical_statement(self):
        candidates = detect_candidates("The API returns 200 on success")
        assert candidates == []
