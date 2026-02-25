"""Tests for recall decision layer — trivial message detection."""

import pytest

from engram.recall.decision import should_skip_recall


class TestShouldSkipRecall:
    """Test trivial message detection."""

    # --- Should skip (trivial) ---

    def test_empty_string(self):
        assert should_skip_recall("") is True

    def test_whitespace_only(self):
        assert should_skip_recall("   ") is True

    def test_single_char(self):
        assert should_skip_recall("k") is True

    def test_ok(self):
        assert should_skip_recall("ok") is True

    def test_okay_uppercase(self):
        assert should_skip_recall("OKAY") is True

    def test_yes(self):
        assert should_skip_recall("yes") is True

    def test_no(self):
        assert should_skip_recall("no") is True

    def test_thanks(self):
        assert should_skip_recall("thanks") is True

    def test_thank_you(self):
        assert should_skip_recall("Thank you") is True

    def test_hello(self):
        assert should_skip_recall("hello") is True

    def test_hi(self):
        assert should_skip_recall("hi") is True

    def test_bye(self):
        assert should_skip_recall("bye") is True

    def test_emoji_only(self):
        assert should_skip_recall("👍👌") is True

    # Vietnamese patterns
    def test_vietnamese_cam_on(self):
        assert should_skip_recall("cảm ơn") is True

    def test_vietnamese_chao(self):
        assert should_skip_recall("chào") is True

    def test_vietnamese_duoc(self):
        assert should_skip_recall("được") is True

    def test_vietnamese_tam_biet(self):
        assert should_skip_recall("tạm biệt") is True

    def test_vietnamese_oke(self):
        assert should_skip_recall("oke") is True

    # --- Should NOT skip (meaningful messages) ---

    def test_question(self):
        assert should_skip_recall("Trâm làm nghề gì?") is False

    def test_statement(self):
        assert should_skip_recall("I deployed the API to production") is False

    def test_long_message(self):
        assert should_skip_recall("Can you help me with the database migration?") is False

    def test_ok_in_sentence(self):
        # "ok" as part of longer message is NOT trivial
        assert should_skip_recall("ok let me check the logs") is False

    def test_greeting_with_content(self):
        assert should_skip_recall("hello, can you help me?") is False

    def test_number(self):
        assert should_skip_recall("42") is False

    def test_mixed_emoji_and_text(self):
        assert should_skip_recall("👍 good job on the deployment") is False
