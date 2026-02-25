"""Tests for entity resolver — temporal and pronoun resolution."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from engram.recall.entity_resolver import (
    resolve_temporal,
    has_pronouns,
    resolve_pronouns,
    resolve,
    _month_ago,
)
from engram.models import ResolvedText


class TestResolveTemporal:
    """Test deterministic temporal reference resolution."""

    def setup_method(self):
        self.ref_date = datetime(2026, 2, 25, 10, 0, 0)

    def test_hom_nay(self):
        text, refs = resolve_temporal("hôm nay tôi đi mall", self.ref_date)
        assert "2026-02-25" in text
        assert "hôm nay" in refs

    def test_hom_qua(self):
        text, refs = resolve_temporal("hôm qua tôi gặp bạn", self.ref_date)
        assert "2026-02-24" in text

    def test_hom_kia(self):
        text, refs = resolve_temporal("hôm kia có mưa", self.ref_date)
        assert "2026-02-23" in text

    def test_ngay_mai(self):
        text, refs = resolve_temporal("ngày mai tôi đi làm", self.ref_date)
        assert "2026-02-26" in text

    def test_tuan_truoc(self):
        text, refs = resolve_temporal("tuần trước tôi họp", self.ref_date)
        assert "2026-02-18" in text

    def test_thang_truoc(self):
        text, refs = resolve_temporal("tháng trước tôi đi Đà Lạt", self.ref_date)
        assert "2026-01" in text

    def test_nam_ngoai(self):
        text, refs = resolve_temporal("năm ngoái tôi tốt nghiệp", self.ref_date)
        assert "2025" in text

    def test_english_today(self):
        text, refs = resolve_temporal("I went shopping today", self.ref_date)
        assert "2026-02-25" in text

    def test_english_yesterday(self):
        text, refs = resolve_temporal("yesterday was busy", self.ref_date)
        assert "2026-02-24" in text

    def test_english_tomorrow(self):
        text, refs = resolve_temporal("meeting tomorrow", self.ref_date)
        assert "2026-02-26" in text

    def test_english_last_week(self):
        text, refs = resolve_temporal("last week I deployed", self.ref_date)
        assert "2026-02-18" in text

    def test_no_temporal_refs(self):
        text, refs = resolve_temporal("Trâm làm nghề gì?", self.ref_date)
        assert text == "Trâm làm nghề gì?"
        assert refs == {}

    def test_multiple_refs_in_one_text(self):
        text, refs = resolve_temporal("hôm nay ok, ngày mai bận", self.ref_date)
        assert "2026-02-25" in text
        assert "2026-02-26" in text
        assert len(refs) == 2

    def test_default_reference_date(self):
        """When no reference_date given, uses datetime.now()."""
        text, refs = resolve_temporal("today is good")
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in text


class TestHasPronouns:
    """Test pronoun detection."""

    def test_vietnamese_co_ay(self):
        assert has_pronouns("cô ấy làm nghề gì?") is True

    def test_vietnamese_anh_ay(self):
        assert has_pronouns("anh ấy ở đâu?") is True

    def test_english_she(self):
        assert has_pronouns("What does she do?") is True

    def test_english_he(self):
        assert has_pronouns("He went to the store") is True

    def test_english_they(self):
        assert has_pronouns("they are coming") is True

    def test_no_pronouns(self):
        assert has_pronouns("Trâm làm nghề gì?") is False

    def test_no_pronouns_technical(self):
        assert has_pronouns("Deploy the API to production") is False


class TestResolvePronounsLLM:
    """Test LLM-based pronoun resolution (mocked)."""

    @pytest.mark.asyncio
    async def test_no_pronouns_skips_llm(self):
        result = await resolve_pronouns("Trâm làm nghề gì?", [])
        assert result.resolved == "Trâm làm nghề gì?"
        assert result.entities == []

    @pytest.mark.asyncio
    async def test_no_context_skips_llm(self):
        result = await resolve_pronouns("cô ấy làm nghề gì?", [])
        assert result.resolved == "cô ấy làm nghề gì?"

    @pytest.mark.asyncio
    async def test_successful_resolution(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"resolved": "Trâm làm nghề gì?", "entities": [{"name": "Trâm", "type": "person"}]}'

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await resolve_pronouns(
                "cô ấy làm nghề gì?",
                [{"role": "user", "content": "Trâm là bạn gái cũ của tôi"}],
            )
        assert "Trâm" in result.resolved
        assert len(result.entities) == 1
        assert result.entities[0].name == "Trâm"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_original(self):
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await resolve_pronouns(
                "cô ấy làm nghề gì?",
                [{"role": "user", "content": "Trâm là bạn"}],
            )
        assert result.resolved == "cô ấy làm nghề gì?"
        assert result.entities == []

    @pytest.mark.asyncio
    async def test_llm_returns_markdown_fenced_json(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"resolved": "Trâm ở đâu?", "entities": [{"name": "Trâm", "type": "person"}]}\n```'

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await resolve_pronouns(
                "cô ấy ở đâu?",
                [{"role": "user", "content": "Trâm sống ở Sài Gòn"}],
            )
        assert "Trâm" in result.resolved


class TestResolveFullPipeline:
    """Test the full resolve() pipeline combining temporal + pronoun."""

    @pytest.mark.asyncio
    async def test_temporal_only(self):
        result = await resolve(
            "hôm nay tôi đi mall",
            reference_date=datetime(2026, 2, 25),
            resolve_pronoun_refs=False,
        )
        assert "2026-02-25" in result.resolved
        assert "hôm nay" in result.temporal_refs

    @pytest.mark.asyncio
    async def test_no_resolution_needed(self):
        result = await resolve(
            "Trâm làm nghề VLTL",
            resolve_temporal_refs=False,
            resolve_pronoun_refs=False,
        )
        assert result.resolved == "Trâm làm nghề VLTL"
        assert result.original == "Trâm làm nghề VLTL"

    @pytest.mark.asyncio
    async def test_temporal_and_pronoun_combined(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"resolved": "2026-02-25 Trâm đi mall", "entities": [{"name": "Trâm", "type": "person"}]}'

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await resolve(
                "hôm nay cô ấy đi mall",
                context=[{"role": "user", "content": "Trâm thích đi mall"}],
                reference_date=datetime(2026, 2, 25),
            )
        assert "2026-02-25" in result.resolved
        assert result.original == "hôm nay cô ấy đi mall"


class TestMonthAgo:
    """Test _month_ago helper."""

    def test_normal_month(self):
        assert _month_ago(datetime(2026, 3, 15)) == "2026-02-15"

    def test_january_wraps_to_december(self):
        assert _month_ago(datetime(2026, 1, 15)) == "2025-12-15"

    def test_march_31_clamps_to_28(self):
        # March 31 - 1 month → Feb 28 (safe clamping)
        assert _month_ago(datetime(2026, 3, 31)) == "2026-02-28"
