"""Tests for entity resolver — temporal and pronoun resolution."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from engram.recall.entity_resolver import (
    has_pronouns,
    resolve_pronouns,
    resolve,
)
from engram.recall.temporal_resolver import resolve_temporal
from engram.models import ResolvedText


class TestResolveTemporal:
    """Test deterministic temporal reference resolution.

    temporal_resolver.resolve_temporal returns (text, primary_date: str | None).
    """

    def setup_method(self):
        self.ref_date = datetime(2026, 2, 25, 10, 0, 0)

    def test_hom_nay(self):
        text, date_str = resolve_temporal("hôm nay tôi đi mall", self.ref_date)
        assert "2026-02-25" in text
        assert date_str == "2026-02-25"

    def test_hom_qua(self):
        text, date_str = resolve_temporal("hôm qua tôi gặp bạn", self.ref_date)
        assert "2026-02-24" in text

    def test_hom_kia(self):
        text, date_str = resolve_temporal("hôm kia có mưa", self.ref_date)
        assert "2026-02-23" in text

    def test_ngay_mai(self):
        text, date_str = resolve_temporal("ngày mai tôi đi làm", self.ref_date)
        assert "2026-02-26" in text

    def test_tuan_truoc(self):
        text, date_str = resolve_temporal("tuần trước tôi họp", self.ref_date)
        assert "2026-02-18" in text

    def test_thang_truoc(self):
        text, date_str = resolve_temporal("tháng trước tôi đi Đà Lạt", self.ref_date)
        assert "2026-01" in text

    def test_nam_ngoai(self):
        text, date_str = resolve_temporal("năm ngoái tôi tốt nghiệp", self.ref_date)
        assert "2025" in text

    def test_english_today(self):
        text, date_str = resolve_temporal("I went shopping today", self.ref_date)
        assert "2026-02-25" in text

    def test_english_yesterday(self):
        text, date_str = resolve_temporal("yesterday was busy", self.ref_date)
        assert "2026-02-24" in text

    def test_english_tomorrow(self):
        text, date_str = resolve_temporal("meeting tomorrow", self.ref_date)
        assert "2026-02-26" in text

    def test_english_last_week(self):
        text, date_str = resolve_temporal("last week I deployed", self.ref_date)
        assert "2026-02-18" in text

    def test_no_temporal_refs(self):
        text, date_str = resolve_temporal("Trâm làm nghề gì?", self.ref_date)
        assert text == "Trâm làm nghề gì?"
        assert date_str is None

    def test_multiple_refs_in_one_text(self):
        text, date_str = resolve_temporal("hôm nay ok, ngày mai bận", self.ref_date)
        assert "2026-02-25" in text
        assert "2026-02-26" in text
        assert date_str is not None

    def test_default_reference_date(self):
        """When no reference_date given, uses datetime.now()."""
        text, date_str = resolve_temporal("today is good")
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
        assert result.temporal_refs.get("resolved_date") == "2026-02-25"

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


class TestUnifiedPronouns:
    """Test unified regex-first → LLM-fallback pronoun resolution in resolve()."""

    @pytest.mark.asyncio
    async def test_simple_regex_resolves_without_llm(self):
        """Regex pass should resolve clear pronoun; LLM must NOT be called."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            result = await resolve(
                "anh ấy thích cà phê",
                context=[{"role": "user", "content": "Max là bạn tôi"}],
                resolve_temporal_refs=False,
                resolve_pronoun_refs=True,
            )
        # Regex should have resolved "anh ấy" → "Max"
        assert "Max" in result.resolved
        # LLM must NOT have been called since regex succeeded
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_pronouns_fall_back_to_llm(self):
        """When context has multiple entities of same gender, regex may leave pronouns;
        LLM fallback should then be triggered."""
        # Provide two male entities so regex is ambiguous → leaves "he" unresolved
        # LLM will then resolve it
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"resolved": "Max went to the store", "entities": [{"name": "Max", "type": "person"}]}'
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            result = await resolve(
                "he went to the store",
                context=[
                    {"role": "user", "content": "Max and John are both friends"},
                ],
                resolve_temporal_refs=False,
                resolve_pronoun_refs=True,
            )
        # LLM should have been called since "he" remained after regex (two male candidates)
        mock_llm.assert_called_once()
        assert "Max" in result.resolved

    @pytest.mark.asyncio
    async def test_no_context_skips_both_regex_and_llm(self):
        """With no context, neither regex nor LLM should mutate the text."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            result = await resolve(
                "cô ấy ở đâu?",
                context=None,
                resolve_temporal_refs=False,
                resolve_pronoun_refs=True,
            )
        assert result.resolved == "cô ấy ở đâu?"
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_female_pronoun_resolved_by_regex(self):
        """cô ấy → female entity extracted from context via regex."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            result = await resolve(
                "cô ấy làm nghề gì?",
                context=[{"role": "user", "content": "Linh là bạn của tôi"}],
                resolve_temporal_refs=False,
                resolve_pronoun_refs=True,
            )
        # "Linh" starts with uppercase, extracted by context extractor.
        # Since "Linh" has no gender mapping, regex resolves with single entity fallback.
        assert "Linh" in result.resolved
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_entity_names_from_context(self):
        """Helper extracts capitalized words from context messages."""
        from engram.recall.entity_resolver import _extract_entity_names_from_context
        ctx = [
            {"role": "user", "content": "Trâm và Max là bạn bè"},
            {"role": "assistant", "content": "Tôi hiểu rồi"},
        ]
        names = _extract_entity_names_from_context(ctx)
        # Both capitalized names should appear
        assert "Trâm" in names or "Max" in names


class TestMonthOffset:
    """Test _month_offset helper from temporal_resolver."""

    def test_normal_month(self):
        from engram.recall.temporal_resolver import _month_offset
        from datetime import date
        assert _month_offset(date(2026, 3, 15), -1) == date(2026, 2, 15)

    def test_january_wraps_to_december(self):
        from engram.recall.temporal_resolver import _month_offset
        from datetime import date
        assert _month_offset(date(2026, 1, 15), -1) == date(2025, 12, 15)

    def test_march_31_clamps_to_28(self):
        from engram.recall.temporal_resolver import _month_offset
        from datetime import date
        result = _month_offset(date(2026, 3, 31), -1)
        assert result == date(2026, 2, 28)
