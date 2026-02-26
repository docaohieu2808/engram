"""Tests for pronoun_resolver — regex-based pronoun-to-entity resolution."""

import pytest

from engram.recall.pronoun_resolver import resolve_pronouns, has_resolvable_pronouns


class TestVietnamesePronouns:
    """Vietnamese pronoun resolution."""

    def test_anh_ay(self):
        result = resolve_pronouns("anh ấy thích cà phê", ["Max"])
        assert result == "Max thích cà phê"

    def test_chi_ay(self):
        result = resolve_pronouns("chị ấy làm ở đâu?", ["Trâm"])
        assert result == "Trâm làm ở đâu?"

    def test_co_ay(self):
        result = resolve_pronouns("cô ấy rất giỏi", ["Linh"])
        assert result == "Linh rất giỏi"

    def test_ong_ay(self):
        result = resolve_pronouns("ông ấy đã về hưu", ["Ông Nam"])
        assert result == "Ông Nam đã về hưu"

    def test_ba_ay(self):
        result = resolve_pronouns("bà ấy dạy toán", ["Bà Hoa"])
        assert result == "Bà Hoa dạy toán"

    def test_no_short_form(self):
        # "nó" → resolved to first entity
        result = resolve_pronouns("nó hay quậy lắm", ["Mèo"])
        assert result == "Mèo hay quậy lắm"

    def test_han_short_form(self):
        result = resolve_pronouns("hắn chạy đi rồi", ["Kẻ trộm"])
        assert result == "Kẻ trộm chạy đi rồi"

    def test_anh_short_form(self):
        result = resolve_pronouns("ảnh đang ngủ", ["Ba"])
        assert result == "Ba đang ngủ"

    def test_co_short_form(self):
        result = resolve_pronouns("cổ hay cười", ["Hà"])
        assert result == "Hà hay cười"


class TestEnglishPronouns:
    """English pronoun resolution."""

    def test_he(self):
        result = resolve_pronouns("he went to the store", ["John"])
        assert result == "John went to the store"

    def test_him(self):
        result = resolve_pronouns("I called him yesterday", ["David"])
        assert result == "I called David yesterday"

    def test_his(self):
        result = resolve_pronouns("his car is red", ["Tom"])
        assert result == "Tom car is red"

    def test_she(self):
        result = resolve_pronouns("she is a doctor", ["Sarah"])
        assert result == "Sarah is a doctor"

    def test_her(self):
        result = resolve_pronouns("I met her at the café", ["Anna"])
        assert result == "I met Anna at the café"

    def test_hers(self):
        result = resolve_pronouns("the bag is hers", ["Maria"])
        assert result == "the bag is Maria"


class TestNoContext:
    """Empty or missing context returns unchanged content."""

    def test_empty_context_unchanged(self):
        result = resolve_pronouns("anh ấy thích cà phê", [])
        assert result == "anh ấy thích cà phê"

    def test_empty_content_unchanged(self):
        result = resolve_pronouns("", ["Max"])
        assert result == ""

    def test_none_like_empty_content(self):
        # Empty string edge case
        result = resolve_pronouns("", [])
        assert result == ""


class TestNoPronouns:
    """Content without pronouns returns unchanged."""

    def test_no_pronouns_unchanged(self):
        result = resolve_pronouns("Trâm làm nghề gì?", ["Max"])
        assert result == "Trâm làm nghề gì?"

    def test_technical_content_unchanged(self):
        result = resolve_pronouns("Deploy the API to production", ["Max"])
        assert result == "Deploy the API to production"


class TestWordBoundary:
    """Pronouns must match word boundaries — don't resolve substrings."""

    def test_she_not_in_shear(self):
        # "shear" must not be mangled — regex uses \bshe\b
        result = resolve_pronouns("shear the wool", ["Anna"])
        # "shear" does NOT contain standalone "she" at word boundary
        assert "shear" in result
        assert "Anna" not in result

    def test_he_not_in_the(self):
        result = resolve_pronouns("the quick brown fox", ["Bob"])
        assert "the" in result
        assert "Bob" not in result

    def test_her_not_in_there(self):
        result = resolve_pronouns("there is no issue", ["Alice"])
        assert "there" in result

    def test_he_not_in_here(self):
        result = resolve_pronouns("here we go", ["Bob"])
        assert "here" in result
        assert "Bob" not in result


class TestMultiplePronounsWithSingleEntity:
    """Multiple pronouns in one sentence all resolve to same entity."""

    def test_multiple_he_resolved(self):
        result = resolve_pronouns("he said he will call him back", ["Peter"])
        assert result == "Peter said Peter will call Peter back"

    def test_vietnamese_multiple(self):
        result = resolve_pronouns("anh ấy nói anh ấy bận", ["Minh"])
        assert result == "Minh nói Minh bận"


class TestMostRecentEntityFirst:
    """First entity in context_entities list is used as resolution target."""

    def test_uses_first_entity(self):
        # With multiple entities and no gender map, male pronouns need a gender hint.
        # Pass entity_genders so "he" resolves to the male entity (Alice listed as male here).
        result = resolve_pronouns(
            "he is tall",
            ["Alice", "Bob", "Charlie"],
            entity_genders={"Alice": "male"},
        )
        assert result == "Alice is tall"

    def test_single_entity_list(self):
        result = resolve_pronouns("she laughed", ["Diana"])
        assert result == "Diana laughed"


class TestHasResolvablePronouns:
    """Test pronoun detection helper."""

    def test_detects_anh_ay(self):
        assert has_resolvable_pronouns("anh ấy đâu rồi?") is True

    def test_detects_she(self):
        assert has_resolvable_pronouns("she is here") is True

    def test_detects_he(self):
        assert has_resolvable_pronouns("he left early") is True

    def test_no_pronouns(self):
        assert has_resolvable_pronouns("Trâm làm nghề gì?") is False

    def test_empty_string(self):
        assert has_resolvable_pronouns("") is False

    def test_technical_no_pronouns(self):
        assert has_resolvable_pronouns("Deploy API to production") is False
