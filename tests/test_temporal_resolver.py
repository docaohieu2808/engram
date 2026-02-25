"""Tests for temporal_resolver — relative date resolution to ISO format."""

import pytest
from datetime import date

from engram.recall.temporal_resolver import resolve_temporal


REF = date(2026, 2, 25)  # Wednesday


class TestVietnamesePatterns:
    """Vietnamese temporal phrases."""

    def test_hom_nay(self):
        content, iso = resolve_temporal("Đi mall hôm nay", REF)
        assert "(ngày 2026-02-25)" in content
        assert iso == "2026-02-25"

    def test_hom_qua(self):
        content, iso = resolve_temporal("hôm qua tôi gặp bạn", REF)
        assert "(ngày 2026-02-24)" in content
        assert iso == "2026-02-24"

    def test_hom_kia(self):
        content, iso = resolve_temporal("hôm kia có mưa", REF)
        assert "(ngày 2026-02-23)" in content
        assert iso == "2026-02-23"

    def test_ngay_mai(self):
        content, iso = resolve_temporal("ngày mai tôi đi làm", REF)
        assert "(ngày 2026-02-26)" in content
        assert iso == "2026-02-26"

    def test_ngay_mot(self):
        content, iso = resolve_temporal("ngày mốt anh về", REF)
        assert "(ngày 2026-02-27)" in content
        assert iso == "2026-02-27"

    def test_ngay_kia(self):
        content, iso = resolve_temporal("ngày kia họp team", REF)
        assert "(ngày 2026-02-27)" in content
        assert iso == "2026-02-27"

    def test_tuan_truoc(self):
        content, iso = resolve_temporal("tuần trước tôi họp", REF)
        assert "(ngày 2026-02-18)" in content
        assert iso == "2026-02-18"

    def test_tuan_roi(self):
        content, iso = resolve_temporal("tuần rồi deploy production", REF)
        assert "(ngày 2026-02-18)" in content
        assert iso == "2026-02-18"

    def test_tuan_sau(self):
        content, iso = resolve_temporal("tuần sau đi Đà Lạt", REF)
        assert "(ngày 2026-03-04)" in content
        assert iso == "2026-03-04"

    def test_tuan_toi(self):
        content, iso = resolve_temporal("tuần tới họp board", REF)
        assert "(ngày 2026-03-04)" in content
        assert iso == "2026-03-04"

    def test_thang_truoc(self):
        content, iso = resolve_temporal("tháng trước tôi đi Hà Nội", REF)
        assert "2026-01" in content
        assert iso.startswith("2026-01")

    def test_thang_roi(self):
        content, iso = resolve_temporal("tháng rồi lương tăng", REF)
        assert "2026-01" in content

    def test_thang_sau(self):
        content, iso = resolve_temporal("tháng sau khai trương", REF)
        assert "2026-03" in content

    def test_thang_toi(self):
        content, iso = resolve_temporal("tháng tới nghỉ phép", REF)
        assert "2026-03" in content

    def test_nam_ngoai(self):
        content, iso = resolve_temporal("năm ngoái tôi tốt nghiệp", REF)
        assert "2025" in content
        assert iso == "2025"

    def test_nam_truoc(self):
        content, iso = resolve_temporal("năm trước lạm phát cao", REF)
        assert "2025" in content

    def test_nam_sau(self):
        content, iso = resolve_temporal("năm sau mua nhà", REF)
        assert "2027" in content

    def test_nam_toi(self):
        content, iso = resolve_temporal("năm tới lên kế hoạch", REF)
        assert "2027" in content

    def test_sang_nay(self):
        content, iso = resolve_temporal("sáng nay uống cà phê", REF)
        assert "(ngày 2026-02-25)" in content
        assert iso == "2026-02-25"

    def test_chieu_nay(self):
        content, iso = resolve_temporal("chiều nay họp team", REF)
        assert "(ngày 2026-02-25)" in content
        assert iso == "2026-02-25"

    def test_toi_qua(self):
        content, iso = resolve_temporal("tối qua xem phim", REF)
        assert "(ngày 2026-02-24)" in content
        assert iso == "2026-02-24"


class TestEnglishPatterns:
    """English temporal phrases."""

    def test_today(self):
        content, iso = resolve_temporal("I went shopping today", REF)
        assert "(ngày 2026-02-25)" in content
        assert iso == "2026-02-25"

    def test_yesterday(self):
        content, iso = resolve_temporal("yesterday was very busy", REF)
        assert "(ngày 2026-02-24)" in content
        assert iso == "2026-02-24"

    def test_tomorrow(self):
        content, iso = resolve_temporal("meeting tomorrow at 10am", REF)
        assert "(ngày 2026-02-26)" in content
        assert iso == "2026-02-26"

    def test_last_week(self):
        content, iso = resolve_temporal("I deployed last week", REF)
        assert "(ngày 2026-02-18)" in content
        assert iso == "2026-02-18"

    def test_last_month(self):
        content, iso = resolve_temporal("last month was hectic", REF)
        assert "2026-01" in content

    def test_last_year(self):
        content, iso = resolve_temporal("last year we shipped v2", REF)
        assert "2025" in content

    def test_last_night(self):
        content, iso = resolve_temporal("last night I couldn't sleep", REF)
        assert "(ngày 2026-02-24)" in content
        assert iso == "2026-02-24"

    def test_this_morning(self):
        content, iso = resolve_temporal("this morning had coffee", REF)
        assert "(ngày 2026-02-25)" in content
        assert iso == "2026-02-25"


class TestNoTemporalRef:
    """Cases with no temporal references."""

    def test_no_ref_returns_original(self):
        content, iso = resolve_temporal("Trâm làm nghề gì?", REF)
        assert content == "Trâm làm nghề gì?"
        assert iso is None

    def test_empty_string(self):
        content, iso = resolve_temporal("", REF)
        assert content == ""
        assert iso is None

    def test_technical_content(self):
        content, iso = resolve_temporal("Deploy API to production", REF)
        assert iso is None

    def test_name_only(self):
        content, iso = resolve_temporal("Max", REF)
        assert iso is None


class TestMultipleRefs:
    """Multiple temporal references in one text."""

    def test_two_different_refs(self):
        content, iso = resolve_temporal("hôm nay ok, ngày mai bận", REF)
        assert "(ngày 2026-02-25)" in content
        assert "(ngày 2026-02-26)" in content
        # primary_date is the first pattern that matches in the registry order;
        # ngày mai appears before hôm nay in the pattern list → 2026-02-26 is primary
        assert iso in ("2026-02-25", "2026-02-26")

    def test_three_refs(self):
        content, iso = resolve_temporal("hôm qua họp, hôm nay báo cáo, ngày mai nghỉ", REF)
        assert "(ngày 2026-02-24)" in content
        assert "(ngày 2026-02-25)" in content
        assert "(ngày 2026-02-26)" in content

    def test_mixed_languages(self):
        content, iso = resolve_temporal("hôm nay and yesterday were good", REF)
        assert "(ngày 2026-02-25)" in content
        assert "(ngày 2026-02-24)" in content


class TestDefaultReferenceDate:
    """When no reference_date is given, defaults to today."""

    def test_defaults_to_today(self):
        import datetime
        content, iso = resolve_temporal("today is nice")
        today = datetime.date.today().isoformat()
        assert iso == today

    def test_accepts_datetime_object(self):
        from datetime import datetime
        dt = datetime(2026, 2, 25, 14, 30, 0)
        content, iso = resolve_temporal("hôm nay", dt)
        assert iso == "2026-02-25"
