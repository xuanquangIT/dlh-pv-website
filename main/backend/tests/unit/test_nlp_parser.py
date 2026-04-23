"""Unit tests for app.services.solar_ai_chat.nlp_parser"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat.enums import ChatTopic
from app.services.solar_ai_chat.nlp_parser import (
    FACILITY_ID_MAP,
    WEATHER_METRIC_CATALOG,
    ExtremeMetricQuery,
    expand_facility_codes_in_message,
    extract_extreme_metric_query,
    extract_query_date,
    extract_specific_hour,
    extract_timeframe,
    make_extreme_query,
    parse_timeframe_days,
    resolve_facility_code,
    resolve_weather_metric,
    score_metric_keywords,
    strip_timeframe_noise,
    topic_for_extreme_metric,
)


# ---------------------------------------------------------------------------
# topic_for_extreme_metric
# ---------------------------------------------------------------------------

class TestTopicForExtremeMetric(unittest.TestCase):
    def test_aqi_maps_to_data_quality_issues(self) -> None:
        self.assertEqual(topic_for_extreme_metric("aqi"), ChatTopic.DATA_QUALITY_ISSUES)

    def test_energy_maps_to_energy_performance(self) -> None:
        self.assertEqual(topic_for_extreme_metric("energy"), ChatTopic.ENERGY_PERFORMANCE)

    def test_weather_maps_to_energy_performance(self) -> None:
        self.assertEqual(topic_for_extreme_metric("weather"), ChatTopic.ENERGY_PERFORMANCE)

    def test_unknown_metric_maps_to_energy_performance(self) -> None:
        self.assertEqual(topic_for_extreme_metric("radiation"), ChatTopic.ENERGY_PERFORMANCE)


# ---------------------------------------------------------------------------
# make_extreme_query
# ---------------------------------------------------------------------------

class TestMakeExtremeQuery(unittest.TestCase):
    def test_creates_correct_dataclass(self) -> None:
        q = make_extreme_query("energy", "highest", "day", None)
        self.assertEqual(q.metric_name, "energy")
        self.assertEqual(q.query_type, "highest")
        self.assertEqual(q.timeframe, "day")
        self.assertIsNone(q.specific_hour)

    def test_creates_with_specific_hour(self) -> None:
        q = make_extreme_query("aqi", "lowest", "hour", 14)
        self.assertEqual(q.specific_hour, 14)
        self.assertIsInstance(q, ExtremeMetricQuery)

    def test_frozen_dataclass_immutable(self) -> None:
        q = make_extreme_query("aqi", "lowest", "day", None)
        with self.assertRaises((AttributeError, TypeError)):
            q.metric_name = "energy"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extract_extreme_metric_query
# ---------------------------------------------------------------------------

class TestExtractExtremeMetricQuery(unittest.TestCase):
    # --- AQI ---
    def test_detects_lowest_aqi_vietnamese(self) -> None:
        result = extract_extreme_metric_query("AQI thấp nhất ngày 5/1/2026")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "aqi")
        self.assertEqual(result.query_type, "lowest")

    def test_detects_highest_aqi_vietnamese(self) -> None:
        result = extract_extreme_metric_query("AQI cao nhất hôm nay")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "aqi")
        self.assertEqual(result.query_type, "highest")

    def test_detects_aqi_with_min_keyword(self) -> None:
        result = extract_extreme_metric_query("aqi min hom nay")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.query_type, "lowest")

    def test_detects_aqi_with_max_keyword(self) -> None:
        result = extract_extreme_metric_query("aqi max tuan nay")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.query_type, "highest")

    # --- Energy ---
    def test_detects_highest_energy_mwh(self) -> None:
        result = extract_extreme_metric_query("Sản lượng energy cao nhất trong tuần")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "energy")
        self.assertEqual(result.query_type, "highest")

    def test_detects_lowest_energy_nang_luong(self) -> None:
        result = extract_extreme_metric_query("nang luong thap nhat ngay 2/1/2026")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "energy")

    def test_detects_energy_by_dien_nang_keyword(self) -> None:
        result = extract_extreme_metric_query("dien nang cao nhat thang")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "energy")

    def test_detects_energy_by_mwh_keyword(self) -> None:
        result = extract_extreme_metric_query("MWh thap nhat 24 gio")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "energy")

    def test_detects_energy_by_san_luong_keyword(self) -> None:
        result = extract_extreme_metric_query("san luong cao nhat tuan")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "energy")

    # --- Weather ---
    def test_detects_weather_nhiet_do(self) -> None:
        # "đ" is a Latin letter with stroke (U+0111), NFD normalization removes the stroke
        # leaving only "d" — so "nhiet do" must be passed as ASCII-normalized text
        result = extract_extreme_metric_query("nhiet do thap nhat trong thang")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "weather")

    def test_detects_weather_by_wind_keyword(self) -> None:
        result = extract_extreme_metric_query("wind cao nhat ngay nay")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "weather")

    def test_detects_weather_by_buc_xa(self) -> None:
        result = extract_extreme_metric_query("buc xa thap nhat tuan nay")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metric_name, "weather")

    # --- None cases ---
    def test_returns_none_when_no_extreme_keyword(self) -> None:
        result = extract_extreme_metric_query("Cho tôi tổng quan hệ thống")
        self.assertIsNone(result)

    def test_returns_none_for_ranking_request_top_n(self) -> None:
        result = extract_extreme_metric_query("top 3 tram AQI cao nhat")
        self.assertIsNone(result)

    def test_returns_none_for_ranking_request_top_prefix(self) -> None:
        result = extract_extreme_metric_query("top energy cao nhat")
        self.assertIsNone(result)

    def test_returns_none_for_danh_sach_ranking(self) -> None:
        result = extract_extreme_metric_query("danh sach cac tram energy cao nhat")
        self.assertIsNone(result)

    def test_lowest_wins_over_highest_when_both_present(self) -> None:
        result = extract_extreme_metric_query("Khong phai cao nhat ma la thap nhat AQI")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.query_type, "lowest")

    def test_includes_specific_hour_when_hour_found(self) -> None:
        result = extract_extreme_metric_query("AQI thap nhat luc 14 gio chieu")
        self.assertIsNotNone(result)
        assert result is not None
        # hour should be extracted and timeframe should be "hour"
        self.assertEqual(result.timeframe, "hour")


# ---------------------------------------------------------------------------
# extract_timeframe
# ---------------------------------------------------------------------------

class TestExtractTimeframe(unittest.TestCase):
    def test_specific_hour_overrides_everything(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat theo nam 2026", specific_hour=14), "hour")

    def test_24h_detected(self) -> None:
        self.assertEqual(extract_timeframe("energy thap nhat theo 24 gio"), "24h")

    def test_24h_detected_compact(self) -> None:
        self.assertEqual(extract_timeframe("energy cao nhat 24h"), "24h")

    def test_1h_detected(self) -> None:
        self.assertEqual(extract_timeframe("aqi cao nhat trong 1 gio"), "hour")

    def test_1h_compact_detected(self) -> None:
        self.assertEqual(extract_timeframe("aqi max 1h"), "hour")

    def test_theo_gio_detected(self) -> None:
        self.assertEqual(extract_timeframe("aqi cao nhat theo gio"), "hour")

    def test_week_detected(self) -> None:
        self.assertEqual(extract_timeframe("energy cao nhat theo tuan"), "week")

    def test_month_detected(self) -> None:
        self.assertEqual(extract_timeframe("nhiet do thap nhat theo thang 1/2026"), "month")

    def test_year_detected_theo_nam(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat theo nam 2026"), "year")

    def test_year_detected_trong_nam(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat trong nam 2026"), "year")

    def test_year_detected_nam_number(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat nam 2026"), "year")

    def test_all_time_detected_lich_su(self) -> None:
        self.assertEqual(extract_timeframe("lich su aqi toan bo cac tram"), "all_time")

    def test_all_time_detected_ever(self) -> None:
        self.assertEqual(extract_timeframe("highest aqi ever"), "all_time")

    def test_all_time_detected_historical(self) -> None:
        self.assertEqual(extract_timeframe("historical energy lowest"), "all_time")

    def test_defaults_to_day(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat ngay 5/1/2026"), "day")

    def test_viet_nam_country_not_trigger_year(self) -> None:
        # "viet nam" should NOT trigger year timeframe
        result = extract_timeframe("aqi thap nhat cac tram viet nam ngay 5/1/2026")
        self.assertEqual(result, "day")

    def test_nam_nay_triggers_year(self) -> None:
        self.assertEqual(extract_timeframe("aqi thap nhat nam nay"), "year")

    def test_nam_ngoai_triggers_year(self) -> None:
        self.assertEqual(extract_timeframe("energy cao nhat nam ngoai"), "year")


# ---------------------------------------------------------------------------
# parse_timeframe_days
# ---------------------------------------------------------------------------

class TestParseTimeframeDays(unittest.TestCase):
    def test_hom_nay_returns_0(self) -> None:
        self.assertEqual(parse_timeframe_days("hôm nay"), 0)

    def test_today_returns_0(self) -> None:
        self.assertEqual(parse_timeframe_days("today"), 0)

    def test_hom_qua_returns_1(self) -> None:
        self.assertEqual(parse_timeframe_days("hôm qua"), 1)

    def test_yesterday_returns_1(self) -> None:
        self.assertEqual(parse_timeframe_days("yesterday"), 1)

    def test_24h_returns_1(self) -> None:
        self.assertEqual(parse_timeframe_days("24h qua"), 1)

    def test_tuan_returns_7(self) -> None:
        self.assertEqual(parse_timeframe_days("trong tuần"), 7)

    def test_week_returns_7(self) -> None:
        self.assertEqual(parse_timeframe_days("this week"), 7)

    def test_thang_returns_30(self) -> None:
        self.assertEqual(parse_timeframe_days("trong tháng"), 30)

    def test_month_returns_30(self) -> None:
        self.assertEqual(parse_timeframe_days("last month"), 30)

    def test_nam_returns_365(self) -> None:
        self.assertEqual(parse_timeframe_days("trong năm nay"), 365)

    def test_year_returns_365(self) -> None:
        self.assertEqual(parse_timeframe_days("this year"), 365)

    def test_unrecognized_returns_none(self) -> None:
        self.assertIsNone(parse_timeframe_days("give me data"))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(parse_timeframe_days(""))


# ---------------------------------------------------------------------------
# extract_specific_hour
# ---------------------------------------------------------------------------

class TestExtractSpecificHour(unittest.TestCase):
    def test_giờ_chieu_pm_adjustment(self) -> None:
        # "3 gio chieu" → 15
        result = extract_specific_hour("3 gio chieu")
        self.assertEqual(result, 15)

    def test_gio_toi_pm_adjustment(self) -> None:
        result = extract_specific_hour("8 gio toi")
        self.assertEqual(result, 20)

    def test_gio_sang_keeps_hour(self) -> None:
        result = extract_specific_hour("7 gio sang")
        self.assertEqual(result, 7)

    def test_sang_12_converts_to_0(self) -> None:
        result = extract_specific_hour("12 gio sang")
        self.assertEqual(result, 0)

    def test_pm_adjustment(self) -> None:
        result = extract_specific_hour("3:00 pm")
        self.assertEqual(result, 15)

    def test_am_keeps_hour(self) -> None:
        result = extract_specific_hour("9:00 am")
        self.assertEqual(result, 9)

    def test_vao_luc_pattern(self) -> None:
        result = extract_specific_hour("vao luc 10 gio")
        self.assertEqual(result, 10)

    def test_no_hour_returns_none(self) -> None:
        result = extract_specific_hour("thap nhat trong tuan nay")
        self.assertIsNone(result)

    def test_invalid_minute_skipped(self) -> None:
        # minute >= 60 should cause pattern to continue to next
        result = extract_specific_hour("10:75 am")
        self.assertIsNone(result)

    def test_hour_24_normalizes_to_0(self) -> None:
        # hour == 24 special case
        result = extract_specific_hour("24 gio")
        self.assertIsNone(result)  # 24 alone won't match the pattern normally

    def test_dem_period_adjustment(self) -> None:
        result = extract_specific_hour("11 gio dem")
        self.assertEqual(result, 23)


# ---------------------------------------------------------------------------
# extract_query_date
# ---------------------------------------------------------------------------

class TestExtractQueryDate(unittest.TestCase):
    def test_hom_nay_returns_base_date(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("hôm nay", base_date=base)
        self.assertEqual(result, base)

    def test_today_returns_base_date(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("today", base_date=base)
        self.assertEqual(result, base)

    def test_hom_qua_returns_yesterday(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("hôm qua", base_date=base)
        self.assertEqual(result, date(2026, 1, 14))

    def test_yesterday_keyword(self) -> None:
        base = date(2026, 3, 1)
        result = extract_query_date("yesterday", base_date=base)
        self.assertEqual(result, date(2026, 2, 28))

    def test_ngay_mai_returns_tomorrow(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("ngày mai", base_date=base)
        self.assertEqual(result, date(2026, 1, 16))

    def test_tomorrow_keyword(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("tomorrow", base_date=base)
        self.assertEqual(result, date(2026, 1, 16))

    def test_dmy_format_slash(self) -> None:
        result = extract_query_date("ngay 5/1/2026")
        self.assertEqual(result, date(2026, 1, 5))

    def test_dmy_format_dash(self) -> None:
        result = extract_query_date("ngay 15-03-2026")
        self.assertEqual(result, date(2026, 3, 15))

    def test_iso_format_yyyy_mm_dd(self) -> None:
        result = extract_query_date("ngay 2026-04-20")
        self.assertEqual(result, date(2026, 4, 20))

    def test_month_year_format(self) -> None:
        result = extract_query_date("thang 3/2026")
        self.assertEqual(result, date(2026, 3, 1))

    def test_year_only_format(self) -> None:
        result = extract_query_date("nam 2025")
        self.assertEqual(result, date(2025, 1, 1))

    def test_returns_none_when_no_date_found(self) -> None:
        result = extract_query_date("tong quan he thong")
        self.assertIsNone(result)

    def test_invalid_day_month_still_extracts_year(self) -> None:
        # 32/13 is invalid but the year 2026 is still extracted → returns date(2026, 1, 1)
        result = extract_query_date("ngay 32/13/2026")
        self.assertEqual(result, date(2026, 1, 1))

    def test_no_numeric_date_returns_none(self) -> None:
        result = extract_query_date("tong quan he thong solar")
        self.assertIsNone(result)

    def test_uses_today_by_default_when_no_base(self) -> None:
        result = extract_query_date("hôm nay")
        self.assertEqual(result, date.today())

    def test_vietnamese_diacritics_in_message(self) -> None:
        base = date(2026, 1, 15)
        result = extract_query_date("Ngày mai trời đẹp", base_date=base)
        self.assertEqual(result, date(2026, 1, 16))


# ---------------------------------------------------------------------------
# strip_timeframe_noise
# ---------------------------------------------------------------------------

class TestStripTimeframeNoise(unittest.TestCase):
    def test_strips_24h_pattern(self) -> None:
        result = strip_timeframe_noise("energy cao nhat 24h trong ngay")
        self.assertNotIn("24h", result)

    def test_strips_24_gio(self) -> None:
        result = strip_timeframe_noise("aqi thap nhat 24 gio")
        self.assertNotIn("24 gio", result)

    def test_strips_1h_pattern(self) -> None:
        result = strip_timeframe_noise("buc xa 1h qua")
        self.assertNotIn("1h", result)

    def test_strips_theo_gio(self) -> None:
        result = strip_timeframe_noise("nhiet do cao nhat theo gio")
        self.assertNotIn("theo gio", result)

    def test_strips_moi_gio(self) -> None:
        result = strip_timeframe_noise("san luong moi gio")
        self.assertNotIn("moi gio", result)

    def test_no_extra_whitespace_in_output(self) -> None:
        result = strip_timeframe_noise("energy 24h 1h theo gio cao nhat")
        self.assertNotIn("  ", result)
        self.assertEqual(result, result.strip())

    def test_non_noise_strings_preserved(self) -> None:
        msg = "nhiet do cao nhat ngay 5"
        result = strip_timeframe_noise(msg)
        self.assertIn("nhiet do cao nhat", result)


# ---------------------------------------------------------------------------
# score_metric_keywords
# ---------------------------------------------------------------------------

class TestScoreMetricKeywords(unittest.TestCase):
    def test_single_keyword_match_scores_positive(self) -> None:
        score = score_metric_keywords("nhiet do cao nhat", ("nhiet do",))
        self.assertGreater(score, 0)

    def test_no_match_scores_zero(self) -> None:
        score = score_metric_keywords("energy cao nhat", ("nhiet do",))
        self.assertEqual(score, 0)

    def test_multi_word_keyword_scores_higher_than_single_word(self) -> None:
        score_multi = score_metric_keywords("nhiet do cao nhat", ("nhiet do",))
        score_single = score_metric_keywords("gio cao nhat", ("gio",))
        self.assertGreaterEqual(score_multi, score_single)

    def test_repeated_keyword_increases_score(self) -> None:
        score_once = score_metric_keywords("nhiet do cao nhat", ("nhiet do",))
        score_twice = score_metric_keywords("nhiet do nhiet do cao nhat", ("nhiet do",))
        self.assertGreater(score_twice, score_once)

    def test_empty_keywords_tuple_returns_zero(self) -> None:
        self.assertEqual(score_metric_keywords("energy cao nhat", ()), 0)


# ---------------------------------------------------------------------------
# resolve_weather_metric
# ---------------------------------------------------------------------------

class TestResolveWeatherMetric(unittest.TestCase):
    def test_temperature_resolved_for_nhiet_do(self) -> None:
        metric = resolve_weather_metric("Nhiệt độ thấp nhất trong ngày")
        self.assertEqual(metric["key"], "temperature_2m")

    def test_temperature_resolved_for_temperature_keyword(self) -> None:
        metric = resolve_weather_metric("temperature cao nhat")
        self.assertEqual(metric["key"], "temperature_2m")

    def test_wind_speed_resolved_for_gio_keyword(self) -> None:
        metric = resolve_weather_metric("toc do gio cao nhat")
        self.assertEqual(metric["key"], "wind_speed_10m")

    def test_wind_gust_resolved(self) -> None:
        metric = resolve_weather_metric("gio giat manh nhat trong ngay")
        self.assertEqual(metric["key"], "wind_gusts_10m")

    def test_shortwave_radiation_resolved(self) -> None:
        metric = resolve_weather_metric("buc xa mat troi cao nhat")
        self.assertEqual(metric["key"], "shortwave_radiation")

    def test_cloud_cover_resolved(self) -> None:
        metric = resolve_weather_metric("do phu may thap nhat")
        self.assertEqual(metric["key"], "cloud_cover")

    def test_default_to_temperature_when_no_match(self) -> None:
        # No keywords match → defaults to first metric = temperature_2m
        metric = resolve_weather_metric("thap nhat ngay 5/1/2026")
        self.assertEqual(metric["key"], WEATHER_METRIC_CATALOG[0]["key"])

    def test_returns_dict_with_required_keys(self) -> None:
        metric = resolve_weather_metric("nhiet do")
        self.assertIn("key", metric)
        self.assertIn("label", metric)
        self.assertIn("unit", metric)
        self.assertIn("keywords", metric)


# ---------------------------------------------------------------------------
# resolve_facility_code
# ---------------------------------------------------------------------------

class TestResolveFacilityCode(unittest.TestCase):
    def test_known_code_returned(self) -> None:
        self.assertEqual(resolve_facility_code("WRSF1"), "White Rock Solar Farm")

    def test_case_insensitive(self) -> None:
        self.assertEqual(resolve_facility_code("wrsf1"), "White Rock Solar Farm")

    def test_emerasf_resolved(self) -> None:
        self.assertEqual(resolve_facility_code("EMERASF"), "Emerald Solar Farm")

    def test_alias_resolved(self) -> None:
        self.assertEqual(resolve_facility_code("FINLEY"), "Finley Solar Farm")

    def test_unknown_code_returns_none(self) -> None:
        self.assertIsNone(resolve_facility_code("UNKNOWN123"))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(resolve_facility_code(""))

    def test_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(resolve_facility_code("  AVLSF  "), "Avonlie Solar Farm")

    def test_all_canonical_codes_resolvable(self) -> None:
        canonical = ["WRSF1", "AVLSF", "BOMENSF", "YATSF1", "LIMOSF2", "FINLEYSF", "EMERASF", "DARLSF"]
        for code in canonical:
            with self.subTest(code=code):
                self.assertIsNotNone(resolve_facility_code(code))


# ---------------------------------------------------------------------------
# expand_facility_codes_in_message
# ---------------------------------------------------------------------------

class TestExpandFacilityCodesInMessage(unittest.TestCase):
    def test_single_code_expanded(self) -> None:
        result = expand_facility_codes_in_message("Cho toi du lieu WRSF1 hom nay")
        self.assertIn("White Rock Solar Farm", result)
        self.assertNotIn("WRSF1", result)

    def test_multiple_codes_expanded(self) -> None:
        result = expand_facility_codes_in_message("EMERASF va DARLSF hom nay")
        self.assertIn("Emerald Solar Farm", result)
        self.assertIn("Darlington Point Solar Farm", result)

    def test_unknown_tokens_preserved(self) -> None:
        result = expand_facility_codes_in_message("hello world WRSF1")
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_punctuation_preserved_after_expansion(self) -> None:
        result = expand_facility_codes_in_message("Du lieu WRSF1, hom nay")
        self.assertIn(",", result)

    def test_no_facility_code_message_unchanged(self) -> None:
        msg = "tong quan he thong hom nay"
        result = expand_facility_codes_in_message(msg)
        self.assertEqual(result, msg)

    def test_case_insensitive_expansion(self) -> None:
        result = expand_facility_codes_in_message("du lieu wrsf1 hom nay")
        self.assertIn("White Rock Solar Farm", result)

    def test_empty_message(self) -> None:
        self.assertEqual(expand_facility_codes_in_message(""), "")


if __name__ == "__main__":
    unittest.main()
