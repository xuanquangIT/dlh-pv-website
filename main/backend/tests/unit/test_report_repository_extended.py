"""Extended coverage for ReportRepository internals."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.report_repository import ReportRepository


def _make_repo() -> ReportRepository:
    return ReportRepository(settings=SolarChatSettings())


# ---------------------------------------------------------------------------
# _normalize_station_filter
# ---------------------------------------------------------------------------

class TestNormalizeStationFilter:
    def test_none_returns_none(self) -> None:
        assert ReportRepository._normalize_station_filter(None) is None

    def test_empty_returns_none(self) -> None:
        assert ReportRepository._normalize_station_filter("   ") is None

    def test_sentinel_words_return_none(self) -> None:
        for sentinel in ("all", "any", "none", "null", "ALL", "Any"):
            assert ReportRepository._normalize_station_filter(sentinel) is None

    def test_short_code_kept_unchanged(self) -> None:
        assert ReportRepository._normalize_station_filter("AVLSF") == "AVLSF"
        assert ReportRepository._normalize_station_filter("DARLSF") == "DARLSF"

    def test_strips_solar_farm_suffix(self) -> None:
        assert ReportRepository._normalize_station_filter("Avonlie Solar Farm") == "Avonlie"

    def test_strips_station_suffix(self) -> None:
        result = ReportRepository._normalize_station_filter("Darlington Point Station")
        assert result == "Darlington Point"

    def test_keeps_original_if_stripped_too_short(self) -> None:
        result = ReportRepository._normalize_station_filter("AB Solar Farm")
        assert result == "AB Solar Farm"


# ---------------------------------------------------------------------------
# _normalize_date_value
# ---------------------------------------------------------------------------

class TestNormalizeDateValue:
    def test_none(self) -> None:
        assert ReportRepository._normalize_date_value(None) is None

    def test_date_object(self) -> None:
        assert ReportRepository._normalize_date_value(date(2026, 4, 1)) == "2026-04-01"

    def test_string_passthrough(self) -> None:
        assert ReportRepository._normalize_date_value("2026-04-01") == "2026-04-01"


# ---------------------------------------------------------------------------
# _merge_date_ranges
# ---------------------------------------------------------------------------

class TestMergeDateRanges:
    def test_none_inputs_return_next(self) -> None:
        mn, mx = ReportRepository._merge_date_ranges(None, None, "2026-01-01", "2026-04-01")
        assert mn == "2026-01-01" and mx == "2026-04-01"

    def test_widens_range(self) -> None:
        mn, mx = ReportRepository._merge_date_ranges("2026-02-01", "2026-03-01", "2026-01-01", "2026-04-01")
        assert mn == "2026-01-01" and mx == "2026-04-01"

    def test_keeps_current_when_next_narrower(self) -> None:
        mn, mx = ReportRepository._merge_date_ranges("2026-01-01", "2026-05-01", "2026-02-01", "2026-04-01")
        assert mn == "2026-01-01" and mx == "2026-05-01"

    def test_ignores_none_next(self) -> None:
        mn, mx = ReportRepository._merge_date_ranges("2026-01-01", "2026-05-01", None, None)
        assert mn == "2026-01-01" and mx == "2026-05-01"


# ---------------------------------------------------------------------------
# _report_sources_for_metrics
# ---------------------------------------------------------------------------

class TestReportSourcesForMetrics:
    def test_energy_only(self) -> None:
        sources = ReportRepository._report_sources_for_metrics({"energy_mwh"})
        assert sources == [("gold.mart_energy_daily", "lh_silver_clean_hourly_energy.csv")]

    def test_weather_metrics(self) -> None:
        sources = ReportRepository._report_sources_for_metrics({"temperature_2m", "cloud_cover"})
        assert ("silver.weather", "lh_silver_clean_hourly_weather.csv") in sources

    def test_aqi(self) -> None:
        sources = ReportRepository._report_sources_for_metrics({"aqi_value"})
        assert ("silver.air_quality", "lh_silver_clean_hourly_air_quality.csv") in sources

    def test_all_three(self) -> None:
        sources = ReportRepository._report_sources_for_metrics(
            {"energy_mwh", "temperature_2m", "aqi_value"}
        )
        assert len(sources) == 3


# ---------------------------------------------------------------------------
# _build_station_filter_clause
# ---------------------------------------------------------------------------

class TestBuildStationFilterClause:
    def test_no_station_returns_empty(self) -> None:
        assert ReportRepository._build_station_filter_clause(None, "facility_name") == ""

    def test_sentinel_returns_empty(self) -> None:
        assert ReportRepository._build_station_filter_clause("all", "facility_name") == ""

    def test_normal_name_produces_like_clause(self) -> None:
        clause = ReportRepository._build_station_filter_clause("Avonlie Solar Farm", "facility_name")
        assert "LIKE LOWER('%Avonlie%')" in clause
        assert "LOWER(facility_name)" in clause

    def test_escapes_sql_wildcards(self) -> None:
        clause = ReportRepository._build_station_filter_clause("Av_On%lie", "facility_name")
        # underscores and percents escaped
        assert "\\%" in clause or "\\_" in clause


# ---------------------------------------------------------------------------
# fetch_station_hourly_report
# ---------------------------------------------------------------------------

class TestFetchStationHourlyReport:
    def test_with_explicit_date_returns_hourly_rows(self) -> None:
        repo = _make_repo()
        mock_rows = [
            {"hr": 10, "facility": "Avonlie", "energy_mwh": 12.3456, "capacity_factor_pct": 45.12},
            {"hr": 11, "facility": "Avonlie", "energy_mwh": 15.678, "capacity_factor_pct": None},
        ]
        with patch.object(repo, "_execute_query", return_value=mock_rows):
            result, sources = repo.fetch_station_hourly_report(
                station_name="Avonlie", anchor_date=date(2026, 4, 1)
            )

        assert result["report_date"] == "2026-04-01"
        assert result["row_count"] == 2
        assert result["total_energy_mwh"] == round(12.3456 + 15.678, 4)
        assert result["has_data"] is True
        assert result["hourly_rows"][0]["hour"] == 10
        assert result["hourly_rows"][0]["capacity_factor_pct"] == 45.12
        assert result["hourly_rows"][1]["capacity_factor_pct"] is None
        assert len(sources) == 2

    def test_no_date_uses_max_date_from_query(self) -> None:
        repo = _make_repo()
        call_count = {"n": 0}

        def fake_execute(sql: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [{"max_date": date(2026, 3, 15)}]
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            result, _ = repo.fetch_station_hourly_report(station_name=None, anchor_date=None)
        assert result["report_date"] == "2026-03-15"
        assert result["has_data"] is False

    def test_no_date_string_max_date(self) -> None:
        repo = _make_repo()
        call_count = {"n": 0}

        def fake_execute(sql: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [{"max_date": "2026-02-20"}]
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            result, _ = repo.fetch_station_hourly_report(station_name=None, anchor_date=None)
        assert result["report_date"] == "2026-02-20"

    def test_no_date_invalid_string_falls_back_today(self) -> None:
        repo = _make_repo()
        call_count = {"n": 0}

        def fake_execute(sql: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [{"max_date": "not-a-date"}]
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            result, _ = repo.fetch_station_hourly_report(station_name=None, anchor_date=None)
        assert result["report_date"] == date.today().isoformat()

    def test_no_date_empty_result_falls_back_today(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", return_value=[]):
            result, _ = repo.fetch_station_hourly_report(station_name=None, anchor_date=None)
        assert result["report_date"] == date.today().isoformat()

    def test_station_filter_applied(self) -> None:
        repo = _make_repo()
        captured_sql: list[str] = []

        def fake_execute(sql: str):
            captured_sql.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo.fetch_station_hourly_report(station_name="Avonlie Solar Farm", anchor_date=date(2026, 4, 1))

        joined = " ".join(captured_sql)
        assert "Avonlie" in joined


# ---------------------------------------------------------------------------
# _station_report_date_range_databricks
# ---------------------------------------------------------------------------

class TestStationReportDateRangeDatabricks:
    def test_merges_ranges_across_tables(self) -> None:
        repo = _make_repo()
        responses = [
            [{"min_date": date(2026, 1, 1), "max_date": date(2026, 4, 1)}],   # energy
            [{"min_date": date(2025, 12, 1), "max_date": date(2026, 3, 1)}],  # weather
            [{"min_date": date(2026, 2, 1), "max_date": date(2026, 5, 1)}],   # aqi
        ]
        with patch.object(repo, "_execute_query", side_effect=responses):
            mn, mx = repo._station_report_date_range_databricks(
                {"energy_mwh", "temperature_2m", "aqi_value"}
            )
        assert mn == "2025-12-01"
        assert mx == "2026-05-01"

    def test_empty_rows_skipped(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", return_value=[]):
            mn, mx = repo._station_report_date_range_databricks({"energy_mwh"})
        assert mn is None and mx is None


# ---------------------------------------------------------------------------
# _station_daily_report_databricks
# ---------------------------------------------------------------------------

class TestStationDailyReportDatabricks:
    def test_fetches_from_gold_mart_happy_path(self) -> None:
        repo = _make_repo()
        mart_rows = [
            {
                "facility": "Avonlie",
                "energy_mwh": 125.4321,
                "shortwave_radiation": 450.78,
                "temperature_2m": 24.567,
                "aqi_value": 15.333,
                "weighted_capacity_factor_pct": 40.0,
                "dominant_weather_condition": "Clear",
            },
        ]
        with patch.object(repo, "_execute_query", return_value=mart_rows):
            stations = repo._station_daily_report_databricks(
                date(2026, 4, 1),
                {"energy_mwh", "shortwave_radiation", "temperature_2m", "aqi_value"},
            )
        assert len(stations) == 1
        s = stations[0]
        assert s["facility"] == "Avonlie"
        assert s["energy_mwh"] == 125.4321
        assert s["shortwave_radiation"] == 450.78
        assert s["temperature_2m"] == 24.57
        assert s["aqi_value"] == 15.33

    def test_fallback_to_silver_when_mart_empty(self) -> None:
        repo = _make_repo()
        call_count = {"n": 0}

        def fake_execute(sql: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Gold mart returns nothing
                return []
            if call_count["n"] == 2:
                # silver.energy_readings
                return [{"facility": "Avonlie", "total_energy_kwh": 100000.0}]
            if call_count["n"] == 3:
                # silver.weather
                return [{"facility": "Avonlie", "avg_temperature_2m": 25.0}]
            if call_count["n"] == 4:
                # silver.air_quality
                return [{"facility": "Avonlie", "avg_aqi": 12.5}]
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            stations = repo._station_daily_report_databricks(
                date(2026, 4, 1),
                {"energy_mwh", "temperature_2m", "aqi_value"},
            )
        assert len(stations) == 1
        assert stations[0]["energy_mwh"] == 100.0
        assert stations[0]["temperature_2m"] == 25.0
        assert stations[0]["aqi_value"] == 12.5

    def test_station_filter_in_mart_query(self) -> None:
        repo = _make_repo()
        captured: list[str] = []

        def fake_execute(sql: str):
            captured.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo._station_daily_report_databricks(
                date(2026, 4, 1), {"energy_mwh"}, station_name="Avonlie",
            )

        joined = " ".join(captured)
        assert "Avonlie" in joined
