"""Extended coverage for BaseRepository utility methods and helpers."""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.base_repository import (
    BaseRepository,
    DatabricksDataUnavailableError,
)


def _make_repo() -> BaseRepository:
    # Force CSV path / no databricks by default
    return BaseRepository(settings=SolarChatSettings())


class TestToFloat:
    def test_none_returns_default(self) -> None:
        assert BaseRepository._to_float(None) == 0.0
        assert BaseRepository._to_float(None, default=5.0) == 5.0

    def test_empty_string_returns_default(self) -> None:
        assert BaseRepository._to_float("") == 0.0

    def test_valid_float(self) -> None:
        assert BaseRepository._to_float("3.14") == 3.14

    def test_invalid_returns_default(self) -> None:
        assert BaseRepository._to_float("not_a_num", default=99.0) == 99.0


class TestParseDatetime:
    def test_none_returns_none(self) -> None:
        assert BaseRepository._parse_datetime(None) is None
        assert BaseRepository._parse_datetime("") is None

    def test_invalid_returns_none(self) -> None:
        assert BaseRepository._parse_datetime("not-a-date") is None

    def test_valid_naive(self) -> None:
        result = BaseRepository._parse_datetime("2026-04-01T10:00:00")
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_with_z_suffix_converts_to_utc(self) -> None:
        result = BaseRepository._parse_datetime("2026-04-01T10:00:00Z")
        assert isinstance(result, datetime)
        assert result.tzinfo is None  # stripped after conversion


class TestExtractIssues:
    def test_empty(self) -> None:
        assert BaseRepository._extract_issues(None) == []
        assert BaseRepository._extract_issues("") == []

    def test_pipe_separated(self) -> None:
        assert BaseRepository._extract_issues("a|b|c") == ["a", "b", "c"]

    def test_filters_empty(self) -> None:
        assert BaseRepository._extract_issues("a||b") == ["a", "b"]


class TestCalculateRSquared:
    def test_short_returns_zero(self) -> None:
        assert BaseRepository._calculate_r_squared([1.0], [2.0]) == 0.0

    def test_mismatched_returns_zero(self) -> None:
        assert BaseRepository._calculate_r_squared([1.0, 2.0], [3.0]) == 0.0

    def test_perfect_correlation(self) -> None:
        r2 = BaseRepository._calculate_r_squared([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert r2 == pytest.approx(1.0)

    def test_zero_variance_returns_zero(self) -> None:
        r2 = BaseRepository._calculate_r_squared([1.0, 1.0, 1.0], [2.0, 4.0, 6.0])
        assert r2 == 0.0


class TestFormatObservedAt:
    def test_none(self) -> None:
        assert BaseRepository._format_observed_at(None) == ""

    def test_datetime_instance(self) -> None:
        dt = datetime(2026, 4, 1, 10, 30)
        assert BaseRepository._format_observed_at(dt) == "2026-04-01T10:30"

    def test_iso_string_with_z(self) -> None:
        assert BaseRepository._format_observed_at("2026-04-01T10:30:00Z") == "2026-04-01T10:30"

    def test_invalid_returns_str(self) -> None:
        assert BaseRepository._format_observed_at("garbage") == "garbage"


class TestResolvePeriodWindow:
    def test_hour_with_specific(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("hour", date(2026, 4, 1), specific_hour=10)
        assert start == datetime(2026, 4, 1, 10, 0)
        assert end == datetime(2026, 4, 1, 11, 0)
        assert "10" in label

    def test_hour_without_specific(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("hour", date(2026, 4, 1))
        assert start == datetime(2026, 4, 1, 0, 0)
        assert end == datetime(2026, 4, 2, 0, 0)

    def test_24h(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("24h", date(2026, 4, 1))
        assert (end - start).total_seconds() == 86400

    def test_week(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("week", date(2026, 4, 3))  # Friday
        assert start.weekday() == 0  # Monday
        assert (end - start).days == 7

    def test_month(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("month", date(2026, 4, 15))
        assert start == datetime(2026, 4, 1)
        assert end == datetime(2026, 5, 1)
        assert label == "2026-04"

    def test_month_december_rolls_over(self) -> None:
        start, end, _ = BaseRepository._resolve_period_window("month", date(2026, 12, 15))
        assert end == datetime(2027, 1, 1)

    def test_year(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("year", date(2026, 4, 15))
        assert start == datetime(2026, 1, 1)
        assert end == datetime(2027, 1, 1)
        assert label == "2026"

    def test_all_time(self) -> None:
        start, end, label = BaseRepository._resolve_period_window("all_time", date(2026, 4, 15))
        assert start.year == 2016
        assert label == "all time"

    def test_unknown_falls_back_to_day(self) -> None:
        start, end, _ = BaseRepository._resolve_period_window("unknown", date(2026, 4, 1))
        assert (end - start).days == 1


class TestResolveLatestDate:
    def test_returns_date_from_iso_string(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", return_value=[{"latest": "2026-04-01"}]):
            assert repo._resolve_latest_date("silver.weather") == date(2026, 4, 1)

    def test_returns_date_from_date_object(self) -> None:
        repo = _make_repo()
        d = date(2026, 3, 15)
        with patch.object(repo, "_execute_query", return_value=[{"latest": d}]):
            assert repo._resolve_latest_date("silver.air_quality") == d

    def test_falls_back_on_error(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", side_effect=DatabricksDataUnavailableError("down")):
            result = repo._resolve_latest_date("silver.weather")
        assert result == date.today()

    def test_empty_rows_returns_today(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", return_value=[]):
            assert repo._resolve_latest_date("silver.weather") == date.today()


class TestResolveLatestDatetime:
    def test_returns_formatted_datetime(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", return_value=[{"latest": datetime(2026, 4, 1, 10, 30)}]):
            result = repo._resolve_latest_datetime("silver.weather")
        assert result == "2026-04-01T10:30"

    def test_error_returns_empty(self) -> None:
        repo = _make_repo()
        with patch.object(repo, "_execute_query", side_effect=DatabricksDataUnavailableError("x")):
            assert repo._resolve_latest_datetime("silver.weather") == ""


class TestWithDatabricksQuery:
    def test_happy_path_tags_sources(self) -> None:
        repo = _make_repo()
        sources_template = [{"layer": "Gold", "dataset": "gold.x"}]
        metrics, sources = repo._with_databricks_query(
            "topic_x",
            lambda: {"k": 1},
            sources_template,
        )
        assert metrics == {"k": 1}
        assert sources[0]["data_source"] == "databricks"

    def test_raises_databricks_unavailable(self) -> None:
        repo = _make_repo()
        def raiser():
            raise DatabricksDataUnavailableError("no connection")
        with pytest.raises(DatabricksDataUnavailableError):
            repo._with_databricks_query("topic", raiser, [{"layer": "Gold", "dataset": "g"}])

    def test_wraps_unexpected_errors(self) -> None:
        repo = _make_repo()
        def raiser():
            raise ValueError("bad")
        with pytest.raises(DatabricksDataUnavailableError):
            repo._with_databricks_query("topic", raiser, [{"layer": "Gold", "dataset": "g"}])

    def test_four_arg_signature_with_fallback_and_template(self) -> None:
        """When sources_template is provided as the 4th arg, it takes precedence
        over the 3rd positional arg (fallback_fn)."""
        repo = _make_repo()
        sources_template = [{"layer": "Gold", "dataset": "gold.x"}]
        fallback_fn = lambda: {"should": "not be used"}
        metrics, sources = repo._with_databricks_query(
            "topic_x",
            lambda: {"k": 1},
            fallback_fn,
            sources_template,
        )
        assert metrics == {"k": 1}
        assert sources[0]["dataset"] == "gold.x"


class TestResolveFacility:
    def test_prefers_facility_name(self) -> None:
        assert BaseRepository._resolve_facility({"facility_name": "A", "facility_id": "B"}) == "A"

    def test_falls_through(self) -> None:
        assert BaseRepository._resolve_facility({"facility_id": "B"}) == "B"
        assert BaseRepository._resolve_facility({"location_id": "L"}) == "L"

    def test_unknown_fallback(self) -> None:
        assert BaseRepository._resolve_facility({}) == "Unknown"
