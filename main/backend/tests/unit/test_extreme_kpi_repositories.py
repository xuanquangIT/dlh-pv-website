"""Unit tests for ExtremeRepository and KpiRepository.

Mocking strategy:
  - All Databricks I/O is isolated by patching `_execute_query` on the repo
    instance, or by patching `_databricks_connection` with a MagicMock context
    manager that exposes a fake cursor.
  - SQL construction is verified by capturing the `sql` argument passed to
    `_execute_query`.
  - Data transformation is verified by asserting on the returned metrics dict.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.base_repository import (
    BaseRepository,
    DatabricksDataUnavailableError,
)
from app.repositories.solar_ai_chat.extreme_repository import ExtremeRepository
from app.repositories.solar_ai_chat.kpi_repository import KpiRepository


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> SolarChatSettings:
    s = SolarChatSettings()
    s.databricks_host = overrides.get("databricks_host", "https://adb-123.azuredatabricks.net")
    s.databricks_token = overrides.get("databricks_token", "dapi-test-token")
    s.databricks_sql_http_path = overrides.get(
        "databricks_sql_http_path", "/sql/1.0/warehouses/abc123"
    )
    s.uc_catalog = overrides.get("uc_catalog", "pv")
    s.uc_silver_schema = overrides.get("uc_silver_schema", "silver")
    s.uc_gold_schema = overrides.get("uc_gold_schema", "gold")
    return s


def _make_extreme_repo(**overrides) -> ExtremeRepository:
    return ExtremeRepository(settings=_make_settings(**overrides))


def _make_kpi_repo(**overrides) -> KpiRepository:
    return KpiRepository(settings=_make_settings(**overrides))


# Fake anchor date used across tests
ANCHOR = date(2026, 4, 15)

# ---------------------------------------------------------------------------
# Fake cursor / connection helpers
# ---------------------------------------------------------------------------

def _fake_connection_ctx(rows: list[tuple], col_names: list[str]):
    """Return a context manager that yields a fake connection with a cursor."""
    cursor = MagicMock()
    cursor.description = [(c,) for c in col_names]
    cursor.fetchall.return_value = rows

    conn = MagicMock()
    conn.cursor.return_value = cursor

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, cursor


# ===========================================================================
# BaseRepository._execute_query (tested via ExtremeRepository)
# ===========================================================================

class TestExecuteQuery:
    def test_execute_query_returns_dicts(self):
        repo = _make_extreme_repo()
        ctx, _ = _fake_connection_ctx(
            rows=[("WRSF1", 42.5, datetime(2026, 4, 15, 10))],
            col_names=["facility", "metric_value", "observed_at"],
        )
        with patch.object(repo, "_databricks_connection", return_value=ctx):
            result = repo._execute_query("SELECT 1")
        assert len(result) == 1
        assert result[0]["facility"] == "WRSF1"
        assert result[0]["metric_value"] == 42.5

    def test_execute_query_empty_result(self):
        repo = _make_extreme_repo()
        ctx, _ = _fake_connection_ctx(rows=[], col_names=["facility", "metric_value", "observed_at"])
        with patch.object(repo, "_databricks_connection", return_value=ctx):
            result = repo._execute_query("SELECT 1")
        assert result == []

    def test_execute_query_wraps_generic_exception(self):
        repo = _make_extreme_repo()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=RuntimeError("connection refused"))
        ctx.__exit__ = MagicMock(return_value=False)
        with patch.object(repo, "_databricks_connection", return_value=ctx):
            with pytest.raises(DatabricksDataUnavailableError):
                repo._execute_query("SELECT 1")

    def test_execute_query_reraises_databricks_unavailable(self):
        repo = _make_extreme_repo()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(
            side_effect=DatabricksDataUnavailableError("already unavailable")
        )
        ctx.__exit__ = MagicMock(return_value=False)
        with patch.object(repo, "_databricks_connection", return_value=ctx):
            with pytest.raises(DatabricksDataUnavailableError, match="already unavailable"):
                repo._execute_query("SELECT 1")


# ===========================================================================
# ExtremeRepository — fetch_extreme_aqi
# ===========================================================================

class TestFetchExtremeAqi:
    """Tests for ExtremeRepository.fetch_extreme_aqi."""

    def _aqi_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "facility": f"STATION_{i}",
                "metric_value": float(200 - i * 10),
                "observed_at": datetime(2026, 4, 15, 10 + i),
                "aqi_category": "Moderate" if i % 2 == 0 else "Good",
            }
            for i in range(n)
        ]

    def test_highest_aqi_returns_correct_metric_keys(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(3)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, sources = repo.fetch_extreme_aqi(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
            )
        assert metrics["extreme_metric"] == "aqi"
        assert metrics["aqi_query_type"] == "highest"
        assert "highest_station" in metrics
        assert "highest_aqi_value" in metrics
        assert "highest_aqi_category" in metrics
        assert "top_highest_stations" in metrics

    def test_lowest_aqi_returns_correct_metric_keys(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(3)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, sources = repo.fetch_extreme_aqi(
                query_type="lowest",
                timeframe="day",
                anchor_date=ANCHOR,
            )
        assert "lowest_station" in metrics
        assert "lowest_aqi_value" in metrics
        assert "lowest_aqi_category" in metrics

    def test_aqi_source_is_silver_air_quality(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            _, sources = repo.fetch_extreme_aqi(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
            )
        assert sources[0]["dataset"] == "silver.air_quality"
        assert sources[0]["layer"] == "Silver"

    def test_aqi_value_rounded_to_2dp(self):
        repo = _make_extreme_repo()
        rows = [
            {
                "facility": "WRSF1",
                "metric_value": 123.456789,
                "observed_at": datetime(2026, 4, 15, 10),
                "aqi_category": "Good",
            }
        ]
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_aqi(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        assert metrics["highest_aqi_value"] == round(123.456789, 2)

    def test_top_stations_capped_at_5(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(10)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_aqi(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        assert len(metrics["top_highest_stations"]) <= 5

    def test_aqi_no_rows_raises_value_error(self):
        repo = _make_extreme_repo()
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=[]):
            with pytest.raises(ValueError, match="No AQI data"):
                repo.fetch_extreme_aqi(
                    query_type="highest", timeframe="day", anchor_date=ANCHOR
                )

    def test_aqi_uses_anchor_date_directly_when_provided(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(1)
        resolve_mock = MagicMock(return_value=ANCHOR)
        with patch.object(repo, "_resolve_latest_date", resolve_mock), \
             patch.object(repo, "_execute_query", return_value=rows):
            repo.fetch_extreme_aqi(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        # _resolve_latest_date should NOT be called since anchor_date was supplied
        resolve_mock.assert_not_called()

    def test_aqi_calls_resolve_latest_date_when_no_anchor(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR) as mock_resolve, \
             patch.object(repo, "_execute_query", return_value=rows):
            repo.fetch_extreme_aqi(
                query_type="highest", timeframe="day", anchor_date=None
            )
        mock_resolve.assert_called_once()

    def test_aqi_missing_category_defaults_to_unknown(self):
        repo = _make_extreme_repo()
        rows = [
            {
                "facility": "WRSF1",
                "metric_value": 99.0,
                "observed_at": datetime(2026, 4, 15, 8),
                # no aqi_category key
            }
        ]
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_aqi(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        assert metrics["highest_aqi_category"] == "Unknown"

    def test_aqi_specific_hour_present_in_base_metrics(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_aqi(
                query_type="highest",
                timeframe="hour",
                anchor_date=ANCHOR,
                specific_hour=10,
            )
        assert metrics["specific_hour"] == 10

    def test_aqi_query_type_normalised_to_highest_lowest(self):
        repo = _make_extreme_repo()
        rows = self._aqi_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_aqi(
                query_type="max",  # not "highest" but truthy-as-not-highest → "lowest"
                timeframe="day",
                anchor_date=ANCHOR,
            )
        # "max" != "highest" so qt == "lowest"
        assert metrics["aqi_query_type"] == "lowest"


# ===========================================================================
# ExtremeRepository — fetch_extreme_energy
# ===========================================================================

class TestFetchExtremeEnergy:
    """Tests for ExtremeRepository.fetch_extreme_energy."""

    def _energy_rows(self, n: int = 2) -> list[dict]:
        return [
            {
                "facility": f"STATION_{i}",
                "metric_value": float((5 - i) * 1000),  # kWh
                "observed_at": datetime(2026, 4, 15, 9 + i),
            }
            for i in range(n)
        ]

    def test_energy_mwh_conversion(self):
        repo = _make_extreme_repo()
        rows = [
            {
                "facility": "WRSF1",
                "metric_value": 5000.0,  # 5 MWh
                "observed_at": datetime(2026, 4, 15, 9),
            }
        ]
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_energy(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        assert metrics["highest_energy_mwh"] == pytest.approx(5.0, rel=1e-3)

    def test_energy_highest_returns_correct_keys(self):
        repo = _make_extreme_repo()
        rows = self._energy_rows(3)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, sources = repo.fetch_extreme_energy(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        assert metrics["extreme_metric"] == "energy"
        assert "highest_station" in metrics
        assert "highest_energy_mwh" in metrics
        assert "top_highest_stations" in metrics
        assert sources[0]["dataset"] == "silver.energy_readings"

    def test_energy_lowest_returns_correct_keys(self):
        repo = _make_extreme_repo()
        rows = self._energy_rows(2)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_energy(
                query_type="lowest", timeframe="day", anchor_date=ANCHOR
            )
        assert "lowest_station" in metrics
        assert "lowest_energy_mwh" in metrics

    def test_energy_no_rows_raises_value_error(self):
        repo = _make_extreme_repo()
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=[]):
            with pytest.raises(ValueError, match="No Energy data"):
                repo.fetch_extreme_energy(
                    query_type="highest", timeframe="day", anchor_date=ANCHOR
                )

    def test_energy_top_stations_structure(self):
        repo = _make_extreme_repo()
        rows = self._energy_rows(4)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_energy(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )
        for station in metrics["top_highest_stations"]:
            assert "facility" in station
            assert "energy_mwh" in station

    def test_energy_week_timeframe(self):
        repo = _make_extreme_repo()
        rows = self._energy_rows(2)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_energy(
                query_type="highest", timeframe="week", anchor_date=ANCHOR
            )
        assert metrics["timeframe"] == "week"

    def test_energy_month_timeframe(self):
        repo = _make_extreme_repo()
        rows = self._energy_rows(2)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_energy(
                query_type="highest", timeframe="month", anchor_date=ANCHOR
            )
        assert metrics["timeframe"] == "month"

    def test_energy_sql_contains_energy_readings_table(self):
        repo = _make_extreme_repo()
        captured_sqls: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured_sqls.append(sql)
            return [
                {
                    "facility": "WRSF1",
                    "metric_value": 3000.0,
                    "observed_at": datetime(2026, 4, 15, 8),
                }
            ]

        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo.fetch_extreme_energy(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )

        assert any("silver.energy_readings" in sql for sql in captured_sqls)

    def test_energy_sql_order_desc_for_highest(self):
        repo = _make_extreme_repo()
        captured_sqls: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured_sqls.append(sql)
            return [
                {
                    "facility": "WRSF1",
                    "metric_value": 3000.0,
                    "observed_at": datetime(2026, 4, 15, 8),
                }
            ]

        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo.fetch_extreme_energy(
                query_type="highest", timeframe="day", anchor_date=ANCHOR
            )

        extreme_sql = [s for s in captured_sqls if "energy_readings" in s]
        assert extreme_sql
        assert "DESC" in extreme_sql[0]

    def test_energy_sql_order_asc_for_lowest(self):
        repo = _make_extreme_repo()
        captured_sqls: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured_sqls.append(sql)
            return [
                {
                    "facility": "WRSF1",
                    "metric_value": 500.0,
                    "observed_at": datetime(2026, 4, 15, 8),
                }
            ]

        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo.fetch_extreme_energy(
                query_type="lowest", timeframe="day", anchor_date=ANCHOR
            )

        extreme_sql = [s for s in captured_sqls if "energy_readings" in s]
        assert extreme_sql
        assert "ASC" in extreme_sql[0]


# ===========================================================================
# ExtremeRepository — fetch_extreme_weather
# ===========================================================================

class TestFetchExtremeWeather:
    """Tests for ExtremeRepository.fetch_extreme_weather."""

    def _weather_rows(self, n: int = 2) -> list[dict]:
        return [
            {
                "facility": f"STATION_{i}",
                "metric_value": float(35.0 - i * 5),
                "observed_at": datetime(2026, 4, 15, 12 + i),
            }
            for i in range(n)
        ]

    def test_weather_highest_temperature_keys(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(2)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, sources = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="temperature_2m",
                weather_metric_label="Temperature",
                weather_unit="°C",
            )
        assert metrics["extreme_metric"] == "weather"
        assert metrics["weather_metric_label"] == "Temperature"
        assert metrics["weather_unit"] == "°C"
        assert "highest_station" in metrics
        assert "highest_weather_value" in metrics
        assert "top_highest_stations" in metrics
        assert sources[0]["dataset"] == "silver.weather"

    def test_weather_column_map_resolves_temperature_2m(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(1)
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            return rows

        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", side_effect=fake_execute):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="temperature_2m",
                weather_metric_label="Temperature",
                weather_unit="°C",
            )
        # resolved column should appear
        assert metrics["weather_metric"] == "temperature_c"

    def test_weather_column_map_resolves_wind_speed(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="wind_speed_10m",
                weather_metric_label="Wind Speed",
                weather_unit="m/s",
            )
        assert metrics["weather_metric"] == "wind_speed_ms"

    def test_weather_column_map_resolves_wind_gusts(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="wind_gusts_10m",
                weather_metric_label="Wind Gust",
                weather_unit="m/s",
            )
        assert metrics["weather_metric"] == "wind_gust_ms"

    def test_weather_column_map_resolves_cloud_cover(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="cloud_cover",
                weather_metric_label="Cloud Cover",
                weather_unit="%",
            )
        assert metrics["weather_metric"] == "cloud_cover_pct"

    def test_weather_no_rows_raises_value_error(self):
        repo = _make_extreme_repo()
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=[]):
            with pytest.raises(ValueError, match="No Weather data"):
                repo.fetch_extreme_weather(
                    query_type="highest",
                    timeframe="day",
                    anchor_date=ANCHOR,
                    weather_metric="temperature_2m",
                    weather_metric_label="Temperature",
                    weather_unit="°C",
                )

    def test_weather_lowest_query(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(2)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="lowest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="temperature_2m",
                weather_metric_label="Temperature",
                weather_unit="°C",
            )
        assert "lowest_station" in metrics
        assert "lowest_weather_value" in metrics

    def test_weather_value_rounded_to_2dp(self):
        repo = _make_extreme_repo()
        rows = [
            {
                "facility": "AVLSF",
                "metric_value": 32.12345,
                "observed_at": datetime(2026, 4, 15, 14),
            }
        ]
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="day",
                anchor_date=ANCHOR,
                weather_metric="temperature_2m",
                weather_metric_label="Temperature",
                weather_unit="°C",
            )
        assert metrics["highest_weather_value"] == round(32.12345, 2)

    def test_weather_all_time_timeframe(self):
        repo = _make_extreme_repo()
        rows = self._weather_rows(1)
        with patch.object(repo, "_resolve_latest_date", return_value=ANCHOR), \
             patch.object(repo, "_execute_query", return_value=rows):
            metrics, _ = repo.fetch_extreme_weather(
                query_type="highest",
                timeframe="all_time",
                anchor_date=ANCHOR,
                weather_metric="temperature_2m",
                weather_metric_label="Temperature",
                weather_unit="°C",
            )
        assert metrics["period_label"] == "all time"


# ===========================================================================
# ExtremeRepository — _query_extreme_rows (SQL construction)
# ===========================================================================

class TestQueryExtremeRows:
    """Verify SQL identifier validation in _query_extreme_rows."""

    def test_disallowed_table_raises_value_error(self):
        repo = _make_extreme_repo()
        with pytest.raises(ValueError, match="not allowed"):
            repo._query_extreme_rows(
                table="malicious.table",
                value_column="aqi_value",
                order="DESC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
            )

    def test_disallowed_column_raises_value_error(self):
        repo = _make_extreme_repo()
        with pytest.raises(ValueError, match="not allowed"):
            repo._query_extreme_rows(
                table="silver.air_quality",
                value_column="drop_table",
                order="DESC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
            )

    def test_air_quality_uses_aqi_timestamp_local(self):
        repo = _make_extreme_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo._query_extreme_rows(
                table="silver.air_quality",
                value_column="aqi_value",
                order="DESC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
            )
        assert captured
        assert "aqi_timestamp_local" in captured[0]

    def test_weather_uses_weather_timestamp_local(self):
        repo = _make_extreme_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo._query_extreme_rows(
                table="silver.weather",
                value_column="temperature_c",
                order="ASC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
            )
        assert "weather_timestamp_local" in captured[0]

    def test_energy_readings_uses_date_hour_local(self):
        repo = _make_extreme_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo._query_extreme_rows(
                table="silver.energy_readings",
                value_column="energy_kwh",
                order="DESC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
            )
        assert "date_hour_local" in captured[0]

    def test_extra_columns_appended_to_select(self):
        repo = _make_extreme_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            return []

        with patch.object(repo, "_execute_query", side_effect=fake_execute):
            repo._query_extreme_rows(
                table="silver.air_quality",
                value_column="aqi_value",
                order="DESC",
                window_start=datetime(2026, 4, 15, 0),
                window_end=datetime(2026, 4, 16, 0),
                extra_columns=("aqi_category",),
            )
        assert "aqi_category" in captured[0]


# ===========================================================================
# ExtremeRepository — _deduplicate_stations
# ===========================================================================

class TestDeduplicateStations:
    def test_deduplicates_by_facility(self):
        repo = _make_extreme_repo()
        rows = [
            {"facility": "WRSF1", "metric_value": 100.0},
            {"facility": "WRSF1", "metric_value": 90.0},
            {"facility": "AVLSF", "metric_value": 80.0},
        ]
        result = repo._deduplicate_stations(rows, highest=True)
        facilities = [r["facility"] for r in result]
        assert len(facilities) == 2
        assert "WRSF1" in facilities
        assert "AVLSF" in facilities

    def test_first_occurrence_kept_for_each_facility(self):
        repo = _make_extreme_repo()
        rows = [
            {"facility": "WRSF1", "metric_value": 100.0},
            {"facility": "WRSF1", "metric_value": 50.0},
        ]
        result = repo._deduplicate_stations(rows, highest=True)
        assert len(result) == 1
        assert result[0]["metric_value"] == 100.0

    def test_sorted_descending_for_highest(self):
        repo = _make_extreme_repo()
        rows = [
            {"facility": "A", "metric_value": 50.0},
            {"facility": "B", "metric_value": 100.0},
            {"facility": "C", "metric_value": 75.0},
        ]
        result = repo._deduplicate_stations(rows, highest=True)
        values = [r["metric_value"] for r in result]
        assert values == sorted(values, reverse=True)

    def test_sorted_ascending_for_lowest(self):
        repo = _make_extreme_repo()
        rows = [
            {"facility": "A", "metric_value": 50.0},
            {"facility": "B", "metric_value": 100.0},
            {"facility": "C", "metric_value": 75.0},
        ]
        result = repo._deduplicate_stations(rows, highest=False)
        values = [r["metric_value"] for r in result]
        assert values == sorted(values)

    def test_facility_none_grouped_as_unknown(self):
        repo = _make_extreme_repo()
        rows = [
            {"facility": None, "metric_value": 10.0},
            {"facility": None, "metric_value": 5.0},
        ]
        result = repo._deduplicate_stations(rows, highest=True)
        assert len(result) == 1
        assert result[0]["facility"] is None  # stored under "Unknown" key, returned as-is


# ===========================================================================
# KpiRepository — _resolve_table_name
# ===========================================================================

class TestKpiResolveTableName:
    def test_canonical_names_resolve_to_themselves(self):
        for key in ("aqi_impact", "energy", "forecast_accuracy", "system_kpi", "weather_impact"):
            assert KpiRepository._resolve_table_name(key) == key

    def test_alias_weather_resolves_to_weather_impact(self):
        assert KpiRepository._resolve_table_name("weather") == "weather_impact"

    def test_alias_aqi_resolves_to_aqi_impact(self):
        assert KpiRepository._resolve_table_name("aqi") == "aqi_impact"

    def test_alias_forecast_resolves_to_forecast_accuracy(self):
        assert KpiRepository._resolve_table_name("forecast") == "forecast_accuracy"

    def test_alias_system_resolves_to_system_kpi(self):
        assert KpiRepository._resolve_table_name("system") == "system_kpi"

    def test_alias_kpi_resolves_to_system_kpi(self):
        assert KpiRepository._resolve_table_name("kpi") == "system_kpi"

    def test_alias_energy_daily_resolves_to_energy(self):
        assert KpiRepository._resolve_table_name("energy_daily") == "energy"

    def test_alias_temperature_resolves_to_energy(self):
        assert KpiRepository._resolve_table_name("temperature") == "energy"

    def test_alias_performance_ratio_resolves_to_energy(self):
        assert KpiRepository._resolve_table_name("performance_ratio") == "energy"

    def test_alias_accuracy_resolves_to_forecast_accuracy(self):
        assert KpiRepository._resolve_table_name("accuracy") == "forecast_accuracy"

    def test_unknown_name_returns_none(self):
        assert KpiRepository._resolve_table_name("nonexistent_table") is None

    def test_empty_string_returns_none(self):
        assert KpiRepository._resolve_table_name("") is None

    def test_case_insensitive_alias_resolution(self):
        # aliases stored lowercase; method lowercases input
        assert KpiRepository._resolve_table_name("AQI") == "aqi_impact"

    def test_leading_trailing_whitespace_stripped(self):
        assert KpiRepository._resolve_table_name("  energy  ") == "energy"

    def test_pr_vs_temperature_resolves_to_energy(self):
        assert KpiRepository._resolve_table_name("pr_vs_temperature") == "energy"

    def test_air_quality_alias_resolves_to_aqi_impact(self):
        assert KpiRepository._resolve_table_name("air_quality") == "aqi_impact"


# ===========================================================================
# KpiRepository — fetch_gold_kpi
# ===========================================================================

DESCRIBE_ROWS = [
    {"col_name": "energy_date", "data_type": "date", "comment": ""},
    {"col_name": "facility_id", "data_type": "string", "comment": ""},
    {"col_name": "total_energy_kwh", "data_type": "double", "comment": ""},
]

ENERGY_DATA_ROWS = [
    {"energy_date": date(2026, 4, 15), "facility_id": "WRSF1", "total_energy_kwh": Decimal("1234.56")},
    {"energy_date": date(2026, 4, 14), "facility_id": "AVLSF", "total_energy_kwh": Decimal("987.65")},
]


class TestFetchGoldKpi:
    """Tests for KpiRepository.fetch_gold_kpi."""

    def _repo_with_queries(self, describe_rows, data_rows) -> KpiRepository:
        repo = _make_kpi_repo()
        call_count = [0]

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                return describe_rows
            if "MAX" in sql and "latest" in sql.lower():
                return [{"latest": date(2026, 4, 16)}]
            return data_rows

        repo._execute_query = fake_execute  # type: ignore[assignment]
        return repo

    def test_basic_energy_table_returns_rows(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, sources = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        assert metrics["table_name"] == "gold.mart_energy_daily"
        assert len(metrics["rows"]) == 2

    def test_alias_resolves_before_query(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, sources = repo.fetch_gold_kpi("temperature", anchor_date="2026-04-15")
        assert metrics["table_name"] == "gold.mart_energy_daily"

    def test_unknown_table_raises_value_error(self):
        repo = _make_kpi_repo()
        with pytest.raises(ValueError, match="Unknown KPI table"):
            repo.fetch_gold_kpi("does_not_exist")

    def test_decimal_values_serialized_to_float(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        for row in metrics["rows"]:
            assert isinstance(row["total_energy_kwh"], float)

    def test_date_values_serialized_to_iso_string(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        for row in metrics["rows"]:
            assert isinstance(row["energy_date"], str)
            # ISO format check
            date.fromisoformat(row["energy_date"])

    def test_discovered_columns_present(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, _ = repo.fetch_gold_kpi("energy")
        assert "energy_date" in metrics["discovered_columns"]
        assert "facility_id" in metrics["discovered_columns"]

    def test_sources_layer_is_gold(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        _, sources = repo.fetch_gold_kpi("energy")
        assert sources[0]["layer"] == "Gold"
        assert sources[0]["data_source"] == "databricks"

    def test_station_filter_applied_when_provided(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", station_name="WRSF1")
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        assert "WRSF1" in data_sql[0]

    def test_station_filter_skipped_for_all_keyword(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", station_name="all")
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        assert "LOWER(facility_id)" not in data_sql[0]

    def test_limit_capped_at_500(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", limit=9999)
        data_sql = [s for s in captured if "LIMIT" in s]
        assert data_sql
        assert "LIMIT 500" in data_sql[0]

    def test_limit_minimum_is_1(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", limit=0)
        data_sql = [s for s in captured if "LIMIT" in s]
        assert data_sql
        assert "LIMIT 1" in data_sql[0]

    def test_invalid_limit_defaults_to_100(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", limit="not_a_number")  # type: ignore[arg-type]
        data_sql = [s for s in captured if "LIMIT" in s]
        assert data_sql
        assert "LIMIT 100" in data_sql[0]

    def test_no_date_columns_skips_order_by(self):
        no_date_describe = [
            {"col_name": "facility_id", "data_type": "string", "comment": ""},
            {"col_name": "value", "data_type": "double", "comment": ""},
        ]
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return no_date_describe
            return [{"facility_id": "WRSF1", "value": 1.0}]

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy")
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        assert "ORDER BY" not in data_sql[0]

    def test_empty_describe_raises_unavailable(self):
        repo = _make_kpi_repo()

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                return []
            return []

        repo._execute_query = fake_execute  # type: ignore[assignment]
        with pytest.raises(DatabricksDataUnavailableError):
            repo.fetch_gold_kpi("energy")

    def test_describe_exception_raises_unavailable(self):
        repo = _make_kpi_repo()

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                raise RuntimeError("catalog not reachable")
            return []

        repo._execute_query = fake_execute  # type: ignore[assignment]
        with pytest.raises(DatabricksDataUnavailableError):
            repo.fetch_gold_kpi("energy")

    def test_data_query_exception_raises_unavailable(self):
        repo = _make_kpi_repo()
        call_idx = [0]

        def fake_execute(sql: str) -> list[dict]:
            call_idx[0] += 1
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            raise RuntimeError("query timed out")

        repo._execute_query = fake_execute  # type: ignore[assignment]
        with pytest.raises(DatabricksDataUnavailableError):
            repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")

    def test_anchor_date_beyond_latest_is_capped(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 10)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        # anchor_date is in the future relative to the "latest" in DB
        repo.fetch_gold_kpi("energy", anchor_date="2099-12-31")
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        # Should filter for the capped date, not 2099
        assert "2099" not in data_sql[0]
        assert "2026-04-10" in data_sql[0]

    def test_no_anchor_date_omits_date_filter(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy")  # no anchor_date
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        assert "WHERE" not in data_sql[0]

    def test_filters_applied_metadata_populated(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15", station_name="WRSF1")
        fa = metrics["filters_applied"]
        assert fa["date_filter"] is not None
        assert fa["station_filter"] == "WRSF1"

    def test_filters_applied_empty_when_no_filters(self):
        repo = self._repo_with_queries(DESCRIBE_ROWS, ENERGY_DATA_ROWS)
        metrics, _ = repo.fetch_gold_kpi("energy")
        fa = metrics["filters_applied"]
        assert fa["date_filter"] is None

    def test_describe_rows_with_partition_metadata_ignored(self):
        describe_with_partition = [
            {"col_name": "energy_date", "data_type": "date", "comment": ""},
            {"col_name": "# Partition Information", "data_type": "", "comment": ""},
            {"col_name": "# col_name", "data_type": "data_type", "comment": "comment"},
            {"col_name": "energy_date", "data_type": "date", "comment": ""},
        ]
        repo = _make_kpi_repo()

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                return describe_with_partition
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return [{"energy_date": date(2026, 4, 15), "total_energy_kwh": Decimal("500.0")}]

        repo._execute_query = fake_execute  # type: ignore[assignment]
        # Should not crash despite partition metadata rows
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        assert "energy_date" in metrics["discovered_columns"]

    def test_mart_table_map_covers_all_canonical_keys(self):
        expected = {"aqi_impact", "energy", "forecast_accuracy", "system_kpi", "weather_impact"}
        assert set(KpiRepository._MART_TABLE_MAP.keys()) == expected

    def test_mart_table_map_values_contain_gold_prefix(self):
        for val in KpiRepository._MART_TABLE_MAP.values():
            assert val.startswith("gold."), f"Expected gold. prefix in {val!r}"

    def test_qualified_table_uses_catalog_prefix(self):
        repo = _make_kpi_repo(uc_catalog="test_catalog")
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return ENERGY_DATA_ROWS

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        describe_sql = [s for s in captured if "DESCRIBE TABLE" in s]
        assert describe_sql
        assert "test_catalog" in describe_sql[0]

    def test_datetime_values_serialized_to_iso_string(self):
        rows_with_datetime = [
            {
                "energy_date": datetime(2026, 4, 15, 0, 0, 0),
                "facility_id": "WRSF1",
                "total_energy_kwh": 1000.0,
            }
        ]
        repo = _make_kpi_repo()

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return rows_with_datetime

        repo._execute_query = fake_execute  # type: ignore[assignment]
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        assert isinstance(metrics["rows"][0]["energy_date"], str)

    def test_empty_data_rows_returns_empty_rows_list(self):
        repo = _make_kpi_repo()

        def fake_execute(sql: str) -> list[dict]:
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return []

        repo._execute_query = fake_execute  # type: ignore[assignment]
        metrics, _ = repo.fetch_gold_kpi("energy", anchor_date="2026-04-15")
        assert metrics["rows"] == []

    def test_sql_injection_in_station_name_escaped(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return []

        repo._execute_query = fake_execute  # type: ignore[assignment]
        repo.fetch_gold_kpi("energy", station_name="WRSF1'; DROP TABLE users; --")
        data_sql = [s for s in captured if "SELECT *" in s]
        assert data_sql
        # single-quote is escaped to double-single-quote
        assert "''" in data_sql[0] or "DROP TABLE" not in data_sql[0]

    def test_sql_injection_in_anchor_date_escaped(self):
        repo = _make_kpi_repo()
        captured: list[str] = []

        def fake_execute(sql: str) -> list[dict]:
            captured.append(sql)
            if "DESCRIBE TABLE" in sql:
                return DESCRIBE_ROWS
            if "MAX(" in sql:
                return [{"latest": date(2026, 4, 16)}]
            return []

        repo._execute_query = fake_execute  # type: ignore[assignment]
        # The cap logic will try to parse this as a date — it will fail and
        # fall through to the raw escaped value.
        try:
            repo.fetch_gold_kpi("energy", anchor_date="2026-04-15' OR '1'='1")
        except Exception:
            pass  # We only care that no unescaped injection passes through

        data_sqls = [s for s in captured if "SELECT *" in s]
        if data_sqls:
            assert "OR '1'='1" not in data_sqls[0] or "''" in data_sqls[0]
