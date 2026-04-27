"""Extreme metric repository: highest/lowest AQI, energy, and weather queries."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from app.repositories.solar_ai_chat.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ExtremeRepository(BaseRepository):
    """Handles fetch_extreme_* queries using Databricks SQL only."""

    _WEATHER_COLUMN_MAP: dict[str, str] = {
        "temperature_2m": "temperature_c",
        "wind_speed_10m": "wind_speed_ms",
        "wind_gusts_10m": "wind_gust_ms",
        "cloud_cover": "cloud_cover_pct",
    }

    def fetch_extreme_aqi(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="silver.air_quality",
            value_column="aqi_value",
            metric_label="AQI",
            query_type=query_type,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
            extra_columns=("aqi_category",),
        )
        selected = station_rows[0]
        metrics = {
            **base,
            "extreme_metric": "aqi",
            "aqi_query_type": qt,
            f"{qt}_station": selected["facility"],
            f"{qt}_aqi_value": round(float(selected["metric_value"]), 2),
            f"{qt}_aqi_category": selected.get("aqi_category", "Unknown"),
            f"top_{qt}_stations": [
                {
                    "facility": s["facility"],
                    "aqi_value": round(float(s["metric_value"]), 2),
                    "aqi_category": s.get("aqi_category", "Unknown"),
                }
                for s in station_rows[:5]
            ],
        }
        return metrics, [{"layer": "Silver", "dataset": "silver.air_quality", "data_source": data_source}]

    def fetch_extreme_energy(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="silver.energy_readings",
            value_column="energy_kwh",
            metric_label="Energy",
            query_type=query_type,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
        )
        selected = station_rows[0]
        metrics = {
            **base,
            "extreme_metric": "energy",
            f"{qt}_station": selected["facility"],
            f"{qt}_energy_mwh": round(float(selected["metric_value"]) / 1000.0, 2),
            f"top_{qt}_stations": [
                {"facility": s["facility"], "energy_mwh": round(float(s["metric_value"]) / 1000.0, 2)}
                for s in station_rows[:5]
            ],
        }
        return metrics, [{"layer": "Silver", "dataset": "silver.energy_readings", "data_source": data_source}]

    def fetch_extreme_weather(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        weather_metric: str,
        weather_metric_label: str,
        weather_unit: str,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        resolved_weather_metric = self._WEATHER_COLUMN_MAP.get(weather_metric, weather_metric)
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="silver.weather",
            value_column=resolved_weather_metric,
            metric_label="Weather",
            query_type=query_type,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
        )
        selected = station_rows[0]
        metrics = {
            **base,
            "extreme_metric": "weather",
            "weather_metric": resolved_weather_metric,
            "weather_metric_label": weather_metric_label,
            "weather_unit": weather_unit,
            f"{qt}_station": selected["facility"],
            f"{qt}_weather_value": round(float(selected["metric_value"]), 2),
            f"top_{qt}_stations": [
                {"facility": s["facility"], "weather_value": round(float(s["metric_value"]), 2)}
                for s in station_rows[:5]
            ],
        }
        return metrics, [{"layer": "Silver", "dataset": "silver.weather", "data_source": data_source}]

    def _fetch_extreme_base(
        self,
        table: str,
        value_column: str,
        metric_label: str,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
        extra_columns: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], str]:
        """Shared setup for all fetch_extreme_* methods.

        Returns (base_metrics, qt, deduplicated_station_rows, data_source).
        Raises ValueError when no rows are found.
        """
        highest = query_type == "highest"
        qt = "highest" if highest else "lowest"
        resolved_date = anchor_date or self._resolve_latest_date(table)
        window_start, window_end, period_label = self._resolve_period_window(
            timeframe, resolved_date, specific_hour,
        )
        rows, data_source = self._query_extreme_rows(
            table=table,
            value_column=value_column,
            order="DESC" if highest else "ASC",
            window_start=window_start,
            window_end=window_end,
            extra_columns=extra_columns,
        )
        if not rows:
            raise ValueError(
                f"No {metric_label} data for timeframe '{timeframe}' around '{period_label}'."
            )
        station_rows = self._deduplicate_stations(rows, highest)
        base: dict[str, Any] = {
            "query_date": resolved_date.isoformat(),
            "timeframe": timeframe,
            "period_label": period_label,
            "specific_hour": specific_hour,
            "query_type": qt,
            "observed_at": self._format_observed_at(station_rows[0].get("observed_at")),
        }
        return base, qt, station_rows, data_source

    def _query_extreme_rows(
        self,
        table: str,
        value_column: str,
        order: str,
        window_start: datetime,
        window_end: datetime,
        extra_columns: tuple[str, ...] = (),
    ) -> tuple[list[dict[str, Any]], str]:
        self._validate_sql_identifier(table, self._ALLOWED_TABLES)
        value_column = self._WEATHER_COLUMN_MAP.get(value_column, value_column)
        self._validate_sql_identifier(value_column, self._ALLOWED_COLUMNS)
        for col in extra_columns:
            self._validate_sql_identifier(col, self._ALLOWED_COLUMNS)

        # Timeframe windows (hour/day/week/month/year) are built from the
        # user's anchor_date which is a wall-clock date — so filter on the
        # station-local timestamp column, not UTC. Otherwise an Australian
        # station's "ngày 17/04" actually maps to UTC 16/04 07:00..17/04 07:00.
        if table == "silver.energy_readings":
            facility_expr = "COALESCE(facility_name, facility_id)"
            timestamp_column = "date_hour_local"
        elif table == "silver.weather":
            facility_expr = "COALESCE(facility_name, location_id)"
            timestamp_column = "weather_timestamp_local"
        elif table == "silver.air_quality":
            facility_expr = "COALESCE(facility_name, location_id)"
            timestamp_column = "aqi_timestamp_local"
        else:
            facility_expr = "COALESCE(facility_name, facility_code)"
            timestamp_column = "date_hour_local"

        extra_select = "".join(f", {col}" for col in extra_columns)
        sql = (
            f"SELECT {facility_expr} AS facility,"
            f"       {value_column} AS metric_value,"
            f"       {timestamp_column} AS observed_at"
            f"       {extra_select}"
            f" FROM {table}"
            f" WHERE {timestamp_column} >= TIMESTAMP '{window_start.strftime('%Y-%m-%d %H:%M:%S')}'"
            f"   AND {timestamp_column} <  TIMESTAMP '{window_end.strftime('%Y-%m-%d %H:%M:%S')}'"
            f"   AND {value_column} IS NOT NULL"
            f" ORDER BY {value_column} {order}"
        )
        return self._execute_query(sql), "databricks"

