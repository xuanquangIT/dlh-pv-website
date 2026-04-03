"""Extreme metric repository: highest/lowest AQI, energy, and weather queries."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from app.repositories.solar_ai_chat.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ExtremeRepository(BaseRepository):
    """Handles fetch_extreme_* queries with Trino-first, CSV-fallback."""

    def fetch_extreme_aqi(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="lh_silver_clean_hourly_air_quality",
            csv_filename="lh_silver_clean_hourly_air_quality.csv",
            value_column="aqi_value",
            metric_label="AQI",
            query_type=query_type,
            timeframe=timeframe,
            anchor_date=anchor_date,
            specific_hour=specific_hour,
            extra_columns=("aqi_category",),
            csv_extra_keys=("aqi_category",),
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
        return metrics, [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality", "data_source": data_source}]

    def fetch_extreme_energy(
        self,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="lh_silver_clean_hourly_energy",
            csv_filename="lh_silver_clean_hourly_energy.csv",
            value_column="energy_mwh",
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
            f"{qt}_energy_mwh": round(float(selected["metric_value"]), 2),
            f"top_{qt}_stations": [
                {"facility": s["facility"], "energy_mwh": round(float(s["metric_value"]), 2)}
                for s in station_rows[:5]
            ],
        }
        return metrics, [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy", "data_source": data_source}]

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
        base, qt, station_rows, data_source = self._fetch_extreme_base(
            table="lh_silver_clean_hourly_weather",
            csv_filename="lh_silver_clean_hourly_weather.csv",
            value_column=weather_metric,
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
            "weather_metric": weather_metric,
            "weather_metric_label": weather_metric_label,
            "weather_unit": weather_unit,
            f"{qt}_station": selected["facility"],
            f"{qt}_weather_value": round(float(selected["metric_value"]), 2),
            f"top_{qt}_stations": [
                {"facility": s["facility"], "weather_value": round(float(s["metric_value"]), 2)}
                for s in station_rows[:5]
            ],
        }
        return metrics, [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather", "data_source": data_source}]

    def _fetch_extreme_base(
        self,
        table: str,
        csv_filename: str,
        value_column: str,
        metric_label: str,
        query_type: str,
        timeframe: str,
        anchor_date: date | None,
        specific_hour: int | None = None,
        extra_columns: tuple[str, ...] = (),
        csv_extra_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], str]:
        """Shared setup for all fetch_extreme_* methods.

        Returns (base_metrics, qt, deduplicated_station_rows, data_source).
        Raises ValueError when no rows are found.
        """
        highest = query_type == "highest"
        qt = "highest" if highest else "lowest"
        resolved_date = anchor_date or self._resolve_latest_date(table, csv_filename)
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
            csv_filename=csv_filename,
            csv_value_key=value_column,
            highest=highest,
            csv_extra_keys=csv_extra_keys,
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
        csv_filename: str,
        csv_value_key: str,
        highest: bool,
        extra_columns: tuple[str, ...] = (),
        csv_extra_keys: tuple[str, ...] = (),
    ) -> tuple[list[dict[str, Any]], str]:
        self._validate_sql_identifier(table, self._ALLOWED_TABLES)
        self._validate_sql_identifier(value_column, self._ALLOWED_COLUMNS)
        for col in extra_columns:
            self._validate_sql_identifier(col, self._ALLOWED_COLUMNS)
        extra_select = "".join(f", {col}" for col in extra_columns)
        sql = (
            f"SELECT COALESCE(facility_name, facility_code) AS facility,"
            f"       {value_column} AS metric_value,"
            f"       date_hour AS observed_at"
            f"       {extra_select}"
            f" FROM {table}"
            f" WHERE date_hour >= TIMESTAMP '{window_start.strftime('%Y-%m-%d %H:%M:%S')}'"
            f"   AND date_hour <  TIMESTAMP '{window_end.strftime('%Y-%m-%d %H:%M:%S')}'"
            f"   AND {value_column} IS NOT NULL"
            f" ORDER BY {value_column} {order}"
        )
        try:
            return self._execute_query(sql), "trino"
        except Exception as exc:
            logger.warning("Trino unavailable for %s (%s), falling back to CSV.", table, exc)
            return self._csv_extreme_fallback(
                csv_filename, csv_value_key, highest, window_start, window_end, csv_extra_keys,
            ), "csv"
