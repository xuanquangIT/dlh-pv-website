"""Station daily report repository: per-facility aggregation for a given date."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from statistics import mean
from typing import Any

from app.repositories.solar_ai_chat.base_repository import BaseRepository

logger = logging.getLogger(__name__)

_ALL_REPORT_METRICS: frozenset[str] = frozenset({
    "energy_mwh", "shortwave_radiation", "aqi_value",
    "temperature_2m", "wind_speed_10m", "cloud_cover",
})


class ReportRepository(BaseRepository):
    """Handles fetch_station_daily_report with Trino-first, CSV-fallback."""

    def fetch_station_daily_report(
        self,
        anchor_date: date,
        metrics: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        requested = set(metrics) & _ALL_REPORT_METRICS if metrics else _ALL_REPORT_METRICS
        if not requested:
            requested = _ALL_REPORT_METRICS

        data_source = "trino"
        try:
            stations = self._station_daily_report_trino(anchor_date, requested)
        except Exception as exc:
            logger.warning("Trino unavailable for station_daily_report (%s), falling back to CSV.", exc)
            stations = self._station_daily_report_csv(anchor_date, requested)
            data_source = "csv"

        sources: list[dict[str, str]] = []
        if requested & {"energy_mwh"}:
            sources.append({"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy", "data_source": data_source})
        if requested & {"shortwave_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"}:
            sources.append({"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather", "data_source": data_source})
        if requested & {"aqi_value"}:
            sources.append({"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality", "data_source": data_source})

        return {
            "report_date": anchor_date.isoformat(),
            "metrics_requested": sorted(requested),
            "stations": stations,
            "station_count": len(stations),
        }, sources

    def _station_daily_report_trino(self, anchor_date: date, requested: set[str]) -> list[dict[str, Any]]:
        date_str = anchor_date.isoformat()
        facility_data: dict[str, dict[str, Any]] = {}

        if "energy_mwh" in requested:
            rows = self._execute_query(
                f"SELECT COALESCE(facility_name, facility_code) AS facility,"
                f"       SUM(energy_mwh) AS total_energy_mwh"
                f" FROM lh_silver_clean_hourly_energy"
                f" WHERE CAST(date_hour AS DATE) = DATE '{date_str}'"
                f" GROUP BY COALESCE(facility_name, facility_code)"
            )
            for r in rows:
                name = r["facility"]
                facility_data.setdefault(name, {"facility": name})
                facility_data[name]["energy_mwh"] = round(float(r["total_energy_mwh"]), 4)

        weather_cols = requested & {"shortwave_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"}
        if weather_cols:
            agg_parts = ", ".join(f"AVG({col}) AS avg_{col}" for col in sorted(weather_cols))
            rows = self._execute_query(
                f"SELECT COALESCE(facility_name, facility_code) AS facility, {agg_parts}"
                f" FROM lh_silver_clean_hourly_weather"
                f" WHERE CAST(date_hour AS DATE) = DATE '{date_str}'"
                f" GROUP BY COALESCE(facility_name, facility_code)"
            )
            for r in rows:
                name = r["facility"]
                facility_data.setdefault(name, {"facility": name})
                for col in weather_cols:
                    val = r.get(f"avg_{col}")
                    if val is not None:
                        facility_data[name][col] = round(float(val), 2)

        if "aqi_value" in requested:
            rows = self._execute_query(
                f"SELECT COALESCE(facility_name, facility_code) AS facility,"
                f"       AVG(aqi_value) AS avg_aqi"
                f" FROM lh_silver_clean_hourly_air_quality"
                f" WHERE CAST(date_hour AS DATE) = DATE '{date_str}'"
                f" GROUP BY COALESCE(facility_name, facility_code)"
            )
            for r in rows:
                name = r["facility"]
                facility_data.setdefault(name, {"facility": name})
                facility_data[name]["aqi_value"] = round(float(r["avg_aqi"]), 2)

        return sorted(facility_data.values(), key=lambda x: x["facility"])

    def _station_daily_report_csv(self, anchor_date: date, requested: set[str]) -> list[dict[str, Any]]:
        date_str = anchor_date.isoformat()
        facility_data: dict[str, dict[str, Any]] = {}

        if "energy_mwh" in requested:
            rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
            energy_by_facility: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                dt = self._parse_datetime(r.get("date_hour"))
                if dt and dt.date().isoformat() == date_str:
                    name = self._resolve_facility(r)
                    val = self._to_float(r.get("energy_mwh"), default=float("nan"))
                    if val == val:
                        energy_by_facility[name].append(val)
            for name, vals in energy_by_facility.items():
                facility_data.setdefault(name, {"facility": name})
                facility_data[name]["energy_mwh"] = round(sum(vals), 4)

        weather_cols = requested & {"shortwave_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"}
        if weather_cols:
            rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
            weather_agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                dt = self._parse_datetime(r.get("date_hour"))
                if dt and dt.date().isoformat() == date_str:
                    name = self._resolve_facility(r)
                    for col in weather_cols:
                        val = self._to_float(r.get(col), default=float("nan"))
                        if val == val:
                            weather_agg[name][col].append(val)
            for name, cols in weather_agg.items():
                facility_data.setdefault(name, {"facility": name})
                for col, vals in cols.items():
                    facility_data[name][col] = round(mean(vals), 2) if vals else 0.0

        if "aqi_value" in requested:
            rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))
            aqi_agg: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                dt = self._parse_datetime(r.get("date_hour"))
                if dt and dt.date().isoformat() == date_str:
                    name = self._resolve_facility(r)
                    val = self._to_float(r.get("aqi_value"))
                    if val is not None:
                        aqi_agg[name].append(val)
            for name, vals in aqi_agg.items():
                facility_data.setdefault(name, {"facility": name})
                facility_data[name]["aqi_value"] = round(mean(vals), 2) if vals else 0.0

        return sorted(facility_data.values(), key=lambda x: x["facility"])
