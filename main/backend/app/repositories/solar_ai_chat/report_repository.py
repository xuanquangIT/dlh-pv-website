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

_WEATHER_METRIC_COLUMN_MAP: dict[str, str] = {
    "shortwave_radiation": "shortwave_radiation",
    "temperature_2m": "temperature_c",
    "wind_speed_10m": "wind_speed_ms",
    "cloud_cover": "cloud_cover_pct",
}

_TABLE_TIMESTAMP_COLUMN: dict[str, str] = {
    "gold.mart_energy_daily": "energy_date",
    "silver.energy_readings": "date_hour",
    "silver.weather": "weather_timestamp",
    "silver.air_quality": "aqi_timestamp",
}

# Silver tables expose a pre-computed station-local DATE column so callers
# can filter by "ngày X" without converting timezones at query time. Prefer
# these over CAST(<utc_ts> AS DATE) for any user-facing day grouping.
_TABLE_LOCAL_DATE_COLUMN: dict[str, str] = {
    "silver.energy_readings": "reading_date_local",
    "silver.weather": "weather_date_local",
    "silver.air_quality": "aqi_date_local",
}


class ReportRepository(BaseRepository):
    """Handles fetch_station_daily_report using Databricks SQL only."""

    def fetch_station_daily_report(
        self,
        anchor_date: date,
        metrics: list[str] | None = None,
        station_name: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        requested = set(metrics) & _ALL_REPORT_METRICS if metrics else _ALL_REPORT_METRICS
        if not requested:
            requested = _ALL_REPORT_METRICS

        data_source = "databricks"
        stations = self._station_daily_report_trino(anchor_date, requested, station_name=station_name)
        available_date_min, available_date_max = self._station_report_date_range_trino(requested)

        has_data = bool(stations)
        no_data_reason: str | None = None
        if not has_data:
            if station_name:
                no_data_reason = (
                    f"không có dữ liệu cho trạm '{station_name}' ngày {anchor_date.isoformat()}"
                )
            else:
                no_data_reason = f"không có dữ liệu cho ngày {anchor_date.isoformat()}"
            logger.info(
                "station_daily_report_no_data date=%s station_name=%s available_date_min=%s available_date_max=%s source=%s",
                anchor_date.isoformat(),
                station_name,
                available_date_min,
                available_date_max,
                data_source,
            )

        sources: list[dict[str, str]] = []
        if requested & {"energy_mwh"}:
            sources.append({"layer": "Gold", "dataset": "gold.mart_energy_daily", "data_source": data_source})
        if requested & {"shortwave_radiation", "temperature_2m"}:
            sources.append({"layer": "Silver", "dataset": "silver.weather", "data_source": data_source})
        if requested & {"aqi_value"}:
            sources.append({"layer": "Silver", "dataset": "silver.air_quality", "data_source": data_source})

        result: dict[str, Any] = {
            "report_date": anchor_date.isoformat(),
            "metrics_requested": sorted(requested),
            "stations": stations,
            "station_count": len(stations),
            "has_data": has_data,
            "no_data_reason": no_data_reason,
            "available_date_min": available_date_min,
            "available_date_max": available_date_max,
        }
        if station_name:
            result["station_filter"] = station_name

        return result, sources

    @staticmethod
    def _normalize_station_filter(raw: str | None) -> str | None:
        """Strip common suffixes so 'Avonlie Solar Farm' still matches 'Avonlie'.

        The canonical facility_name values in dim_facility are short
        (e.g. 'Avonlie', 'Darlington Point'); LLMs often append marketing
        words like 'Solar Farm', 'Station', 'Plant'. We reduce the filter to
        the non-boilerplate remainder, but only if that remainder is
        meaningful (>=3 chars). Also accepts short facility codes unchanged.
        """
        if not raw:
            return None
        value = raw.strip()
        if not value:
            return None
        # Short codes like AVLSF / DARLSF are kept unchanged
        if len(value) <= 8 and value.replace("_", "").replace("-", "").isalnum():
            return value
        import re as _re
        stripped = _re.sub(
            r"\b(solar\s+farm|solar|farm|station|plant|park|power\s+station)\b",
            "", value, flags=_re.IGNORECASE,
        )
        stripped = _re.sub(r"\s+", " ", stripped).strip(" ,-_")
        return stripped if len(stripped) >= 3 else value

    def fetch_station_hourly_report(
        self,
        station_name: str | None,
        anchor_date: date | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Return per-hour energy breakdown for a station on a given date.

        Queries gold.fact_energy (which already stores hourly rows keyed by
        date_hour).  When anchor_date is None the latest date available for
        the station is used.  Missing station -> aggregate across all
        facilities for that date.
        """
        facility_filter_sql = ""
        normalized_station = self._normalize_station_filter(station_name)
        if normalized_station:
            safe = normalized_station.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
            facility_filter_sql = (
                " AND ("
                f"LOWER(COALESCE(d.facility_name, f.facility_id)) LIKE LOWER('%{safe}%')"
                f" OR LOWER(f.facility_id) LIKE LOWER('%{safe}%')"
                ")"
            )

        # Lakehouse stores UTC (date_hour) + station-local (date_hour_local,
        # reading_date_local) keys.  Hourly / daily reports MUST use the
        # local-time columns so "ngày X / giờ Y" matches what operators see
        # on station wall-clocks (e.g. Australian AEST/AEDT). UTC grouping
        # shifts peak hours 10 hours to the "wrong" side of midnight.
        if anchor_date is None:
            row = self._execute_query(
                "SELECT MAX(f.reading_date_local) AS max_date"
                " FROM gold.fact_energy f"
                " LEFT JOIN gold.dim_facility d"
                "   ON f.facility_id = d.facility_id AND d.is_current = true"
                f" WHERE f.energy_mwh IS NOT NULL{facility_filter_sql}"
            )
            max_date_raw = row[0].get("max_date") if row else None
            if isinstance(max_date_raw, date):
                anchor_date = max_date_raw
            elif isinstance(max_date_raw, str):
                try:
                    anchor_date = date.fromisoformat(max_date_raw[:10])
                except ValueError:
                    anchor_date = date.today()
            else:
                anchor_date = date.today()

        date_str = anchor_date.isoformat()
        rows = self._execute_query(
            "SELECT EXTRACT(HOUR FROM f.date_hour_local) AS hr,"
            "       COALESCE(d.facility_name, f.facility_id) AS facility,"
            "       SUM(f.energy_mwh) AS energy_mwh,"
            "       AVG(f.capacity_factor_pct) AS capacity_factor_pct"
            " FROM gold.fact_energy f"
            " LEFT JOIN gold.dim_facility d"
            "   ON f.facility_id = d.facility_id AND d.is_current = true"
            f" WHERE f.reading_date_local = DATE '{date_str}'"
            f"{facility_filter_sql}"
            " GROUP BY EXTRACT(HOUR FROM f.date_hour_local), COALESCE(d.facility_name, f.facility_id)"
            " ORDER BY facility, hr"
        )

        hourly: list[dict[str, Any]] = []
        for r in rows:
            hourly.append({
                "hour": int(r["hr"]) if r.get("hr") is not None else None,
                "facility": r.get("facility"),
                "energy_mwh": round(float(r["energy_mwh"]), 4) if r.get("energy_mwh") is not None else 0.0,
                "capacity_factor_pct": (
                    round(float(r["capacity_factor_pct"]), 2)
                    if r.get("capacity_factor_pct") is not None else None
                ),
            })

        total_mwh = round(sum(h["energy_mwh"] for h in hourly), 4)
        result: dict[str, Any] = {
            "report_date": date_str,
            "station_filter": station_name or "",
            "hourly_rows": hourly,
            "row_count": len(hourly),
            "total_energy_mwh": total_mwh,
            "has_data": bool(hourly),
        }
        sources = [
            {"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"},
            {"layer": "Gold", "dataset": "gold.dim_facility", "data_source": "databricks"},
        ]
        return result, sources

    # Legacy aliases retained for compatibility with existing tests/call sites.
    def _station_report_date_range_trino(self, requested: set[str]) -> tuple[str | None, str | None]:
        return self._station_report_date_range_databricks(requested)

    def _station_daily_report_trino(
        self, anchor_date: date, requested: set[str], station_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._station_daily_report_databricks(anchor_date, requested, station_name=station_name)

    @staticmethod
    def _normalize_date_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _merge_date_ranges(
        current_min: str | None,
        current_max: str | None,
        next_min: str | None,
        next_max: str | None,
    ) -> tuple[str | None, str | None]:
        merged_min = current_min
        if next_min is not None and (merged_min is None or next_min < merged_min):
            merged_min = next_min

        merged_max = current_max
        if next_max is not None and (merged_max is None or next_max > merged_max):
            merged_max = next_max

        return merged_min, merged_max

    @staticmethod
    def _report_sources_for_metrics(requested: set[str]) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []
        if requested & {"energy_mwh"}:
            sources.append(("gold.mart_energy_daily", "lh_silver_clean_hourly_energy.csv"))
        if requested & {"shortwave_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"}:
            sources.append(("silver.weather", "lh_silver_clean_hourly_weather.csv"))
        if requested & {"aqi_value"}:
            sources.append(("silver.air_quality", "lh_silver_clean_hourly_air_quality.csv"))
        return sources

    def _station_report_date_range_databricks(self, requested: set[str]) -> tuple[str | None, str | None]:
        min_date: str | None = None
        max_date: str | None = None

        for table_name, _ in self._report_sources_for_metrics(requested):
            # Prefer station-local DATE column so the range matches
            # user-facing dates (wall-clock), not UTC.
            local_date_column = _TABLE_LOCAL_DATE_COLUMN.get(table_name)
            ts_column = _TABLE_TIMESTAMP_COLUMN.get(table_name, "date_hour")
            if local_date_column:
                date_expr = local_date_column
                filter_col = local_date_column
            else:
                date_expr = f"CAST({ts_column} AS DATE)"
                filter_col = ts_column
            rows = self._execute_query(
                "SELECT"
                f" MIN({date_expr}) AS min_date,"
                f" MAX({date_expr}) AS max_date"
                f" FROM {table_name}"
                f" WHERE {filter_col} IS NOT NULL"
            )
            if not rows:
                continue

            row_min = self._normalize_date_value(rows[0].get("min_date"))
            row_max = self._normalize_date_value(rows[0].get("max_date"))
            min_date, max_date = self._merge_date_ranges(min_date, max_date, row_min, row_max)

        return min_date, max_date

    def _station_report_date_range_csv(self, requested: set[str]) -> tuple[str | None, str | None]:
        min_date: str | None = None
        max_date: str | None = None

        for _, csv_name in self._report_sources_for_metrics(requested):
            rows = self._load_csv(self._dataset_path(csv_name))
            for row in rows:
                dt = self._parse_datetime(row.get("date_hour"))
                if dt is None:
                    continue
                date_str = dt.date().isoformat()
                min_date, max_date = self._merge_date_ranges(min_date, max_date, date_str, date_str)

        return min_date, max_date

    @staticmethod
    def _build_station_filter_clause(
        station_name: str | None,
        facility_expr: str,
    ) -> str:
        """Build a SQL WHERE clause fragment for station name filtering.

        Uses LOWER + LIKE for case-insensitive partial matching.
        Returns an empty string if no station filter is requested.
        """
        normalized = ReportRepository._normalize_station_filter(station_name)
        if not normalized:
            return ""
        # Escape SQL wildcards in user input to prevent injection
        safe_name = normalized.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
        return f" AND LOWER({facility_expr}) LIKE LOWER('%{safe_name}%')"

    def _station_daily_report_databricks(
        self, anchor_date: date, requested: set[str], station_name: str | None = None,
    ) -> list[dict[str, Any]]:
        date_str = anchor_date.isoformat()
        facility_data: dict[str, dict[str, Any]] = {}

        station_filter = self._build_station_filter_clause(
            station_name, "facility_name",
        )
        # Fetch directly from Gold Mart which holds all pre-aggregated KPIs
        rows = self._execute_query(
            f"SELECT facility_name AS facility,"
            f"       energy_mwh_daily AS energy_mwh,"
            f"       avg_shortwave_radiation AS shortwave_radiation,"
            f"       avg_temperature_c AS temperature_2m,"
            f"       avg_aqi_value AS aqi_value,"
            f"       weighted_capacity_factor_pct,"
            f"       dominant_weather_condition"
            f" FROM gold.mart_energy_daily"
            f" WHERE CAST(energy_date AS DATE) = DATE '{date_str}'"
            f"{station_filter}"
        )
        
        for r in rows:
            name = r["facility"]
            facility_data.setdefault(name, {"facility": name})
            if "energy_mwh" in requested and r.get("energy_mwh") is not None:
                facility_data[name]["energy_mwh"] = round(float(r["energy_mwh"]), 4)
            if "shortwave_radiation" in requested and r.get("shortwave_radiation") is not None:
                facility_data[name]["shortwave_radiation"] = round(float(r["shortwave_radiation"]), 2)
            if "temperature_2m" in requested and r.get("temperature_2m") is not None:
                facility_data[name]["temperature_2m"] = round(float(r["temperature_2m"]), 2)
            if "aqi_value" in requested and r.get("aqi_value") is not None:
                facility_data[name]["aqi_value"] = round(float(r["aqi_value"]), 2)

        if not facility_data:
            if "energy_mwh" in requested:
                station_filter = self._build_station_filter_clause(
                    station_name, "COALESCE(facility_name, facility_id)",
                )
                rows = self._execute_query(
                    f"SELECT COALESCE(facility_name, facility_id) AS facility,"
                    f"       SUM(energy_kwh) AS total_energy_kwh"
                    f" FROM silver.energy_readings"
                    f" WHERE reading_date_local = DATE '{date_str}'"
                    f"{station_filter}"
                    f" GROUP BY COALESCE(facility_name, facility_id)"
                )
                for r in rows:
                    name = r["facility"]
                    facility_data.setdefault(name, {"facility": name})
                    facility_data[name]["energy_mwh"] = round(float(r["total_energy_kwh"]) / 1000.0, 4)

            weather_cols = requested & {"shortwave_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"}
            if weather_cols:
                agg_parts = ", ".join(
                    f"AVG({_WEATHER_METRIC_COLUMN_MAP[col]}) AS avg_{col}"
                    for col in sorted(weather_cols)
                )
                station_filter = self._build_station_filter_clause(
                    station_name, "COALESCE(facility_name, location_id)",
                )
                rows = self._execute_query(
                    f"SELECT COALESCE(facility_name, location_id) AS facility, {agg_parts}"
                    f" FROM silver.weather"
                    f" WHERE weather_date_local = DATE '{date_str}'"
                    f"{station_filter}"
                    f" GROUP BY COALESCE(facility_name, location_id)"
                )
                for r in rows:
                    name = r["facility"]
                    facility_data.setdefault(name, {"facility": name})
                    for col in weather_cols:
                        val = r.get(f"avg_{col}")
                        if val is not None:
                            facility_data[name][col] = round(float(val), 2)

            if "aqi_value" in requested:
                station_filter = self._build_station_filter_clause(
                    station_name, "COALESCE(facility_name, location_id)",
                )
                rows = self._execute_query(
                    f"SELECT COALESCE(facility_name, location_id) AS facility,"
                    f"       AVG(aqi_value) AS avg_aqi"
                    f" FROM silver.air_quality"
                    f" WHERE aqi_date_local = DATE '{date_str}'"
                    f"{station_filter}"
                    f" GROUP BY COALESCE(facility_name, location_id)"
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



