"""Base repository: Databricks SQL connection and shared scalar helpers.

All Silver/Gold repositories inherit from this base.
"""
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from statistics import mean
from typing import Any
from urllib.parse import urlparse

from app.core.settings import SolarChatSettings

logger = logging.getLogger(__name__)


class BaseRepository:
    """Shared infrastructure: connection management and scalar helpers."""

    _DEFAULT_UC_CATALOG: str = "pv"
    _DEFAULT_UC_SCHEMA: str = "silver"

    # C1: Allowlists for SQL identifiers used in dynamic query building.
    _ALLOWED_TABLES: frozenset[str] = frozenset({
        "silver.facility_status",
        "silver.energy_readings",
        "silver.weather",
        "silver.air_quality",
        "gold.dim_date",
        "gold.dim_time",
        "gold.dim_aqi_category",
        "gold.dim_weather_condition",
        "gold.fact_energy",
        "gold.dim_facility",
        "gold.forecast_hourly",
        "gold.forecast_daily",
        "gold.model_monitoring_daily",
    })
    _ALLOWED_COLUMNS: frozenset[str] = frozenset({
        "aqi_value", "aqi_category",
        "energy_mwh", "energy_kwh", "completeness_pct",
        "temperature_2m", "temperature_c",
        "wind_speed_10m", "wind_speed_ms",
        "wind_gusts_10m", "wind_gust_ms",
        "shortwave_radiation", "cloud_cover", "cloud_cover_pct",
        "quality_issues", "quality_flag",
        "facility_name", "facility_code", "facility_id", "location_id",
        "location_lat", "location_lng",
        "total_capacity_mw", "total_capacity_registered_mw",
        "total_capacity_maximum_mw",
    })

    def __init__(self, settings: SolarChatSettings) -> None:
        self._settings = settings
        self._catalog = self._resolve_catalog(settings.uc_catalog)
        self._silver_schema = self._resolve_schema(settings.uc_silver_schema)
        self._gold_schema = settings.uc_gold_schema.strip().lower() or "gold"

    @classmethod
    def _resolve_schema(cls, uc_schema: str) -> str:
        normalized_uc_schema = uc_schema.strip().lower()
        if normalized_uc_schema:
            return normalized_uc_schema
        return cls._DEFAULT_UC_SCHEMA

    @classmethod
    def _resolve_catalog(cls, uc_catalog: str) -> str:
        normalized_uc_catalog = uc_catalog.strip().lower()
        if normalized_uc_catalog:
            return normalized_uc_catalog
        return cls._DEFAULT_UC_CATALOG

    @classmethod
    def _validate_sql_identifier(cls, value: str, allowed: frozenset[str]) -> str:
        if value not in allowed:
            raise ValueError(f"SQL identifier not allowed: '{value}'")
        return value

    @contextmanager
    def _databricks_connection(self):
        """Context manager ensuring Databricks SQL connections are always closed."""
        from databricks import sql as databricks_sql

        host = (self._settings.databricks_host or "").strip()
        token = (self._settings.databricks_token or "").strip()
        http_path = (self._settings.resolved_databricks_http_path or "").strip()

        if not host or not token or not http_path:
            raise ValueError(
                "Missing Databricks connection settings. "
                "Required: DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_SQL_HTTP_PATH "
                "(or DATABRICKS_WAREHOUSE_ID)."
            )

        parsed = urlparse(host)
        server_hostname = parsed.netloc if parsed.scheme else host
        if not server_hostname:
            raise ValueError("Invalid DATABRICKS_HOST value.")

        conn = databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
            catalog=self._catalog,
            schema=self._silver_schema,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _execute_query(self, sql: str) -> list[dict[str, Any]]:
        with self._databricks_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _resolve_latest_date(self, table: str) -> date:
        timestamp_column = {
            "silver.weather": "weather_timestamp",
            "silver.air_quality": "aqi_timestamp",
        }.get(table, "date_hour")
        try:
            rows = self._execute_query(
                f"SELECT MAX(CAST({timestamp_column} AS DATE)) AS latest FROM {table}"
            )
            if rows and rows[0]["latest"]:
                val = rows[0]["latest"]
                return date.fromisoformat(str(val)) if isinstance(val, str) else val
        except Exception as exc:
            logger.warning("Cannot resolve latest date from %s (%s). Using current date.", table, exc)
        return date.today()

    def _with_databricks_query(
        self,
        topic_label: str,
        query_fn,
        fallback_or_sources,
        sources_template: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Run Databricks query path and tag sources consistently.

        Supports both signatures:
        - (topic_label, query_fn, sources_template)
        - (topic_label, query_fn, fallback_fn, sources_template)
        """
        if sources_template is None:
            resolved_sources_template = fallback_or_sources
        else:
            resolved_sources_template = sources_template

        data_source = "databricks"
        try:
            metrics = query_fn()
        except Exception as exc:
            logger.error("Databricks query failed for %s (%s).", topic_label, exc)
            raise
        sources = [{**s, "data_source": data_source} for s in resolved_sources_template]
        return metrics, sources

    @staticmethod
    def _resolve_facility(row: dict[str, Any]) -> str:
        return (
            row.get("facility_name")
            or row.get("facility_code")
            or row.get("facility_id")
            or row.get("location_id")
            or "Unknown"
        )

    @staticmethod
    def _to_float(value: str | None, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            return None

    @staticmethod
    def _extract_issues(raw_issues: Any) -> list[str]:
        if not raw_issues:
            return []
        return [issue for issue in str(raw_issues).split("|") if issue]

    @staticmethod
    def _calculate_r_squared(series_x: list[float], series_y: list[float]) -> float:
        if len(series_x) != len(series_y) or len(series_x) < 2:
            return 0.0
        mean_x = mean(series_x)
        mean_y = mean(series_y)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(series_x, series_y))
        den_x = sum((x - mean_x) ** 2 for x in series_x)
        den_y = sum((y - mean_y) ** 2 for y in series_y)
        if den_x == 0.0 or den_y == 0.0:
            return 0.0
        corr = num / ((den_x ** 0.5) * (den_y ** 0.5))
        return max(0.0, min(1.0, corr * corr))

    @staticmethod
    def _format_observed_at(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.replace(tzinfo=None).isoformat(timespec="minutes")
        normalized = str(value).replace("Z", "")
        try:
            return datetime.fromisoformat(normalized).replace(tzinfo=None).isoformat(timespec="minutes")
        except ValueError:
            return str(value)

    @staticmethod
    def _resolve_period_window(
        timeframe: str,
        resolved_anchor_date: date,
        specific_hour: int | None = None,
    ) -> tuple[datetime, datetime, str]:
        day_start = datetime.combine(resolved_anchor_date, datetime.min.time())
        if timeframe == "hour":
            if specific_hour is not None:
                hs = day_start + timedelta(hours=specific_hour)
                return hs, hs + timedelta(hours=1), hs.isoformat(timespec="minutes")
            return day_start, day_start + timedelta(days=1), resolved_anchor_date.isoformat()
        if timeframe == "24h":
            end = day_start + timedelta(hours=24)
            return day_start, end, (
                f"{day_start.isoformat(timespec='minutes')} -> {end.isoformat(timespec='minutes')}"
            )
        if timeframe == "week":
            ws = resolved_anchor_date - timedelta(days=resolved_anchor_date.weekday())
            wsd = datetime.combine(ws, datetime.min.time())
            wed = wsd + timedelta(days=7)
            return wsd, wed, f"{ws.isoformat()} -> {(wed - timedelta(days=1)).date().isoformat()}"
        if timeframe == "month":
            ms = resolved_anchor_date.replace(day=1)
            me = ms.replace(year=ms.year + 1, month=1) if ms.month == 12 else ms.replace(month=ms.month + 1)
            return datetime.combine(ms, datetime.min.time()), datetime.combine(me, datetime.min.time()), ms.strftime("%Y-%m")
        if timeframe == "year":
            ys = resolved_anchor_date.replace(month=1, day=1)
            ye = ys.replace(year=ys.year + 1)
            return datetime.combine(ys, datetime.min.time()), datetime.combine(ye, datetime.min.time()), str(ys.year)
        return day_start, day_start + timedelta(days=1), resolved_anchor_date.isoformat()

    def _deduplicate_stations(self, rows: list[dict[str, Any]], highest: bool) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            facility = row.get("facility") or "Unknown"
            if facility not in seen:
                seen[facility] = row
        result = list(seen.values())
        result.sort(key=lambda r: float(r.get("metric_value", 0)), reverse=highest)
        return result

