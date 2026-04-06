"""Base repository: Trino connection, CSV loading, and shared scalar helpers.

All Silver/Gold repositories inherit from this base.
"""
import csv
import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from trino.dbapi import connect as trino_connect

from app.core.settings import SolarChatSettings

logger = logging.getLogger(__name__)


class BaseRepository:
    """Shared infrastructure: connection management, CSV loader, scalar helpers."""

    _LEGACY_TRINO_CATALOG: str = "postgresql"
    _LAKEHOUSE_TRINO_CATALOG: str = "iceberg"
    _LEGACY_TRINO_SCHEMA: str = "public"
    _DEFAULT_ICEBERG_SCHEMA: str = "silver"
    _ICEBERG_TABLE_MAP: dict[str, str] = {
        "lh_silver_clean_facility_master": "silver.clean_facility_master",
        "lh_silver_clean_hourly_energy": "silver.clean_hourly_energy",
        "lh_silver_clean_hourly_weather": "silver.clean_hourly_weather",
        "lh_silver_clean_hourly_air_quality": "silver.clean_hourly_air_quality",
        "lh_gold_dim_date": "gold.dim_date",
        "lh_gold_dim_time": "gold.dim_time",
        "lh_gold_dim_aqi_category": "gold.dim_aqi_category",
        "lh_gold_fact_solar_environmental": "gold.fact_solar_environmental",
        "lh_gold_dim_facility": "gold.dim_facility",
    }

    # C1: Allowlists for SQL identifiers used in dynamic query building.
    _ALLOWED_TABLES: frozenset[str] = frozenset({
        "lh_silver_clean_facility_master",
        "lh_silver_clean_hourly_energy",
        "lh_silver_clean_hourly_weather",
        "lh_silver_clean_hourly_air_quality",
        "lh_gold_dim_date",
        "lh_gold_dim_time",
        "lh_gold_dim_aqi_category",
        "lh_gold_fact_solar_environmental",
        "lh_gold_dim_facility",
    })
    _ALLOWED_COLUMNS: frozenset[str] = frozenset({
        "aqi_value", "aqi_category",
        "energy_mwh", "completeness_pct",
        "temperature_2m", "wind_speed_10m", "wind_gusts_10m",
        "shortwave_radiation", "cloud_cover",
        "quality_issues", "quality_flag",
    })

    def __init__(self, settings: SolarChatSettings) -> None:
        self._settings = settings
        self._data_root = settings.resolved_data_root
        self._trino_catalog = self._resolve_trino_catalog(settings.trino_catalog)
        self._trino_schema = self._resolve_trino_schema(self._trino_catalog, settings.trino_schema)

    @classmethod
    def _resolve_trino_schema(cls, catalog: str, schema: str) -> str:
        normalized_catalog = catalog.strip().lower()
        normalized_schema = schema.strip().lower()
        if normalized_catalog == cls._LAKEHOUSE_TRINO_CATALOG and normalized_schema == cls._LEGACY_TRINO_SCHEMA:
            return cls._DEFAULT_ICEBERG_SCHEMA
        return schema

    def _rewrite_sql_for_iceberg(self, sql: str) -> str:
        if self._trino_catalog != self._LAKEHOUSE_TRINO_CATALOG:
            return sql

        rewritten = sql
        for legacy_name, iceberg_name in sorted(self._ICEBERG_TABLE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            qualified_name = f"{self._trino_catalog}.{iceberg_name}"
            rewritten = re.sub(rf"\b{re.escape(legacy_name)}\b", qualified_name, rewritten)
        return rewritten

    @classmethod
    def _resolve_trino_catalog(cls, catalog: str) -> str:
        normalized = catalog.strip().lower()
        if normalized == cls._LEGACY_TRINO_CATALOG:
            return cls._LAKEHOUSE_TRINO_CATALOG
        return catalog

    @classmethod
    def _validate_sql_identifier(cls, value: str, allowed: frozenset[str]) -> str:
        if value not in allowed:
            raise ValueError(f"SQL identifier not allowed: '{value}'")
        return value

    @contextmanager
    def _trino_connection(self):
        """Context manager ensuring Trino connections are always closed."""
        conn = trino_connect(
            host=self._settings.trino_host,
            port=self._settings.trino_port,
            user=self._settings.trino_user,
            catalog=self._trino_catalog,
            schema=self._trino_schema,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _execute_query(self, sql: str) -> list[dict[str, Any]]:
        sql = self._rewrite_sql_for_iceberg(sql)
        with self._trino_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _resolve_latest_date(self, table: str, csv_filename: str) -> date:
        try:
            rows = self._execute_query(
                f"SELECT MAX(CAST(date_hour AS DATE)) AS latest FROM {table}"
            )
            if rows and rows[0]["latest"]:
                val = rows[0]["latest"]
                return date.fromisoformat(str(val)) if isinstance(val, str) else val
        except Exception:
            pass
        csv_rows = self._load_csv(self._dataset_path(csv_filename))
        latest_dt = None
        for row in csv_rows:
            dt = self._parse_datetime(row.get("date_hour") or row.get("timestamp"))
            if dt is not None and (latest_dt is None or dt > latest_dt):
                latest_dt = dt
        return latest_dt.date() if latest_dt else date.today()

    def _with_trino_fallback(
        self,
        topic_label: str,
        trino_fn,
        csv_fn,
        sources_template: list[dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Run trino_fn; on any failure log and run csv_fn instead."""
        data_source = "trino"
        try:
            metrics = trino_fn()
        except Exception as exc:
            logger.warning("Trino unavailable for %s (%s), falling back to CSV.", topic_label, exc)
            metrics = csv_fn()
            data_source = "csv"
        sources = [{**s, "data_source": data_source} for s in sources_template]
        return metrics, sources

    def _dataset_path(self, filename: str) -> Path:
        return (self._data_root / filename).resolve()

    @staticmethod
    @lru_cache(maxsize=16)
    def _load_csv(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open(mode="r", encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    @staticmethod
    def _resolve_facility(row: dict[str, Any]) -> str:
        return row.get("facility_name") or row.get("facility_code") or "Unknown"

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

    def _csv_extreme_fallback(
        self,
        csv_filename: str,
        value_key: str,
        highest: bool,
        window_start: datetime,
        window_end: datetime,
        extra_keys: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        all_rows = self._load_csv(self._dataset_path(csv_filename))
        filtered: list[dict[str, Any]] = []
        for row in all_rows:
            dt = self._parse_datetime(row.get("date_hour") or row.get("timestamp"))
            if dt is None or not (window_start <= dt < window_end):
                continue
            val = self._to_float(row.get(value_key), default=float("nan"))
            if val != val:
                continue
            entry: dict[str, Any] = {
                "facility": self._resolve_facility(row),
                "metric_value": val,
                "observed_at": row.get("date_hour") or row.get("timestamp") or "",
            }
            for ek in extra_keys:
                entry[ek] = row.get(ek) or "Unknown"
            filtered.append(entry)
        filtered.sort(key=lambda r: float(r.get("metric_value", 0)), reverse=highest)
        return filtered
