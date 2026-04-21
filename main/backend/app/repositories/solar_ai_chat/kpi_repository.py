"""KPI Repository: dynamically queries Gold-layer mart tables.

Uses `DESCRIBE TABLE` to discover schema dynamically, allowing the Chat LLM
to interpret any fields present without hardcoding them in Python.
"""
import logging
from typing import Any

from app.repositories.solar_ai_chat.base_repository import BaseRepository, DatabricksDataUnavailableError

logger = logging.getLogger(__name__)


class KpiRepository(BaseRepository):
    """Dynamic repository for Gold-layer KPI mart tables."""

    _MART_TABLE_MAP = {
        "aqi_impact": "gold.mart_aqi_impact_daily",
        "energy": "gold.mart_energy_daily",
        "forecast_accuracy": "gold.mart_forecast_accuracy_daily",
        "system_kpi": "gold.mart_system_kpi_daily",
        "weather_impact": "gold.mart_weather_impact_daily",
    }

    # Common aliases the LLM invents — map to canonical short names so queries
    # don't fail with "Unknown KPI table" for obviously-valid intent.
    # NOTE: PR / capacity-factor vs weather correlations go to `energy`
    # (mart_energy_daily) which has both performance_ratio_pct AND
    # avg_temperature_c. `weather_impact` only has weather-band aggregates
    # without explicit PR, so it's NOT the right table for those questions.
    _TABLE_ALIASES = {
        # Weather-band aggregates
        "weather": "weather_impact",
        "weather_band": "weather_impact",
        "cloud_band": "weather_impact",
        # PR / capacity-factor vs weather correlations → mart_energy_daily
        "performance_ratio_vs_temperature": "energy",
        "pr_vs_temperature": "energy",
        "pr_vs_weather": "energy",
        "pr_temperature": "energy",
        "temperature": "energy",
        "temperature_impact": "energy",
        "temperature_correlation": "energy",
        "weather_correlation": "energy",
        "capacity_factor_vs_temperature": "energy",
        "performance_ratio": "energy",
        # AQI variants
        "aqi": "aqi_impact",
        "air_quality": "aqi_impact",
        "air_quality_impact": "aqi_impact",
        "aqi_correlation": "aqi_impact",
        # Forecast variants
        "forecast": "forecast_accuracy",
        "actual_vs_forecast": "forecast_accuracy",
        "accuracy": "forecast_accuracy",
        # Energy variants
        "energy_daily": "energy",
        "mart_energy_daily": "energy",
        "daily_energy": "energy",
        # System variants
        "system": "system_kpi",
        "kpi": "system_kpi",
        "daily_kpi": "system_kpi",
    }

    @classmethod
    def _resolve_table_name(cls, raw: str) -> str | None:
        """Return canonical short name or None if no match."""
        if not raw:
            return None
        key = raw.strip().lower()
        if key in cls._MART_TABLE_MAP:
            return key
        return cls._TABLE_ALIASES.get(key)

    def fetch_gold_kpi(
        self,
        table_short_name: str,
        anchor_date: str | None = None,
        station_name: str | None = None,
        limit: int = 30,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Fetch rows from a KPI mart table, discovering columns dynamically."""
        resolved = self._resolve_table_name(table_short_name)
        if resolved is None:
            valid = sorted(self._MART_TABLE_MAP.keys())
            raise ValueError(
                f"Unknown KPI table identifier: '{table_short_name}'. "
                f"Valid names: {valid}. For 'performance ratio vs temperature' "
                f"or any weather correlation question, use 'weather_impact'."
            )
        if resolved != table_short_name:
            logger.info(
                "kpi_table_alias_resolved raw=%r -> %r",
                table_short_name, resolved,
            )
        table_short_name = resolved

        table_path = self._MART_TABLE_MAP[table_short_name]
        
        # We must use `pv` catalog explicitly as requested by architecture
        # or rely on the configured catalog. We'll use the validated base identifier.
        qualified_table = f"{self._catalog}.{table_path}"

        try:
            # 1. Discover Schema
            # DESCRIBE TABLE returns rows with 'col_name', 'data_type', 'comment'
            desc_rows = self._execute_query(f"DESCRIBE TABLE {qualified_table}")
            
            # Simple heuristic to find date and facility columns for filtering.
            # We look for typical column names. If they change, we might not filter optimally
            # but we won't crash either (as long as we don't hard-crash on missing filters).
            columns = []
            date_cols = []
            facility_cols = []

            for row in desc_rows:
                col_name = str(row.get("col_name", "")).lower()
                if not col_name or col_name.startswith("#") or col_name == "":
                    continue # Ignore partition metadata or empty rows returned by DESCRIBE
                columns.append(col_name)
                if "date" in col_name or "time" in col_name:
                    date_cols.append(col_name)
                if "facility" in col_name or "station" in col_name:
                    facility_cols.append(col_name)

        except DatabricksDataUnavailableError:
            raise
        except Exception as exc:
            logger.error(f"Failed to discover schema for {qualified_table}: {exc}")
            raise DatabricksDataUnavailableError(f"Could not read metadata for {table_short_name}.")

        if not columns:
             raise DatabricksDataUnavailableError(f"Table {qualified_table} has no readable columns.")

        # 2. Build Query
        select_clause = "*"  # We could select explicit columns conceptually
        where_clauses = []

        # Handling anchor_date:
        #  - If omitted  → return the most recent N rows (no date filter). This
        #    is the right default for correlation queries since they need many
        #    data points across days.
        #  - If provided but AFTER the latest available date → cap to latest.
        #  - If provided AND exists in data → filter to exactly that date.
        if anchor_date and date_cols:
            from datetime import date as _date
            primary_date_col = date_cols[0]
            try:
                latest = self._resolve_latest_date(table_path)
                provided = _date.fromisoformat(anchor_date)
                if provided > latest:
                    logger.info(
                        "kpi_anchor_date_capped provided=%s latest=%s table=%s",
                        anchor_date, latest, table_path,
                    )
                    anchor_date = latest.isoformat()
            except Exception as _cap_exc:
                logger.debug("kpi_anchor_date_cap_failed: %s", _cap_exc)
            safe_date = anchor_date.replace("'", "''")
            where_clauses.append(f"CAST({primary_date_col} AS DATE) = CAST('{safe_date}' AS DATE)")
        # No date filter when anchor_date omitted — latest N rows via ORDER BY
        # date_col DESC + LIMIT handles it.

        if station_name and str(station_name).lower() not in ("all", "any", "none", "null") and facility_cols:
            primary_fac_col = facility_cols[0]
            safe_station = station_name.replace("'", "''")
            where_clauses.append(f"LOWER({primary_fac_col}) LIKE LOWER('%{safe_station}%')")

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        try:
            limit_value = int(limit) if limit is not None else 100
        except (TypeError, ValueError):
            limit_value = 100
        # Hard cap at 500 so correlation queries get enough data points even
        # when callers pass a very large limit. A single-date query will still
        # return only that day's rows regardless of limit.
        safe_limit = max(1, min(500, limit_value))
        
        order_clause = ""
        if date_cols:
            order_clause = f" ORDER BY {date_cols[0]} DESC"
            
        sql = f"SELECT {select_clause} FROM {qualified_table}{where_sql}{order_clause} LIMIT {safe_limit}"

        try:
             # 3. Execute Query
             data_rows = self._execute_query(sql)
        except Exception as exc:
             logger.error(f"Failed to execute dynamic KPI query: {sql} - {exc}")
             raise DatabricksDataUnavailableError(f"Query on {table_short_name} failed.")

        # Serialize dates and decimals for JSON compatibility
        from datetime import date, datetime
        from decimal import Decimal
        
        serialized_rows = []
        for row in data_rows:
            new_row = {}
            for k, v in row.items():
                if isinstance(v, (date, datetime)):
                    new_row[k] = v.isoformat()
                elif isinstance(v, Decimal):
                    new_row[k] = float(v)
                else:
                    new_row[k] = v
            serialized_rows.append(new_row)

        # Combine results into the standard metric format expected by prompt_builder
        metrics = {
            "table_name": table_path,
            "discovered_columns": columns,
            "rows": serialized_rows,
            "filters_applied": {
                "date_filter": anchor_date if where_clauses and anchor_date else None,
                "station_filter": station_name if where_clauses and station_name else None
            }
        }
        
        sources = [{"layer": "Gold", "dataset": table_path, "data_source": "databricks"}]
        
        return metrics, sources
