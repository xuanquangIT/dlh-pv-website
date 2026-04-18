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

    def fetch_gold_kpi(
        self,
        table_short_name: str,
        anchor_date: str | None = None,
        station_name: str | None = None,
        limit: int = 30,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Fetch rows from a KPI mart table, discovering columns dynamically."""
        if table_short_name not in self._MART_TABLE_MAP:
            raise ValueError(f"Unknown KPI table identifier: {table_short_name}")

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
        
        if anchor_date and date_cols:
            # Pick the primary date column. Usually the first one found is good enough.
            primary_date_col = date_cols[0]
            # Use CAST to ensure safe comparison regardless of string/date/timestamp type
            safe_date = anchor_date.replace("'", "''") 
            where_clauses.append(f"CAST({primary_date_col} AS DATE) = CAST('{safe_date}' AS DATE)")

        if station_name and facility_cols:
            primary_fac_col = facility_cols[0]
            safe_station = station_name.replace("'", "''")
            where_clauses.append(f"LOWER({primary_fac_col}) LIKE LOWER('%{safe_station}%')")

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        safe_limit = max(1, min(100, limit))
        
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

        # Combine results into the standard metric format expected by prompt_builder
        metrics = {
            "table_name": table_path,
            "discovered_columns": columns,
            "rows": data_rows,
            "filters_applied": {
                "date_filter": anchor_date if where_clauses and anchor_date else None,
                "station_filter": station_name if where_clauses and station_name else None
            }
        }
        
        sources = [{"layer": "Gold", "dataset": table_path, "data_source": "databricks"}]
        
        return metrics, sources
