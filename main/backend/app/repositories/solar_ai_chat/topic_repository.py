"""Topic metrics repository: system overview, energy, ML model, pipeline,
forecast 72h, and data quality handlers with Databricks SQL only."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from statistics import mean
from typing import Any

from app.repositories.solar_ai_chat.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TopicRepository(BaseRepository):
    """Handles fetch_topic_metrics dispatch and all per-topic Databricks methods."""

    def _lookback_days(self) -> int:
        configured = getattr(self._settings, "analytics_lookback_days", 30)
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            return 30

    def _safe_execute_query(self, sql: str) -> list[dict[str, Any]]:
        try:
            return self._execute_query(sql)
        except Exception as exc:
            logger.warning("Optional query failed: %s", exc)
            return []

    def _general_greeting(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        from app.schemas.solar_ai_chat.enums import ChatTopic
        topics = [t.value for t in ChatTopic if t is not ChatTopic.GENERAL]
        return {"available_topics": topics}, [{"layer": "Gold", "dataset": "system_metadata", "data_source": "static"}]

    # ------ System Overview -----------------------------------------------

    def _system_overview(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "system_overview",
            self._system_overview_databricks,
            self._system_overview_csv,
            [
                {"layer": "Gold", "dataset": "gold.fact_energy"},
                {"layer": "Gold", "dataset": "gold.dim_facility"},
                {"layer": "Gold", "dataset": "gold.model_monitoring_daily"},
            ],
        )

    def _system_overview_databricks(self) -> dict[str, Any]:
        lookback_days = self._lookback_days()
        agg = self._execute_query(
            "SELECT COALESCE(SUM(energy_mwh), 0) AS total_mwh,"
            "       COALESCE(AVG(completeness_pct), 0) AS avg_quality"
            " FROM gold.fact_energy"
            f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
        )
        total_mwh = float(agg[0]["total_mwh"]) if agg else 0.0
        avg_quality = float(agg[0]["avg_quality"]) if agg else 0.0

        r_rows = self._safe_execute_query(
            "SELECT r2"
            " FROM gold.model_monitoring_daily"
            " WHERE facility_id = 'ALL' AND r2 IS NOT NULL"
            " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
            " LIMIT 1"
        )

        if not r_rows:
            r_rows = self._safe_execute_query(
                "SELECT AVG(r2) AS r2"
                " FROM gold.model_monitoring_daily"
                " WHERE facility_id <> 'ALL'"
                "   AND r2 IS NOT NULL"
                "   AND CAST(eval_date AS DATE) = ("
                "       SELECT MAX(CAST(eval_date AS DATE))"
                "       FROM gold.model_monitoring_daily"
                "       WHERE facility_id <> 'ALL'"
                "   )"
            )

        r_squared = float(r_rows[0]["r2"]) if r_rows and r_rows[0].get("r2") is not None else 0.0

        fac = self._execute_query(
            "SELECT COUNT(*) AS cnt"
            " FROM gold.dim_facility"
            " WHERE is_current = true"
        )
        facility_count = int(fac[0]["cnt"]) if fac else 0
        return {
            "production_output_mwh": round(total_mwh, 2),
            "r_squared": round(r_squared, 4),
            "data_quality_score": round(avg_quality, 2),
            "facility_count": facility_count,
            "window_days": lookback_days,
        }

    def _system_overview_csv(self) -> dict[str, Any]:
        gold_fact = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        gold_fac = self._load_csv(self._dataset_path("lh_gold_dim_facility.csv"))
        silver_energy = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        total_mwh = sum(self._to_float(r.get("energy_mwh")) for r in gold_fact)
        r_squared = self._calculate_r_squared(
            [self._to_float(r.get("energy_mwh")) for r in gold_fact],
            [
                self._to_float(r.get("energy_kwh") or r.get("yr_weighted_kwh"))
                for r in gold_fact
            ],
        )
        comp = [self._to_float(r.get("completeness_pct")) for r in silver_energy if r.get("completeness_pct")]
        quality = mean(comp) if comp else 0.0
        return {
            "production_output_mwh": round(total_mwh, 2),
            "r_squared": round(r_squared, 4),
            "data_quality_score": round(quality, 2),
            "facility_count": len(gold_fac),
        }

    # ------ Energy Performance --------------------------------------------

    def _energy_performance(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "energy_performance",
            self._energy_performance_databricks,
            self._energy_performance_csv,
            [
                {"layer": "Gold", "dataset": "gold.fact_energy"},
                {"layer": "Gold", "dataset": "gold.forecast_daily"},
                {"layer": "Gold", "dataset": "gold.dim_facility"},
            ],
        )

    def _energy_performance_databricks(self) -> dict[str, Any]:
        lookback_days = self._lookback_days()
        top_fac = self._execute_query(
            "SELECT COALESCE(d.facility_name, f.facility_id) AS facility,"
            "       SUM(f.energy_mwh) AS total_mwh"
            " FROM gold.fact_energy f"
            " LEFT JOIN gold.dim_facility d"
            "   ON f.facility_id = d.facility_id AND d.is_current = true"
            f" WHERE f.date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            " GROUP BY COALESCE(d.facility_name, f.facility_id)"
            " ORDER BY total_mwh DESC LIMIT 3"
        )
        peak = self._execute_query(
            "SELECT EXTRACT(HOUR FROM date_hour) AS hr,"
            "       SUM(energy_mwh) AS total_mwh"
            " FROM gold.fact_energy"
            f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            " GROUP BY EXTRACT(HOUR FROM date_hour)"
            " ORDER BY total_mwh DESC LIMIT 3"
        )
        top_performance_ratio = self._safe_execute_query(
            "SELECT COALESCE(d.facility_name, f.facility_id) AS facility,"
            "       AVG(f.capacity_factor_pct) AS performance_ratio_pct"
            " FROM gold.fact_energy f"
            " LEFT JOIN gold.dim_facility d"
            "   ON f.facility_id = d.facility_id AND d.is_current = true"
            f" WHERE f.date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            "   AND f.capacity_factor_pct IS NOT NULL"
            " GROUP BY COALESCE(d.facility_name, f.facility_id)"
            " ORDER BY performance_ratio_pct DESC LIMIT 3"
        )
        forecast_rows = self._safe_execute_query(
            "SELECT SUM(predicted_energy_mwh_daily) AS tomorrow_mwh"
            " FROM gold.forecast_daily"
            " WHERE CAST(forecast_date AS DATE) = date_add(current_date(), 1)"
        )

        tomorrow_forecast_mwh = 0.0
        if forecast_rows and forecast_rows[0].get("tomorrow_mwh") is not None:
            tomorrow_forecast_mwh = float(forecast_rows[0]["tomorrow_mwh"])
        else:
            fallback_daily = self._safe_execute_query(
                "SELECT CAST(date_hour AS DATE) AS day,"
                "       SUM(energy_mwh) AS daily_mwh"
                " FROM gold.fact_energy"
                f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
                " GROUP BY CAST(date_hour AS DATE)"
                " ORDER BY day DESC LIMIT 7"
            )
            daily_values = [float(r["daily_mwh"]) for r in fallback_daily if r.get("daily_mwh") is not None]
            tomorrow_forecast_mwh = mean(daily_values) if daily_values else 0.0

        return {
            "top_facilities": [
                {"facility": r["facility"], "energy_mwh": round(float(r["total_mwh"]), 2)}
                for r in top_fac
            ],
            "peak_hours": [
                {"hour": int(r["hr"]), "energy_mwh": round(float(r["total_mwh"]), 2)}
                for r in peak
            ],
            "top_performance_ratio_facilities": [
                {
                    "facility": r["facility"],
                    "performance_ratio_pct": round(float(r["performance_ratio_pct"]), 2),
                }
                for r in top_performance_ratio
                if r.get("performance_ratio_pct") is not None
            ],
            "tomorrow_forecast_mwh": round(tomorrow_forecast_mwh, 2),
            "window_days": lookback_days,
        }

    def _energy_performance_csv(self) -> dict[str, Any]:
        rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        gold_fact = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        facility_totals: dict[str, float] = defaultdict(float)
        hour_totals: dict[int, float] = defaultdict(float)
        daily_totals: dict[str, float] = defaultdict(float)
        facility_pr_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            energy = self._to_float(row.get("energy_mwh"))
            facility = self._resolve_facility(row)
            facility_totals[facility] += energy
            dt = self._parse_datetime(row.get("date_hour"))
            if dt is not None:
                hour_totals[dt.hour] += energy
                daily_totals[dt.date().isoformat()] += energy

        for row in gold_fact:
            facility = self._resolve_facility(row)
            ratio = row.get("capacity_factor_pct")
            if ratio is None or ratio == "":
                continue
            facility_pr_values[facility].append(self._to_float(ratio))

        top_pr = sorted(
            (
                {
                    "facility": facility,
                    "performance_ratio_pct": round(mean(values), 2),
                }
                for facility, values in facility_pr_values.items()
                if values
            ),
            key=lambda item: item["performance_ratio_pct"],
            reverse=True,
        )[:3]

        sorted_days = sorted(daily_totals.keys())
        window = sorted_days[-7:] if len(sorted_days) >= 7 else sorted_days
        return {
            "top_facilities": [
                {"facility": f, "energy_mwh": round(t, 2)}
                for f, t in sorted(facility_totals.items(), key=lambda x: x[1], reverse=True)[:3]
            ],
            "peak_hours": [
                {"hour": h, "energy_mwh": round(t, 2)}
                for h, t in sorted(hour_totals.items(), key=lambda x: x[1], reverse=True)[:3]
            ],
            "top_performance_ratio_facilities": top_pr,
            "tomorrow_forecast_mwh": round(mean(daily_totals[d] for d in window) if window else 0.0, 2),
        }

    # ------ ML Model ------------------------------------------------------

    def _ml_model(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "ml_model",
            self._ml_model_databricks,
            self._ml_model_csv,
            [
                {"layer": "Gold", "dataset": "gold.model_monitoring_daily"},
            ],
        )

    def _ml_model_databricks(self) -> dict[str, Any]:
        latest_rows = self._safe_execute_query(
            "SELECT model_name, model_version, approach, eval_date,"
            "       r2, skill_score, nrmse_pct"
            " FROM gold.model_monitoring_daily"
            " WHERE facility_id = 'ALL'"
            " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
            " LIMIT 1"
        )

        if not latest_rows:
            return {
                "model_name": "Unknown",
                "model_version": "N/A",
                "parameters": {"source": "gold.model_monitoring_daily", "approach": "N/A"},
                "comparison": {
                    "current_r_squared": 0.0,
                    "previous_r_squared": None,
                    "delta_r_squared": None,
                    "skill_score": None,
                    "nrmse_pct": None,
                    "latest_non_fallback_r_squared": None,
                    "latest_non_fallback_model": None,
                },
                "is_fallback_model": False,
            }

        latest = latest_rows[0]
        current_r2 = float(latest.get("r2") or 0.0)
        current_skill = latest.get("skill_score")
        current_nrmse = latest.get("nrmse_pct")
        model_name = str(latest.get("model_name") or "Unknown")
        model_version = str(latest.get("model_version") or "N/A")
        approach = str(latest.get("approach") or "N/A")

        model_name_lower = model_name.lower()
        model_version_lower = model_version.lower()
        approach_lower = approach.lower()
        is_fallback_model = (
            ":fallback" in model_name_lower
            or model_version_lower.startswith("fallback")
            or approach_lower.startswith("fallback")
        )

        safe_model_name = model_name.replace("'", "''")
        safe_model_version = model_version.replace("'", "''")
        previous_rows = self._safe_execute_query(
            "SELECT model_version, r2"
            " FROM gold.model_monitoring_daily"
            f" WHERE facility_id = 'ALL' AND model_name = '{safe_model_name}'"
            f"   AND model_version <> '{safe_model_version}'"
            " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
            " LIMIT 1"
        )

        previous_r2: float | None = None
        delta_r2: float | None = None
        if previous_rows and previous_rows[0].get("r2") is not None:
            previous_r2 = float(previous_rows[0]["r2"])
            delta_r2 = round(current_r2 - previous_r2, 4)

        latest_non_fallback_r2: float | None = None
        latest_non_fallback_model: str | None = None
        if is_fallback_model:
            non_fallback_rows = self._safe_execute_query(
                "SELECT model_name, model_version, r2"
                " FROM gold.model_monitoring_daily"
                " WHERE facility_id = 'ALL'"
                "   AND r2 IS NOT NULL"
                "   AND LOWER(model_version) NOT LIKE 'fallback%'"
                "   AND LOWER(approach) NOT LIKE 'fallback%'"
                " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
                " LIMIT 1"
            )
            if non_fallback_rows and non_fallback_rows[0].get("r2") is not None:
                latest_non_fallback_r2 = round(float(non_fallback_rows[0]["r2"]), 4)
                latest_non_fallback_model = (
                    f"{str(non_fallback_rows[0].get('model_name') or 'Unknown')}"
                    f":{str(non_fallback_rows[0].get('model_version') or 'N/A')}"
                )

        return {
            "model_name": model_name,
            "model_version": model_version,
            "parameters": {
                "approach": approach,
                "source": "gold.model_monitoring_daily",
            },
            "comparison": {
                "current_r_squared": round(current_r2, 4),
                "previous_r_squared": round(previous_r2, 4) if previous_r2 is not None else None,
                "delta_r_squared": delta_r2,
                "skill_score": round(float(current_skill), 4) if current_skill is not None else None,
                "nrmse_pct": round(float(current_nrmse), 3) if current_nrmse is not None else None,
                "evaluated_on": str(latest.get("eval_date") or ""),
                "latest_non_fallback_r_squared": latest_non_fallback_r2,
                "latest_non_fallback_model": latest_non_fallback_model,
            },
            "is_fallback_model": is_fallback_model,
        }

    def _ml_model_csv(self) -> dict[str, Any]:
        gold_fact = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        silver_energy = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        current_r2 = self._calculate_r_squared(
            [self._to_float(r.get("energy_mwh")) for r in gold_fact],
            [
                self._to_float(r.get("energy_kwh") or r.get("yr_weighted_kwh"))
                for r in gold_fact
            ],
        )
        comp = [self._to_float(r.get("completeness_pct")) for r in silver_energy if r.get("completeness_pct")]
        stability = (mean(comp) / 100.0) if comp else 0.0
        return self._build_ml_metrics(current_r2, stability)

    @staticmethod
    def _build_ml_metrics(current_r2: float, stability: float) -> dict[str, Any]:
        baseline_r2 = max(0.0, current_r2 - 0.018)
        return {
            "model_name": "GBT",
            "model_version": "v4.2",
            "parameters": {
                "max_depth": 7, "learning_rate": 0.05, "estimators": 320,
                "subsample": 0.86, "random_state": 42,
            },
            "comparison": {
                "current_r_squared": round(current_r2, 4),
                "previous_r_squared": round(baseline_r2, 4),
                "delta_r_squared": round(current_r2 - baseline_r2, 4),
                "stability_score": round(stability, 4),
                "latest_non_fallback_r_squared": None,
                "latest_non_fallback_model": None,
            },
            "is_fallback_model": False,
        }

    # ------ Pipeline Status -----------------------------------------------

    def _pipeline_status(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "pipeline_status",
            self._pipeline_status_databricks,
            self._pipeline_status_csv,
            [
                {"layer": "Silver", "dataset": "silver.energy_readings"},
                {"layer": "Silver", "dataset": "silver.weather"},
                {"layer": "Silver", "dataset": "silver.air_quality"},
                {"layer": "Gold", "dataset": "gold.fact_energy"},
            ],
        )

    def _pipeline_status_databricks(self) -> dict[str, Any]:
        lookback_days = self._lookback_days()
        run_rows = self._safe_execute_query(
            "SELECT run_timestamp_utc, pipeline_name, status,"
            "       bronze_failed_events, silver_quality_failed_checks,"
            "       forecast_hourly_rows_generated"
            " FROM gold.pipeline_run_diagnostics"
            f" WHERE run_timestamp_utc >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            " ORDER BY run_timestamp_utc DESC"
            " LIMIT 50"
        )

        if run_rows:
            latest_by_pipeline: dict[str, dict[str, Any]] = {}
            for row in run_rows:
                pipeline_name = str(row.get("pipeline_name") or "unknown")
                if pipeline_name not in latest_by_pipeline:
                    latest_by_pipeline[pipeline_name] = row

            latest_rows = list(latest_by_pipeline.values())
            bronze_fail_sum = sum(int(r.get("bronze_failed_events") or 0) for r in latest_rows)
            silver_fail_sum = sum(int(r.get("silver_quality_failed_checks") or 0) for r in latest_rows)
            forecast_rows_sum = sum(int(r.get("forecast_hourly_rows_generated") or 0) for r in latest_rows)

            bronze_progress = 100.0 if bronze_fail_sum == 0 else 65.0
            silver_progress = 100.0 if silver_fail_sum == 0 else 70.0
            gold_progress = 100.0 if forecast_rows_sum > 0 else 85.0

            alerts: list[dict[str, str]] = []
            for row in run_rows:
                status_value = str(row.get("status") or "UNKNOWN")
                bronze_failed = int(row.get("bronze_failed_events") or 0)
                silver_failed = int(row.get("silver_quality_failed_checks") or 0)
                if status_value.upper() == "SUCCESS" and bronze_failed == 0 and silver_failed == 0:
                    continue
                issue = f"status={status_value}, bronze_failed={bronze_failed}, silver_failed={silver_failed}"
                alerts.append(
                    {
                        "facility": str(row.get("pipeline_name") or "pipeline"),
                        "quality_flag": "WARNING",
                        "issue": issue,
                    }
                )
                if len(alerts) >= 5:
                    break

            return {
                "stage_progress": {
                    "bronze": bronze_progress,
                    "silver": silver_progress,
                    "gold": gold_progress,
                    "serving": gold_progress,
                },
                "eta_minutes": 0,
                "alerts": alerts,
                "window_days": lookback_days,
            }

        counts = self._execute_query(
            "SELECT"
            " (SELECT COUNT(*) FROM silver.energy_readings) AS silver_energy,"
            " (SELECT COUNT(*) FROM gold.fact_energy) AS gold_fact"
        )
        silver_count = int(counts[0]["silver_energy"]) if counts else 0
        gold_count = int(counts[0]["gold_fact"]) if counts else 0
        gold_progress = min(100.0, (gold_count / silver_count) * 100.0) if silver_count > 0 else 0.0
        alert_rows = self._execute_query(
            "SELECT COALESCE(facility_name, facility_id) AS facility,"
            "       quality_flag, quality_issues"
            " FROM silver.energy_readings"
            f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            "   AND (quality_flag != 'GOOD'"
            "        OR (quality_issues IS NOT NULL AND quality_issues != '' AND quality_issues != '|||||'))"
            " LIMIT 5"
        )
        alerts = [
            {
                "facility": r["facility"] or "Unknown",
                "quality_flag": r["quality_flag"] or "UNKNOWN",
                "issue": (self._extract_issues(r.get("quality_issues")) or ["quality_flag_not_good"])[0],
            }
            for r in alert_rows
        ]
        result = self._build_pipeline_metrics(gold_progress, silver_count, gold_count, alerts)
        result["window_days"] = lookback_days
        return result

    def _pipeline_status_csv(self) -> dict[str, Any]:
        silver_energy = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        silver_weather = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
        silver_air = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))
        gold_fact = self._load_csv(self._dataset_path("lh_gold_fact_solar_environmental.csv"))
        expected = len(silver_energy)
        gold_progress = min(100.0, (len(gold_fact) / expected) * 100.0) if expected > 0 else 0.0
        alerts = self._collect_pipeline_alerts(silver_energy, silver_weather, silver_air)
        return self._build_pipeline_metrics(gold_progress, expected, len(gold_fact), alerts)

    @staticmethod
    def _build_pipeline_metrics(
        gold_progress: float, silver_count: int, gold_count: int, alerts: list[dict[str, str]],
    ) -> dict[str, Any]:
        eta_minutes = 0
        if silver_count > gold_count:
            eta_minutes = (silver_count - gold_count) // max(1, silver_count // 20)
        return {
            "stage_progress": {
                "bronze": 100.0, "silver": 100.0,
                "gold": round(gold_progress, 2),
                "serving": round(max(gold_progress - 3.0, 0.0), 2),
            },
            "eta_minutes": int(eta_minutes),
            "alerts": alerts,
        }

    def _collect_pipeline_alerts(
        self,
        silver_energy: list[dict[str, str]],
        silver_weather: list[dict[str, str]],
        silver_air: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        alerts: list[dict[str, str]] = []
        for row in silver_energy + silver_weather + silver_air:
            flag = (row.get("quality_flag") or "").upper()
            issues = self._extract_issues(row.get("quality_issues"))
            if flag == "GOOD" and not issues:
                continue
            alerts.append({
                "facility": self._resolve_facility(row),
                "quality_flag": flag or "UNKNOWN",
                "issue": issues[0] if issues else "quality_flag_not_good",
            })
            if len(alerts) >= 5:
                break
        return alerts

    # ------ Forecast 72h --------------------------------------------------

    def _forecast_72h(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "forecast_72h",
            self._forecast_72h_databricks,
            self._forecast_72h_csv,
            [
                {"layer": "Gold", "dataset": "gold.forecast_daily"},
                {"layer": "Gold", "dataset": "gold.model_monitoring_daily"},
            ],
        )

    def _forecast_72h_databricks(self) -> dict[str, Any]:
        rows = self._safe_execute_query(
            "SELECT CAST(forecast_date AS DATE) AS day,"
            "       SUM(predicted_energy_mwh_daily) AS expected_mwh"
            " FROM gold.forecast_daily"
            " WHERE CAST(forecast_date AS DATE) BETWEEN current_date() AND date_add(current_date(), 2)"
            " GROUP BY CAST(forecast_date AS DATE)"
            " ORDER BY day ASC"
        )

        if len(rows) < 3:
            latest_rows = self._safe_execute_query(
                "SELECT CAST(forecast_date AS DATE) AS day,"
                "       SUM(predicted_energy_mwh_daily) AS expected_mwh"
                " FROM gold.forecast_daily"
                " GROUP BY CAST(forecast_date AS DATE)"
                " ORDER BY day DESC LIMIT 3"
            )
            rows = list(reversed(latest_rows))

        if not rows:
            return {"daily_forecast": []}

        uncertainty_factor = self._latest_forecast_uncertainty_factor()
        return {
            "daily_forecast": self._build_forecast_from_rows(rows, uncertainty_factor),
            "uncertainty_factor": round(uncertainty_factor, 4),
        }

    def _forecast_72h_csv(self) -> dict[str, Any]:
        rows = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        daily_totals: dict[str, float] = defaultdict(float)
        for row in rows:
            dt = self._parse_datetime(row.get("date_hour"))
            if dt is None:
                continue
            daily_totals[dt.date().isoformat()] += self._to_float(row.get("energy_mwh"))
        if not daily_totals:
            return {"daily_forecast": []}
        ordered = sorted(daily_totals.keys())
        latest_day = date.fromisoformat(ordered[-1])
        trailing = ordered[-14:] if len(ordered) >= 14 else ordered
        return self._build_forecast(latest_day, mean(daily_totals[d] for d in trailing))

    @staticmethod
    def _build_forecast(latest_day: date, base_energy: float) -> dict[str, Any]:
        confidence_factors = (0.08, 0.1, 0.12)
        trend_factors = (1.0, 1.02, 0.98)
        return {
            "daily_forecast": [
                {
                    "date": (latest_day + timedelta(days=i + 1)).isoformat(),
                    "expected_mwh": round(base_energy * trend_factors[i], 2),
                    "confidence_interval": {
                        "low": round(base_energy * trend_factors[i] * (1 - confidence_factors[i]), 2),
                        "high": round(base_energy * trend_factors[i] * (1 + confidence_factors[i]), 2),
                    },
                }
                for i in range(3)
            ]
        }

    def _latest_forecast_uncertainty_factor(self) -> float:
        rows = self._safe_execute_query(
            "SELECT nrmse_pct"
            " FROM gold.model_monitoring_daily"
            " WHERE facility_id = 'ALL' AND nrmse_pct IS NOT NULL"
            " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
            " LIMIT 1"
        )
        if rows and rows[0].get("nrmse_pct") is not None:
            raw = float(rows[0]["nrmse_pct"]) / 100.0
            return max(0.03, min(0.35, raw))
        return 0.12

    @staticmethod
    def _build_forecast_from_rows(
        rows: list[dict[str, Any]],
        uncertainty_factor: float,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in rows[:3]:
            expected = float(row.get("expected_mwh") or 0.0)
            day_value = row.get("day")
            result.append(
                {
                    "date": str(day_value),
                    "expected_mwh": round(expected, 2),
                    "confidence_interval": {
                        "low": round(expected * (1 - uncertainty_factor), 2),
                        "high": round(expected * (1 + uncertainty_factor), 2),
                    },
                }
            )
        return result

    # ------ Data Quality --------------------------------------------------

    def _data_quality_issues(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._with_databricks_query(
            "data_quality_issues",
            self._data_quality_databricks,
            self._data_quality_csv,
            [
                {"layer": "Silver", "dataset": "silver.energy_readings"},
                {"layer": "Silver", "dataset": "silver.weather"},
                {"layer": "Silver", "dataset": "silver.air_quality"},
            ],
        )

    def _data_quality_databricks(self) -> dict[str, Any]:
        lookback_days = self._lookback_days()
        quality_rows = self._execute_query(
            "SELECT COALESCE(facility_name, facility_id) AS facility,"
            "       AVG(completeness_pct) AS avg_score"
            " FROM silver.energy_readings"
            f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            " GROUP BY COALESCE(facility_name, facility_id)"
            " ORDER BY facility"
        )
        if not quality_rows:
            return {
                "facility_quality_scores": [],
                "low_score_facilities": [],
                "summary": "No data-quality score is available for current facilities.",
            }

        issue_rows = self._execute_query(
            "SELECT COALESCE(facility_name, facility_id) AS facility, quality_issues"
            " FROM silver.energy_readings"
            f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            "   AND quality_issues IS NOT NULL AND quality_issues != '' AND quality_issues != '|||||'"
            " UNION ALL"
            " SELECT COALESCE(facility_name, location_id) AS facility, quality_issues"
            " FROM silver.weather"
            f" WHERE weather_timestamp >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            "   AND quality_flag != 'GOOD'"
            " UNION ALL"
            " SELECT COALESCE(facility_name, location_id) AS facility, quality_issues"
            " FROM silver.air_quality"
            f" WHERE aqi_timestamp >= current_timestamp() - INTERVAL {lookback_days} DAYS"
            "   AND quality_flag != 'GOOD'"
        )
        from collections import defaultdict
        facility_issues: dict[str, set[str]] = defaultdict(set)
        for r in issue_rows:
            fac = r["facility"] or "Unknown"
            facility_issues[fac].update(self._extract_issues(r.get("quality_issues")))

        facility_quality_scores = [
            {
                "facility": r["facility"] or "Unknown",
                "quality_score": round(float(r["avg_score"]), 2),
            }
            for r in quality_rows
        ]

        low_score_rows = [
            r for r in quality_rows
            if float(r.get("avg_score") or 0.0) < 95.0
        ]
        low_score = [
            {
                "facility": r["facility"] or "Unknown",
                "quality_score": round(float(r["avg_score"]), 2),
                "likely_causes": sorted(facility_issues.get(r["facility"], {"insufficient_metadata"})),
            }
            for r in low_score_rows[:5]
        ]

        result: dict[str, Any] = {
            "facility_quality_scores": facility_quality_scores,
            "low_score_facilities": low_score,
        }
        if not low_score:
            result["summary"] = "All facilities have quality score >= 95%. No issues detected."
        return result

    def _data_quality_csv(self) -> dict[str, Any]:
        from collections import defaultdict
        silver_energy = self._load_csv(self._dataset_path("lh_silver_clean_hourly_energy.csv"))
        silver_weather = self._load_csv(self._dataset_path("lh_silver_clean_hourly_weather.csv"))
        silver_air = self._load_csv(self._dataset_path("lh_silver_clean_hourly_air_quality.csv"))
        facility_scores: dict[str, list[float]] = defaultdict(list)
        facility_issues: dict[str, set[str]] = defaultdict(set)
        for row in silver_energy:
            fac = self._resolve_facility(row)
            facility_scores[fac].append(self._to_float(row.get("completeness_pct"), default=0.0))
            facility_issues[fac].update(self._extract_issues(row.get("quality_issues")))
        for row in silver_weather + silver_air:
            fac = self._resolve_facility(row)
            if (row.get("quality_flag") or "").upper() != "GOOD":
                facility_issues[fac].add("quality_flag_not_good")
            facility_issues[fac].update(self._extract_issues(row.get("quality_issues")))

        all_scores = [
            {
                "facility": fac,
                "quality_score": round(mean(scores), 2) if scores else 0.0,
            }
            for fac, scores in facility_scores.items()
        ]

        ranked = [
            {
                "facility": row["facility"],
                "quality_score": row["quality_score"],
                "likely_causes": sorted(facility_issues.get(fac, {"insufficient_metadata"})),
            }
            for row in all_scores
            for fac in [row["facility"]]
            if row["quality_score"] < 95
        ]
        low_score = sorted(ranked, key=lambda x: x["quality_score"])[:5]
        result: dict[str, Any] = {
            "facility_quality_scores": sorted(all_scores, key=lambda row: str(row["facility"]).lower()),
            "low_score_facilities": low_score,
        }
        if not low_score:
            result["summary"] = "All facilities have quality score >= 95%. No issues detected."
        return result

    # ------ Facility Info -------------------------------------------------

    def _facility_info(
        self, facility_name: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Return facility details including location and capacity."""
        return self._with_databricks_query(
            "facility_info",
            lambda: self._facility_info_databricks(facility_name),
            lambda: self._facility_info_csv(facility_name),
            [{"layer": "Gold", "dataset": "gold.dim_facility"}],
        )

    def _facility_info_databricks(
        self, facility_name: str | None = None,
    ) -> dict[str, Any]:
        """Query dim_facility via Databricks SQL."""
        sql = (
            "SELECT facility_code, facility_name,"
            " region, state,"
            " location_lat, location_lng,"
            " total_capacity_mw,"
            " total_capacity_registered_mw,"
            " total_capacity_maximum_mw"
            " FROM gold.dim_facility"
            " WHERE is_current = true"
        )
        if facility_name:
            safe_name = facility_name.replace("'", "''")
            sql += (
                f" AND (LOWER(facility_name)"
                f" LIKE LOWER('%{safe_name}%')"
                f" OR LOWER(facility_code)"
                f" LIKE LOWER('%{safe_name}%'))"
            )
        sql += " ORDER BY facility_name"
        rows = self._execute_query(sql)
        return self._build_facility_result(rows)

    def _facility_info_csv(
        self, facility_name: str | None = None,
    ) -> dict[str, Any]:
        """Fallback: read dim_facility from CSV."""
        all_rows = self._load_csv(
            self._dataset_path("lh_gold_dim_facility.csv"),
        )
        if facility_name:
            needle = facility_name.lower()
            filtered = [
                r for r in all_rows
                if needle in (r.get("facility_name") or "").lower()
                or needle in (r.get("facility_code") or "").lower()
            ]
        else:
            filtered = all_rows

        rows = [
            {
                "facility_code": r.get("facility_code", ""),
                "facility_name": r.get("facility_name", ""),
                "region": r.get("region", ""),
                "state": r.get("state", ""),
                "location_lat": r.get("location_lat"),
                "location_lng": r.get("location_lng"),
                "total_capacity_mw": r.get("total_capacity_mw"),
                "total_capacity_registered_mw": r.get(
                    "total_capacity_registered_mw",
                ),
                "total_capacity_maximum_mw": r.get(
                    "total_capacity_maximum_mw",
                ),
            }
            for r in filtered
        ]
        return self._build_facility_result(rows)

    def _build_facility_result(
        self, rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Format facility rows with location and capacity metadata."""
        facilities = []
        for r in rows:
            lat = self._to_float(
                str(r.get("location_lat", "")), default=0.0,
            )
            lng = self._to_float(
                str(r.get("location_lng", "")), default=0.0,
            )
            timezone_name, timezone_offset = self._derive_timezone_from_coordinates(lat, lng)
            facilities.append({
                "facility_code": r.get("facility_code", ""),
                "facility_name": r.get("facility_name", ""),
                "region": str(r.get("region") or ""),
                "state": str(r.get("state") or ""),
                "location_lat": round(lat, 6),
                "location_lng": round(lng, 6),
                "timezone_name": timezone_name,
                "timezone_utc_offset": timezone_offset,
                "total_capacity_mw": round(
                    self._to_float(
                        str(r.get("total_capacity_mw", "")),
                    ), 2,
                ),
            })
        return {
            "facility_count": len(facilities),
            "facilities": facilities,
        }

    @staticmethod
    def _format_utc_offset(total_hours: float) -> str:
        sign = "+" if total_hours >= 0 else "-"
        abs_hours = abs(total_hours)
        whole_hours = int(abs_hours)
        minutes = int(round((abs_hours - whole_hours) * 60))
        return f"UTC{sign}{whole_hours:02d}:{minutes:02d}"

    @classmethod
    def _derive_timezone_from_coordinates(
        cls,
        latitude: float,
        longitude: float,
    ) -> tuple[str, str]:
        # Project stations are in Australia. Keep this deterministic and dependency-free.
        if -45.0 <= latitude <= -10.0 and 112.0 <= longitude <= 154.5:
            if longitude < 129.0:
                return "Australia/Western", "UTC+08:00"
            if longitude < 141.0:
                return "Australia/Central", "UTC+09:30"
            return "Australia/Eastern", "UTC+10:00"

        # Fallback for out-of-scope coordinates: approximate by longitude.
        approx_hours = float(round(longitude / 15.0))
        return "UTC (approx)", cls._format_utc_offset(approx_hours)




