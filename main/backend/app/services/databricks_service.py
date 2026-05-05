import logging
import os
import re
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import CronSchedule, JobSettings, PauseStatus
from app.core.settings import get_solar_chat_settings
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Defensive SQL-interpolation validators ────────────────────────
_ALLOWED_HORIZONS = frozenset({1, 3, 5, 7})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _assert_valid_horizon(horizon: int) -> None:
    if horizon not in _ALLOWED_HORIZONS:
        raise ValueError(f"Invalid horizon {horizon}; must be one of {sorted(_ALLOWED_HORIZONS)}")


def _assert_valid_date(value: str, param_name: str) -> None:
    if not _DATE_RE.match(value):
        raise ValueError(f"Invalid {param_name}: must be YYYY-MM-DD, got {value!r}")

@lru_cache(maxsize=1)
def get_databricks_client() -> WorkspaceClient:
    """
    Initializes and returns the Databricks WorkspaceClient.
    Picks up credentials from SolarChatSettings which reads from .env.
    """
    settings = get_solar_chat_settings()
    return WorkspaceClient(
        host=settings.databricks_host,
        token=settings.databricks_token
    )

def get_job_info(job_id: int) -> dict:
    client = get_databricks_client()
    job = client.jobs.get(job_id=job_id)
    return job.as_dict()

def get_all_jobs() -> list[dict]:
    """Return a compact list of all Databricks jobs (id + name + schedule status)."""
    client = get_databricks_client()
    jobs = []
    # expand_tasks=False returns lightweight BaseJob objects without full task spec
    for job in client.jobs.list(expand_tasks=False):
        try:
            job_id  = job.job_id
            name    = (job.settings.name if job.settings else None) or f"Job {job_id}"
            schedule = job.settings.schedule if job.settings else None
            pause_status = (
                schedule.pause_status.value
                if schedule and schedule.pause_status
                else "UNPAUSED"
            )
            jobs.append({
                "job_id":       job_id,
                "name":         name,
                "pause_status": pause_status,
            })
        except Exception:
            # Skip any malformed entries silently
            continue
    jobs.sort(key=lambda j: j["name"].lower())
    return jobs


def get_job_runs(job_id: int, limit: int = 5) -> list[dict]:
    client = get_databricks_client()
    runs_iter = client.jobs.list_runs(job_id=job_id, limit=limit)
    return [run.as_dict() for run in runs_iter]

def trigger_job_run(job_id: int) -> dict:
    client = get_databricks_client()
    run_response = client.jobs.run_now(job_id=job_id)
    return {"run_id": run_response.run_id}

def get_run(run_id: int) -> dict:
    client = get_databricks_client()
    return client.jobs.get_run(run_id=run_id).as_dict()

def cancel_job_run(run_id: int) -> dict:
    client = get_databricks_client()
    client.jobs.cancel_run(run_id=run_id)
    return {"message": f"Run {run_id} cancelled successfully"}


def update_job_schedule(
    job_id: int,
    quartz_cron_expression: str,
    timezone_id: str | None = None,
    pause_status: str | None = None,
) -> dict:
    client = get_databricks_client()

    job = client.jobs.get(job_id=job_id)
    current_schedule = job.settings.schedule if job and job.settings else None

    resolved_timezone = timezone_id or (current_schedule.timezone_id if current_schedule else None) or "UTC"
    resolved_pause = pause_status or (
        current_schedule.pause_status.value if current_schedule and current_schedule.pause_status else "UNPAUSED"
    )

    try:
        pause_enum = PauseStatus[resolved_pause.upper()]
    except KeyError as exc:
        raise ValueError("pause_status must be 'PAUSED' or 'UNPAUSED'.") from exc

    new_schedule = CronSchedule(
        quartz_cron_expression=quartz_cron_expression.strip(),
        timezone_id=resolved_timezone,
        pause_status=pause_enum,
    )

    client.jobs.update(job_id=job_id, new_settings=JobSettings(schedule=new_schedule))
    updated_job = client.jobs.get(job_id=job_id)
    updated_schedule = updated_job.settings.schedule if updated_job and updated_job.settings else None

    return {
        "job_id": job_id,
        "message": "Schedule updated successfully",
        "schedule": {
            "quartz_cron_expression": updated_schedule.quartz_cron_expression if updated_schedule else quartz_cron_expression,
            "timezone_id": updated_schedule.timezone_id if updated_schedule else resolved_timezone,
            "pause_status": (
                updated_schedule.pause_status.value if updated_schedule and updated_schedule.pause_status else resolved_pause
            ),
        },
    }

def execute_sql(query: str) -> list[dict]:
    client = get_databricks_client()
    settings = get_solar_chat_settings()
    warehouse_id = settings.databricks_warehouse_id
    if not warehouse_id:
        raise ValueError("DATABRICKS_WAREHOUSE_ID is not set.")

    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="30s"
    )

    if not response or not response.manifest or not response.result:
        return []

    columns = [col.name for col in response.manifest.schema.columns]
    data = response.result.data_array or []

    results = []
    for row in data:
        results.append(dict(zip(columns, row)))
    return results

def get_quality_summary_metrics() -> dict:
    # Only aggregate hourly-granularity tables (energy, weather, air_quality).
    # facility_status is SCD2 (one row per change event, not per hour) so
    # including it skews the totals. We count distinct date_hour slots for
    # energy_readings as the primary denominator source.
    query = """
    WITH all_data AS (
      SELECT quality_flag FROM pv.silver.energy_readings
      UNION ALL
      SELECT quality_flag FROM pv.silver.weather
      UNION ALL
      SELECT quality_flag FROM pv.silver.air_quality
    )
    SELECT
      SUM(CASE WHEN quality_flag = 'GOOD'    THEN 1 ELSE 0 END) as valid,
      SUM(CASE WHEN quality_flag = 'WARNING' THEN 1 ELSE 0 END) as warning,
      SUM(CASE WHEN quality_flag = 'BAD'     THEN 1 ELSE 0 END) as invalid
    FROM all_data
    """
    results = execute_sql(query)
    if results:
        row = results[0]
        valid = int(row.get('valid') or 0)
        warning = int(row.get('warning') or 0)
        invalid = int(row.get('invalid') or 0)
    else:
        valid = warning = invalid = 0

    total = valid + warning + invalid

    if total > 0:
        valid_ratio   = f"{(valid   / total) * 100:.1f}%"
        warning_ratio = f"{(warning / total) * 100:.1f}%"
        invalid_ratio = f"{(invalid / total) * 100:.1f}%"
    else:
        valid_ratio = warning_ratio = invalid_ratio = "0.0%"

    return {
        "valid":         f"{valid:,}",
        "warning":       f"{warning:,}",
        "invalid":       f"{invalid:,}",
        "valid_ratio":   f"{valid_ratio} of total",
        "warning_ratio": f"{warning_ratio} of total",
        "invalid_ratio": f"{invalid_ratio} of total",
    }

def get_facility_quality_scores() -> list[dict]:
    query = """
    WITH all_data AS (
      SELECT facility_id as facility, quality_flag FROM pv.silver.energy_readings
      UNION ALL
      SELECT location_id as facility, quality_flag FROM pv.silver.weather
      UNION ALL
      SELECT location_id as facility, quality_flag FROM pv.silver.air_quality
      UNION ALL
      SELECT facility_id as facility, quality_flag FROM pv.silver.facility_status
    )
    SELECT
      facility,
      COUNT(*) as total,
      SUM(CASE WHEN quality_flag = 'GOOD' THEN 1 ELSE 0 END) as valid,
      SUM(CASE WHEN quality_flag = 'WARNING' THEN 1 ELSE 0 END) as warning,
      SUM(CASE WHEN quality_flag = 'BAD' THEN 1 ELSE 0 END) as invalid
    FROM all_data
    GROUP BY facility
    ORDER BY facility
    """
    results = execute_sql(query)
    scores = []
    for row in results:
        total = int(row.get('total') or 0)
        valid = int(row.get('valid') or 0)
        warning = int(row.get('warning') or 0)
        invalid = int(row.get('invalid') or 0)
        if total > 0:
            valid_pct = (valid / total) * 100
            scores.append({
                "facility": str(row.get('facility')),
                "valid": f"{valid_pct:.1f}%",
                "warning": f"{(warning / total) * 100:.1f}%",
                "invalid": f"{(invalid / total) * 100:.1f}%",
                "score": f"{int(valid_pct)}"
            })
    return scores

def get_recent_quality_issues() -> list[dict]:
    # status column in data_quality_log is either 'PASS' or 'FAIL' (from silver_utils.py).
    # Severity is derived from failed_rate (0.0-1.0 ratio, NOT 0-100%):
    #   - failed_rate = 0   -> unlikely for a FAIL row, treat as WARNING
    #   - failed_rate <= 0.05 -> WARNING  (up to 5% rows failed)
    #   - failed_rate >  0.05 -> BAD      (more than 5% rows failed)
    # rule_type column stores the check category (duplicate_key, null_rate,
    # range_check, ratio_bounds, freshness, timestamp_parse) — use it as 'sensor'.
    query = """
    SELECT
      date_format(check_timestamp, 'HH:mm z')   AS time,
      table_name                                 AS facility,
      rule_type                                  AS sensor,
      rule_name                                  AS issue,
      CONCAT(CAST(failed_rows AS STRING), ' records') AS affected,
      CAST(failed_rate AS DOUBLE)                AS failed_rate,
      CASE
        WHEN CAST(failed_rate AS DOUBLE) <= 0.05 THEN 'WARNING'
        ELSE 'BAD'
      END                                        AS severity,
      'Flagged'                                  AS action
    FROM pv.silver.data_quality_log
    WHERE status = 'FAIL'
    ORDER BY check_timestamp DESC
    LIMIT 10
    """
    results = execute_sql(query)
    issues = []
    for row in results:
        issues.append({
            "time":     row.get('time', ''),
            "facility": row.get('facility', ''),
            "sensor":   row.get('sensor', ''),
            "issue":    row.get('issue', ''),
            "affected": row.get('affected', ''),
            "severity": row.get('severity', ''),
            "action":   row.get('action', ''),
        })
    return issues

def get_facility_heatmap_data() -> list[dict]:
    # Only use hourly-granularity tables for the heatmap to keep score
    # denominators consistent across facilities and dates.
    # facility_status is excluded: it's SCD2 and has only 1 row per change event,
    # so including it distorts the per-day score.
    # Upper bound is current_date() (no dependency on energy_readings max date
    # which could lag behind weather/air_quality tables).
    query = """
    WITH all_data AS (
      SELECT reading_date AS date, facility_id AS facility, quality_flag
      FROM pv.silver.energy_readings
      WHERE reading_date BETWEEN current_date() - INTERVAL 14 DAYS AND current_date()
      UNION ALL
      SELECT weather_date AS date, location_id AS facility, quality_flag
      FROM pv.silver.weather
      WHERE weather_date BETWEEN current_date() - INTERVAL 14 DAYS AND current_date()
      UNION ALL
      SELECT aqi_date AS date, location_id AS facility, quality_flag
      FROM pv.silver.air_quality
      WHERE aqi_date BETWEEN current_date() - INTERVAL 14 DAYS AND current_date()
    )
    SELECT
      CAST(date AS STRING)                                                AS date,
      facility,
      SUM(CASE WHEN quality_flag = 'GOOD' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)                                                        AS score
    FROM all_data
    GROUP BY date, facility
    ORDER BY date ASC, facility ASC
    """
    results = execute_sql(query)
    return results

def get_daily_forecast(
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int = 1,
    exclude_bad_actuals: bool = True,
) -> list[dict]:
    # Filter by a single forecast_horizon to avoid summing D+1+D+3+D+5+D+7
    # for the same forecast_date (which would give 4x the real predicted value).
    _assert_valid_horizon(horizon)
    if start_date:
        _assert_valid_date(start_date, "start_date")
    if end_date:
        _assert_valid_date(end_date, "end_date")
    horizon_filter = f"AND f.forecast_horizon = {horizon}"

    if start_date and end_date:
        where_clause = f"WHERE f.forecast_date BETWEEN '{start_date}' AND '{end_date}'"
        energy_where_clause = f"WHERE reading_date BETWEEN '{start_date}' AND '{end_date}'"
        limit_clause = "LIMIT 60"
    elif start_date:
        where_clause = f"WHERE f.forecast_date >= '{start_date}'"
        energy_where_clause = f"WHERE reading_date >= '{start_date}'"
        limit_clause = "LIMIT 60"
    else:
        where_clause = "WHERE f.forecast_date >= current_date() - INTERVAL 14 DAYS"
        energy_where_clause = "WHERE reading_date >= current_date() - INTERVAL 14 DAYS"
        limit_clause = "LIMIT 28"

    quality_cte = ""
    join_clause = ""
    actual_expr = "SUM(f.actual_energy_mwh_daily)"
    predicted_expr = "SUM(f.predicted_energy_mwh_daily)"
    energy_eps = 0.001
    past_date_cond = "CAST(f.forecast_date AS DATE) <= current_date()"

    if exclude_bad_actuals:
        quality_cte = f"""
    WITH quality_bad AS (
        SELECT
          UPPER(facility_id) AS facility_id,
          CAST(reading_date AS DATE) AS forecast_date,
          MAX(CASE
                WHEN quality_flag = 'BAD'
                  OR COALESCE(quality_issues, '') LIKE '%DAYTIME_ZERO_ENERGY%'
                  OR COALESCE(quality_issues, '') LIKE '%EQUIPMENT_DOWNTIME%'
                THEN 1 ELSE 0 END) AS is_bad
        FROM pv.silver.energy_readings
        {energy_where_clause}
        GROUP BY UPPER(facility_id), CAST(reading_date AS DATE)
    )
    """
        join_clause = """
    LEFT JOIN quality_bad q
      ON q.facility_id = UPPER(f.facility_id)
     AND q.forecast_date = CAST(f.forecast_date AS DATE)
    """
        invalid_actual_cond = (
            "q.is_bad = 1 AND f.actual_energy_mwh_daily IS NOT NULL "
            f"AND f.actual_energy_mwh_daily <= {energy_eps}"
        )
        missing_actual_cond = "f.actual_energy_mwh_daily IS NULL"
        actual_expr = (
            "SUM(CASE WHEN " + invalid_actual_cond + " THEN NULL "
            "ELSE f.actual_energy_mwh_daily END)"
        )
        predicted_expr = (
            "SUM(CASE WHEN " + past_date_cond + " AND (" + missing_actual_cond
            + " OR " + invalid_actual_cond + ") THEN NULL "
            "ELSE f.predicted_energy_mwh_daily END)"
        )

    query = f"""
    {quality_cte}
    SELECT
      date,
      horizon,
      actual,
      predicted,
      predicted * 0.95 AS lower,
      predicted * 1.05 AS upper
    FROM (
        SELECT
          CAST(f.forecast_date AS STRING) AS date,
          {horizon} AS horizon,
          {actual_expr} AS actual,
          {predicted_expr} AS predicted
        FROM pv.gold.forecast_daily f
        {join_clause}
        {where_clause}
        {horizon_filter}
        GROUP BY f.forecast_date
        ORDER BY f.forecast_date DESC
        {limit_clause}
    )
    ORDER BY date ASC
    """
    return execute_sql(query)

def get_forecast_champion_meta(model_name: str) -> dict:
    """Return champion alias version and model label for a UC model name.

    Falls back to best-effort heuristics if system tables are unavailable.
    """
    if not re.match(r"^pv\.gold\.daily_forecast_d\d+$", model_name):
        raise ValueError(f"Unexpected model_name format: {model_name!r}")
    alias_row = {}
    try:
        alias_query = f"""
        SELECT
          model_name,
          alias,
          version
        FROM system.ml.model_aliases
        WHERE model_name = '{model_name}' AND alias = 'champion'
        LIMIT 1
        """
        alias_rows = execute_sql(alias_query)
        if alias_rows:
            alias_row = alias_rows[0]
    except Exception:
        alias_row = {}

    version = alias_row.get("version") if alias_row else None
    model_label = "Stacking"

    return {
        "model_name": model_name,
        "model_alias": alias_row.get("alias") if alias_row else None,
        "model_version": version,
        "model_label": model_label,
    }

def _get_mlflow_latest_run_payload(model_name: str, *, champion_only: bool = False) -> dict:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        logger.warning("MLflow import failed for %s: %s", model_name, exc)
        return {}

    settings = get_solar_chat_settings()
    if settings.databricks_host:
        os.environ.setdefault("DATABRICKS_HOST", settings.databricks_host)
    if settings.databricks_token:
        os.environ.setdefault("DATABRICKS_TOKEN", settings.databricks_token)

    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        client = MlflowClient()
    except Exception as exc:
        logger.warning("MlflowClient init failed for %s: %s", model_name, exc)
        return {}

    # If champion_only, resolve the champion alias first
    target_version_obj = None
    if champion_only:
        try:
            target_version_obj = client.get_model_version_by_alias(model_name, "champion")
        except Exception as exc:
            logger.warning("get_model_version_by_alias(%s, champion) failed: %s", model_name, exc)
            target_version_obj = None

    if target_version_obj:
        version_obj = target_version_obj
    else:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:
            logger.warning("search_model_versions(%s) failed: %s", model_name, exc)
            return {}
        if not versions:
            logger.warning("No versions found for %s", model_name)
            return {}
        version_obj = max(versions, key=lambda v: int(v.version))

    created = version_obj.creation_timestamp
    created_date = None
    if created:
        created_date = datetime.utcfromtimestamp(created / 1000).date().isoformat()

    try:
        run = client.get_run(version_obj.run_id)
    except Exception as exc:
        logger.warning("get_run(%s) failed: %s", version_obj.run_id, exc)
        return {
            "model_version": version_obj.version,
            "run_id": version_obj.run_id,
            "date": created_date,
        }

    metrics = run.data.metrics or {}
    logger.info(
        "MLflow run payload for %s v%s: %d metrics, keys=%s",
        model_name, version_obj.version, len(metrics), list(metrics.keys())[:10],
    )
    return {
        "model_version": version_obj.version,
        "run_id": version_obj.run_id,
        "date": created_date,
        "metrics": metrics,
        "params": run.data.params or {},
    }

def _format_forecast_kpis(
    metrics: dict,
    metric_keys: dict,
    eval_date: str,
    meta: dict,
) -> dict:
    def pick_metric(keys: list[str]) -> str | float:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]
        return "N/A"

    return {
        "rmse": pick_metric(metric_keys["rmse"]),
        "mae": pick_metric(metric_keys["mae"]),
        "r2": pick_metric(metric_keys["r2"]),
        "mape": pick_metric(metric_keys["mape"]),
        "skill_score": pick_metric(metric_keys["skill_score"]),
        "date": eval_date,
        **meta,
    }

def _merge_missing_kpis(base: dict, fallback: dict) -> dict:
    merged = dict(base)
    for key in ("rmse", "mae", "r2", "mape", "skill_score", "date"):
        val = merged.get(key)
        if val in (None, "N/A") and fallback.get(key) is not None:
            merged[key] = fallback.get(key)
    return merged

def _get_mlflow_horizon_kpis(metrics: dict, horizon: int, eval_date: str) -> dict:
    stacking_prefix = f"stacking__d_{horizon}"
    persistence_prefix = f"persistence__d_{horizon}"

    r2 = metrics.get(f"{stacking_prefix}_r2")
    nrmse = metrics.get(f"{stacking_prefix}_nrmse_pct")
    nmae = metrics.get(f"{stacking_prefix}_nmae_pct")
    baseline_nrmse = metrics.get(f"{persistence_prefix}_nrmse_pct")

    skill_score = None
    if isinstance(nrmse, (int, float)) and isinstance(baseline_nrmse, (int, float)):
        if baseline_nrmse > 0:
            # Match daily_utils._compute_skill_score: 1 - (candidate / reference)
            skill_score = 1.0 - (float(nrmse) / float(baseline_nrmse))

    mape = metrics.get(f"{stacking_prefix}_mape_pct")
    rmse_mwh = metrics.get(f"{stacking_prefix}_rmse_mwh")
    mae_mwh = metrics.get(f"{stacking_prefix}_mae_mwh")

    return {
        "rmse": nrmse or rmse_mwh,
        "mae": nmae or mae_mwh,
        "r2": r2,
        "mape": mape,
        "skill_score": skill_score,
        "date": eval_date,
    }


def get_forecast_registry_kpis(model_name: str, horizon: int) -> dict:
    """Fetch KPI metrics from MLflow model registry for the champion version."""
    meta = get_forecast_champion_meta(model_name)
    version = meta.get("model_version")

    metric_keys = {
        "r2": [f"stacking__d_{horizon}_r2", "r2", "test_r2"],
        "rmse": [f"stacking__d_{horizon}_nrmse_pct", "nrmse_pct", "rmse_mwh", "rmse"],
        "mae": [f"stacking__d_{horizon}_nmae_pct", "nmae_pct", "mae_mwh", "mae"],
        "mape": [f"stacking__d_{horizon}_mape_pct", "mape_pct", "mape"],
        "skill_score": [f"stacking__d_{horizon}_skill", "skill_score", "skill"],
    }

    rows = []
    if version:
        try:
            flat_keys = [k for keys in metric_keys.values() for k in keys]
            key_list = ", ".join([f"'{k}'" for k in dict.fromkeys(flat_keys)])
            metrics_query = f"""
            SELECT
              key,
              value
            FROM system.ml.model_version_metrics
            WHERE model_name = '{model_name}'
              AND version = '{version}'
              AND key IN ({key_list})
            """
            rows = execute_sql(metrics_query)
        except Exception:
            rows = []

    metrics = {row.get("key"): row.get("value") for row in rows if row.get("key")}
    eval_date = "N/A"
    if version:
        try:
            date_query = f"""
            SELECT CAST(creation_time AS DATE) AS created
            FROM system.ml.model_versions
            WHERE model_name = '{model_name}' AND version = '{version}'
            LIMIT 1
            """
            date_rows = execute_sql(date_query)
            if date_rows and date_rows[0].get("created"):
                eval_date = str(date_rows[0]["created"])
        except Exception:
            eval_date = "N/A"

    result = _format_forecast_kpis(metrics, metric_keys, eval_date, meta)
    has_registry_values = any(
        result[k] != "N/A" for k in ("rmse", "mae", "r2", "mape", "skill_score")
    )
    mlflow_payload = _get_mlflow_latest_run_payload(model_name)
    if mlflow_payload:
        if not version and mlflow_payload.get("model_version"):
            meta = {**meta, "model_version": mlflow_payload.get("model_version")}
        mlflow_metrics = mlflow_payload.get("metrics") or {}
        mlflow_date = mlflow_payload.get("date") or eval_date
        mlflow_kpis = _get_mlflow_horizon_kpis(mlflow_metrics, horizon, mlflow_date)
        result = _merge_missing_kpis(result, mlflow_kpis)
        if not has_registry_values:
            return result
        return result

    if has_registry_values:
        return result

    return result

def _get_registry_models_sql_fallback(
    model_names: list[str], horizon_map: dict[str, int]
) -> list[dict]:
    """SQL-only fallback when MLflow client is unavailable.

    Gets latest champion version from forecast_daily and computes metrics.
    """
    name_list = ", ".join(f"'{n}'" for n in model_names)
    query = f"""
    WITH latest AS (
        SELECT model_name, MAX(CAST(model_version AS INT)) AS max_ver
        FROM pv.gold.forecast_daily
        WHERE model_name IN ({name_list})
          AND actual_energy_mwh_daily IS NOT NULL
        GROUP BY model_name
    ),
    eval_data AS (
        SELECT f.model_name, CAST(f.model_version AS INT) AS model_version,
               f.forecast_horizon AS horizon_days,
               f.predicted_energy_mwh_daily AS pred,
               f.actual_energy_mwh_daily AS actual
        FROM pv.gold.forecast_daily f
        JOIN latest l ON f.model_name = l.model_name
                     AND CAST(f.model_version AS INT) = l.max_ver
        WHERE f.actual_energy_mwh_daily IS NOT NULL
          AND f.forecast_date >= CURRENT_DATE - 30
    ),
    stats AS (
        SELECT model_name, model_version, horizon_days,
               AVG(actual) AS avg_actual,
               SQRT(AVG(POWER(actual - pred, 2))) AS rmse,
               AVG(ABS(actual - pred)) AS mae,
               COUNT(*) AS sample_count
        FROM eval_data
        GROUP BY model_name, model_version, horizon_days
    ),
    sse_sst AS (
        SELECT e.model_name, e.model_version, e.horizon_days,
               SUM(POWER(e.actual - e.pred, 2)) AS sse,
               SUM(POWER(e.actual - s.avg_actual, 2)) AS sst
        FROM eval_data e
        JOIN stats s ON e.model_name = s.model_name
                    AND e.model_version = s.model_version
                    AND e.horizon_days = s.horizon_days
        GROUP BY e.model_name, e.model_version, e.horizon_days
    )
    SELECT s.model_name, s.model_version, s.horizon_days,
           ROUND(1 - ss.sse / NULLIF(ss.sst, 0), 4) AS r2,
           ROUND(s.rmse, 2) AS rmse_mwh,
           ROUND(s.mae, 2) AS mae_mwh,
           ROUND(100.0 * s.mae / NULLIF(s.avg_actual, 0), 2) AS mape,
           ROUND(100.0 * s.rmse / NULLIF(s.avg_actual, 0), 2) AS nrmse_pct,
           s.sample_count
    FROM stats s
    JOIN sse_sst ss ON s.model_name = ss.model_name
                   AND s.model_version = ss.model_version
                   AND s.horizon_days = ss.horizon_days
    ORDER BY s.horizon_days
    """
    try:
        rows = execute_sql(query)
    except Exception as exc:
        logger.error("SQL fallback for registry models failed: %s", exc)
        return []

    results = []
    for row in rows:
        name = row.get("model_name", "")
        results.append({
            "model_name": name,
            "version": str(row.get("model_version", "?")),
            "approach": "Stacking",
            "algorithm": "Ensemble",
            "rmse": row.get("nrmse_pct"),
            "mae": row.get("mae_mwh"),
            "r2": row.get("r2"),
            "mape": row.get("mape"),
            "skill_score": None,
            "created": None,
            "champion": True,
        })
    return results


def get_registry_models() -> list[dict]:
    """Fetch champion model info: same source as Model Registry page.

    Uses _get_mlflow_latest_run_payload (MLflow run metrics) + _get_mlflow_horizon_kpis
    to extract R²/RMSE/MAE/MAPE from the training run, identical to the page.
    """
    model_names = [
        "pv.gold.daily_forecast_d1",
        "pv.gold.daily_forecast_d3",
        "pv.gold.daily_forecast_d5",
        "pv.gold.daily_forecast_d7",
    ]
    horizon_map = {
        "pv.gold.daily_forecast_d1": 1,
        "pv.gold.daily_forecast_d3": 3,
        "pv.gold.daily_forecast_d5": 5,
        "pv.gold.daily_forecast_d7": 7,
    }

    results = []
    for name in model_names:
        try:
            payload = _get_mlflow_latest_run_payload(name, champion_only=True)
        except Exception as exc:
            logger.error("get_registry_models MLflow failed for %s: %s", name, exc)
            payload = {}

        if not payload:
            logger.warning("get_registry_models: empty payload for %s", name)
            continue

        horizon = horizon_map.get(name, 1)
        metrics = payload.get("metrics", {})
        created = payload.get("date")
        kpis = _get_mlflow_horizon_kpis(metrics, horizon, created or "N/A")

        results.append({
            "model_name": name,
            "version": payload.get("model_version", "?"),
            "approach": "Stacking",
            "algorithm": "Ensemble",
            "rmse": kpis.get("rmse"),
            "mae": kpis.get("mae"),
            "r2": kpis.get("r2"),
            "mape": kpis.get("mape"),
            "skill_score": kpis.get("skill_score"),
            "created": created,
            "champion": True,
        })

    if not results:
        logger.warning("get_registry_models: MLflow returned nothing, falling back to SQL")
        results = _get_registry_models_sql_fallback(model_names, horizon_map)

    return results
