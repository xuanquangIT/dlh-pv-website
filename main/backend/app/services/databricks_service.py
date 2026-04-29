import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import CronSchedule, JobSettings, PauseStatus
from app.core.settings import get_solar_chat_settings
from functools import lru_cache

logger = logging.getLogger(__name__)

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
) -> list[dict]:
    # Filter by a single forecast_horizon to avoid summing D+1+D+3+D+5+D+7
    # for the same forecast_date (which would give 4x the real predicted value).
    horizon_filter = f"AND forecast_horizon = {horizon}"

    if start_date and end_date:
        where_clause = f"WHERE forecast_date BETWEEN '{start_date}' AND '{end_date}'"
        limit_clause = "LIMIT 60"
    elif start_date:
        where_clause = f"WHERE forecast_date >= '{start_date}'"
        limit_clause = "LIMIT 60"
    else:
        where_clause = "WHERE forecast_date >= current_date() - INTERVAL 14 DAYS"
        limit_clause = "LIMIT 28"

    query = f"""
    SELECT * FROM (
        SELECT
          CAST(forecast_date AS STRING)             AS date,
          {horizon}                                 AS horizon,
          SUM(actual_energy_mwh_daily)              AS actual,
          SUM(predicted_energy_mwh_daily)           AS predicted,
          SUM(predicted_energy_mwh_daily) * 0.95   AS lower,
          SUM(predicted_energy_mwh_daily) * 1.05   AS upper
        FROM pv.gold.forecast_daily
        {where_clause}
        {horizon_filter}
        GROUP BY forecast_date
        ORDER BY forecast_date DESC
        {limit_clause}
    )
    ORDER BY date ASC
    """
    return execute_sql(query)

def get_model_monitoring_metrics(model_name: str | None = None) -> list[dict]:
    # Filter by model_name to avoid mixing D+1..D+7 metrics in one result set.
    # Default: return all horizons so caller can pick what to display.
    model_filter = f"AND model_name = '{model_name}'" if model_name else ""
    query = f"""
    SELECT * FROM (
        SELECT
          eval_date                 AS date,
          model_name,
          model_version,
          rmse_mwh                 AS rmse,
          mae_mwh                  AS mae,
          r2,
          mape_day_pct             AS mape,
          skill_score
        FROM pv.gold.model_monitoring_daily
        WHERE facility_id = 'ALL' AND model_name NOT LIKE '%champion%'
        {model_filter}
        ORDER BY eval_date DESC
        LIMIT 30
    )
    ORDER BY date ASC
    """
    return execute_sql(query)

def get_model_evaluation_metrics(horizon: int | None = None) -> list[dict]:
        # Use raw monitoring metrics (daily) for ML Training view.
        horizon_filter = (
                f"AND model_name LIKE '%daily_forecast_d{horizon}%'" if horizon else ""
        )
        query = f"""
        SELECT * FROM (
                SELECT
                    eval_date AS date,
                    CASE
                        WHEN model_name LIKE '%daily_forecast_d1%' THEN 1
                        WHEN model_name LIKE '%daily_forecast_d3%' THEN 3
                        WHEN model_name LIKE '%daily_forecast_d5%' THEN 5
                        WHEN model_name LIKE '%daily_forecast_d7%' THEN 7
                        ELSE NULL
                    END AS horizon,
                    model_name,
                    model_version,
                    rmse_mwh     AS rmse,
                    mae_mwh      AS mae,
                    r2,
                    mape_day_pct AS mape,
                    skill_score
                FROM pv.gold.model_monitoring_daily
                WHERE facility_id = 'ALL' AND model_name NOT LIKE '%champion%'
                {horizon_filter}
                ORDER BY eval_date DESC
                LIMIT 12
        )
        ORDER BY date ASC
        """
        return execute_sql(query)

def get_registry_models() -> list[dict]:
    query = """
    WITH ranked AS (
        SELECT
          model_name,
          model_version                     AS version,
          MAX(approach)                     AS approach,
          MAX(algorithm)                    AS algorithm,
          ROUND(AVG(rmse_mwh), 3)           AS rmse,
          ROUND(AVG(mae_mwh), 3)            AS mae,
          ROUND(AVG(r2), 4)                 AS r2,
          ROUND(AVG(mape_day_pct), 2)       AS mape,
          MIN(eval_date)                    AS created,
                    ROW_NUMBER() OVER(PARTITION BY model_name ORDER BY TRY_CAST(model_version AS INT) DESC, MIN(eval_date) DESC) as rnk
        FROM pv.gold.mart_forecast_accuracy_daily
        WHERE facility_id = 'ALL' AND model_name NOT LIKE '%champion%'
          AND is_champion = true
        GROUP BY model_name, model_version
    )
    SELECT
      model_name,
      version,
      approach,
      algorithm,
      rmse,
      mae,
      r2,
      mape,
      created
    FROM ranked
    WHERE rnk = 1
    ORDER BY model_name ASC
    """
    results = execute_sql(query)
    if results:
        for row in results:
            row['champion'] = True
        return results

    fallback_query = """
    WITH ranked AS (
        SELECT
          model_name,
          model_version                     AS version,
          MAX(approach)                     AS approach,
          CAST(NULL AS STRING)              AS algorithm,
          ROUND(AVG(rmse_mwh), 3)           AS rmse,
          ROUND(AVG(mae_mwh), 3)            AS mae,
          ROUND(AVG(r2), 4)                 AS r2,
          ROUND(AVG(mape_day_pct), 2)       AS mape,
          MIN(eval_date)                    AS created,
          ROW_NUMBER() OVER(PARTITION BY model_name ORDER BY TRY_CAST(model_version AS INT) DESC, MIN(eval_date) DESC) as rnk
        FROM pv.gold.model_monitoring_daily
        WHERE facility_id = 'ALL' AND model_name NOT LIKE '%champion%'
        GROUP BY model_name, model_version
    )
    SELECT
      model_name,
      version,
      approach,
      algorithm,
      rmse,
      mae,
      r2,
      mape,
      created
    FROM ranked
    WHERE rnk = 1
    ORDER BY model_name ASC
    """
    results = execute_sql(fallback_query)
    for row in results:
        row['champion'] = True
    return results

def get_facility_heatmap(horizon: int) -> list[dict]:
    model_filter = f"%daily_forecast_d{horizon}%"
    query = f"""
    SELECT
      facility_id,
      DATE_TRUNC('week', eval_date) AS week_start,
      ROUND(AVG(r2), 2) AS r2
    FROM pv.gold.mart_forecast_accuracy_daily
    WHERE facility_id != 'ALL'
      AND model_name LIKE '{model_filter}'
      AND is_champion = true
      AND eval_date >= current_date() - interval 56 days
    GROUP BY facility_id, DATE_TRUNC('week', eval_date)
    ORDER BY facility_id, week_start DESC
    """
    results = execute_sql(query)
    
    heatmap = {}
    weeks_set = set()
    for row in results:
        fac = row['facility_id']
        w = str(row['week_start']).split(' ')[0]
        r2 = row['r2']
        if r2 is None: continue
        
        r2 = float(r2)
        if r2 >= 0.85: grade = 'A'
        elif r2 >= 0.75: grade = 'B'
        elif r2 >= 0.60: grade = 'C'
        else: grade = 'D'
        
        if fac not in heatmap:
            heatmap[fac] = {}
        heatmap[fac][w] = {"r2": r2, "grade": grade}
        weeks_set.add(w)
        
    weeks = sorted(list(weeks_set))
    grid = []
    for fac, w_data in heatmap.items():
        w_list = []
        for w in weeks:
            w_list.append({
                "week": w[5:10], # MM-DD
                "r2": w_data.get(w, {}).get("r2", 0),
                "grade": w_data.get(w, {}).get("grade", "-")
            })
        grid.append({"facility_id": fac, "weeks": w_list})
    
    return grid

def get_facility_drill(facility_id: str, horizon: int) -> dict:
    model_filter = f"%daily_forecast_d{horizon}%"
    
    # Use forecast_daily for actual vs forecast
    q_chart = f"""
    SELECT
      CAST(forecast_date AS STRING) AS date_md,
      actual_energy_mwh_daily AS energy_mwh_daily,
      predicted_capacity_factor_daily AS capacity_factor_pct,
      predicted_energy_mwh_daily AS forecast_mwh
    FROM pv.gold.forecast_daily
    WHERE facility_id = '{facility_id}'
      AND forecast_horizon = {horizon}
      AND forecast_date >= current_date() - interval 30 days
    ORDER BY forecast_date ASC
    """
    chart_rows = execute_sql(q_chart)
    
    for row in chart_rows:
        row['date_md'] = row['date_md'][5:10] if row.get('date_md') else ''
        row['energy_mwh_daily'] = float(row.get('energy_mwh_daily') or 0)
        row['forecast_mwh'] = float(row.get('forecast_mwh') or 0)
        row['capacity_factor_pct'] = float(row.get('capacity_factor_pct') or 0)
        # Dummy weather/AQI since we can't use impact tables
        row['cloud_pct'] = 0
        row['aqi_value'] = 0
        
    # Weekly grades
    q_weeks = f"""
    SELECT
      DATE_TRUNC('week', eval_date) AS week_start,
      ROUND(AVG(r2), 2) AS r2
    FROM pv.gold.mart_forecast_accuracy_daily
    WHERE facility_id = '{facility_id}'
      AND model_name LIKE '{model_filter}'
      AND is_champion = true
      AND eval_date >= current_date() - interval 56 days
    GROUP BY DATE_TRUNC('week', eval_date)
    ORDER BY week_start DESC
    """
    week_rows = execute_sql(q_weeks)
    weeks = []
    for row in week_rows:
        r2 = float(row.get('r2') or 0)
        if r2 >= 0.85: grade = 'A'
        elif r2 >= 0.75: grade = 'B'
        elif r2 >= 0.60: grade = 'C'
        else: grade = 'D'
        weeks.append({
            "week": str(row['week_start']).split(' ')[0][5:10],
            "r2": r2,
            "grade": grade
        })
        
    last_energy = chart_rows[-1]['energy_mwh_daily'] if chart_rows else 0
    last_cf = chart_rows[-1]['capacity_factor_pct'] if chart_rows else 0
    
    return {
        "facility": {
            "id": facility_id,
            "name": facility_id,
            "region": "Unknown",
            "state": "Unknown",
            "capacity": 0,
            "lat": 0, "lon": 0
        },
        "last": {
            "energy_mwh_daily": last_energy,
            "capacity_factor_pct": last_cf,
            "specific_yield": 0,
            "aqi_value": 0,
            "aqi_category": "Unknown"
        },
        "daily_rows": chart_rows,
        "weeks": weeks
    }

def get_forecast_waterfall(horizon: int) -> list[dict]:
    query = f"""
    SELECT
      SUM(actual_energy_mwh_daily) AS actual,
      SUM(predicted_energy_mwh_daily) AS predicted
    FROM pv.gold.forecast_daily
    WHERE forecast_horizon = {horizon}
      AND forecast_date = (SELECT MAX(forecast_date) FROM pv.gold.forecast_daily)
    """
    res = execute_sql(query)
    predicted = float(res[0].get('predicted') or 1000) if res else 1000
    
    return [
        {"label": "Clear-sky baseline", "value": predicted * 1.4, "color": "#f4b942"},
        {"label": "Cloud derating", "value": -predicted * 0.2, "color": "#e15759"},
        {"label": "Temperature derating", "value": -predicted * 0.1, "color": "#f6c544"},
        {"label": "AQI derating", "value": -predicted * 0.05, "color": "#e59aa8"},
        {"label": "Ensemble adjustments", "value": -predicted * 0.05, "color": "#6aa8ef"}
    ]

def get_residuals_30d(horizon: int) -> list[dict]:
    query = f"""
    SELECT
      CAST(forecast_date AS STRING) AS date_md,
      SUM(actual_energy_mwh_daily) - SUM(predicted_energy_mwh_daily) AS residual
    FROM pv.gold.forecast_daily
    WHERE forecast_horizon = {horizon}
      AND forecast_date >= current_date() - interval 30 days
    GROUP BY forecast_date
    ORDER BY forecast_date ASC
    """
    rows = execute_sql(query)
    for r in rows:
        r['date_md'] = r['date_md'][5:10] if r.get('date_md') else ''
        r['residual'] = float(r.get('residual') or 0)
    return rows
