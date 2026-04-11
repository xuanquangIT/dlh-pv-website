from databricks.sdk import WorkspaceClient
from app.core.settings import get_solar_chat_settings

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
    query = """
    WITH all_data AS (
      SELECT quality_flag FROM pv.silver.energy_readings
      UNION ALL
      SELECT quality_flag FROM pv.silver.weather
      UNION ALL
      SELECT quality_flag FROM pv.silver.air_quality
      UNION ALL
      SELECT quality_flag FROM pv.silver.facility_status
    )
    SELECT 
      SUM(CASE WHEN quality_flag = 'GOOD' THEN 1 ELSE 0 END) as valid,
      SUM(CASE WHEN quality_flag = 'WARNING' THEN 1 ELSE 0 END) as warning,
      SUM(CASE WHEN quality_flag = 'BAD' THEN 1 ELSE 0 END) as invalid
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
        valid_ratio = f"{(valid / total) * 100:.1f}%"
        warning_ratio = f"{(warning / total) * 100:.1f}%"
        invalid_ratio = f"{(invalid / total) * 100:.1f}%"
    else:
        valid_ratio = warning_ratio = invalid_ratio = "0.0%"
        
    return {
        "valid": f"{valid:,}",
        "warning": f"{warning:,}",
        "invalid": f"{invalid:,}",
        "valid_ratio": f"{valid_ratio} of total",
        "warning_ratio": f"{warning_ratio} of total",
        "invalid_ratio": f"{invalid_ratio} of total"
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
    query = """
    SELECT 
      date_format(check_timestamp, 'HH:mm z') as time,
      table_name as facility,
      SPLIT_PART(rule_name, '_', 1) as sensor,
      rule_name as issue,
      CONCAT(failed_rows, ' records') as affected,
      CASE WHEN status = 'FAIL' THEN 'Error' ELSE 'Warning' END as severity,
      'Flagged' as action
    FROM pv.silver.data_quality_log
    WHERE status != 'PASS'
    ORDER BY check_timestamp DESC
    LIMIT 10
    """
    results = execute_sql(query)
    issues = []
    for row in results:
        issues.append({
            "time": row.get('time', ''),
            "facility": row.get('facility', ''),
            "sensor": row.get('sensor', ''),
            "issue": row.get('issue', ''),
            "affected": row.get('affected', ''),
            "severity": row.get('severity', ''),
            "action": row.get('action', '')
        })
    return issues

def get_facility_heatmap_data() -> list[dict]:
    query = """
    WITH all_data AS (
      SELECT reading_date as date, facility_id as facility, quality_flag 
      FROM pv.silver.energy_readings
      WHERE reading_date > current_date() - INTERVAL 14 DAYS
      UNION ALL
      SELECT weather_date as date, location_id as facility, quality_flag 
      FROM pv.silver.weather
      WHERE weather_date > current_date() - INTERVAL 14 DAYS
      UNION ALL
      SELECT aqi_date as date, location_id as facility, quality_flag 
      FROM pv.silver.air_quality
      WHERE aqi_date > current_date() - INTERVAL 14 DAYS
      UNION ALL
      SELECT CAST(effective_from AS DATE) as date, facility_id as facility, quality_flag 
      FROM pv.silver.facility_status
      WHERE CAST(effective_from AS DATE) > current_date() - INTERVAL 14 DAYS
    )
    SELECT 
      CAST(date AS STRING) as date,
      facility,
      SUM(CASE WHEN quality_flag = 'GOOD' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as score
    FROM all_data
    WHERE date <= (SELECT MAX(reading_date) FROM pv.silver.energy_readings)
    GROUP BY date, facility
    ORDER BY date ASC, facility ASC
    """
    results = execute_sql(query)
    return results

