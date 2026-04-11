from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import require_role
from app.services.databricks_service import get_job_info, get_job_runs, trigger_job_run, get_run, cancel_job_run

router = APIRouter(prefix="/data-pipeline", tags=["Data Pipeline"])

# Default Job ID based on the Databricks setup JSON
DEFAULT_JOB_ID = 28718564084384

@router.get("/status")
def get_data_pipeline_status(
    _: object = Depends(require_role(["data_engineer", "system"])),
) -> dict[str, str]:
    return {
        "module": "data_pipeline",
        "message": "Data Pipeline API placeholder is ready.",
    }


@router.get("/jobs")
def get_jobs(
    job_id: int = DEFAULT_JOB_ID,
    _: object = Depends(require_role(["data_engineer", "admin"])),
):
    try:
        return get_job_info(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/runs")
def get_runs(
    job_id: int = DEFAULT_JOB_ID,
    limit: int = 5,
    _: object = Depends(require_role(["data_engineer", "admin"])),
):
    try:
        return get_job_runs(job_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/run")
def run_job(
    job_id: int = DEFAULT_JOB_ID,
    _: object = Depends(require_role(["data_engineer", "admin"])),
):
    try:
        return trigger_job_run(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/runs/{run_id}")
def get_run_details(
    run_id: int,
    _: object = Depends(require_role(["data_engineer", "admin"])),
):
    try:
        return get_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/runs/{run_id}/cancel")
def cancel_run(
    run_id: int,
    _: object = Depends(require_role(["data_engineer", "admin"])),
):
    try:
        return cancel_job_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
