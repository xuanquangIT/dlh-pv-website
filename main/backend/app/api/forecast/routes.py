from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import get_daily_forecast, get_model_monitoring_metrics

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/summary-kpi", response_model=Dict[str, Any])
def get_forecast_summary_kpi(_: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))):
    try:
        # Fetch the latest monitoring metrics to use as overall KPIs
        metrics = get_model_monitoring_metrics()
        if not metrics:
            return {"rmse": "N/A", "mae": "N/A", "r2": "N/A", "mape": "N/A", "skill_score": "N/A", "date": "N/A"}
        latest = metrics[-1] # The results are ordered by date ASC, so the last is latest
        return latest
    except Exception as e:
        logger.error(f"Error fetching forecast KPI summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast KPI summary")

@router.get("/daily", response_model=List[Dict[str, Any]])
def get_forecast_daily(_: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))):
    try:
        return get_daily_forecast()
    except Exception as e:
        logger.error(f"Error fetching daily forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch daily forecast")
