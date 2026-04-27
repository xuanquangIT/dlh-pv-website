from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import get_model_evaluation_metrics, get_model_monitoring_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-training", tags=["ML Training"])

_HORIZON_MODEL = {
    1: "pv.gold.daily_forecast_d1",
    3: "pv.gold.daily_forecast_d3",
    5: "pv.gold.daily_forecast_d5",
    7: "pv.gold.daily_forecast_d7",
}

@router.get("/monitoring", response_model=List[Dict[str, Any]])
def get_ml_monitoring(
    horizon: Optional[int] = Query(None, description="Horizon (1, 3, 5, 7). Omit for all."),
    _: object = Depends(require_role(["admin", "ml_engineer", "data_engineer"]))
):
    # Uses model_evaluation_weekly: 30-day rolling window (240 pts) → reliable R²
    try:
        return get_model_evaluation_metrics(horizon=horizon)
    except Exception as e:
        logger.error(f"Error fetching ML evaluation data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ML evaluation data")

@router.get("/monitoring-daily", response_model=List[Dict[str, Any]])
def get_ml_monitoring_daily(
    horizon: Optional[int] = Query(None, description="Horizon (1, 3, 5, 7). Omit for all."),
    _: object = Depends(require_role(["admin", "ml_engineer", "data_engineer"]))
):
    # Raw daily monitoring (8 pts/day → noisy R², use for trend only)
    try:
        model_name = _HORIZON_MODEL.get(horizon) if horizon else None
        return get_model_monitoring_metrics(model_name=model_name)
    except Exception as e:
        logger.error(f"Error fetching ML monitoring data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ML monitoring data")
