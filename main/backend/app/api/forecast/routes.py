from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import get_daily_forecast, get_model_monitoring_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecast"])

_HORIZON_MODEL = {
    1: "pv.gold.daily_forecast_d1",
    3: "pv.gold.daily_forecast_d3",
    5: "pv.gold.daily_forecast_d5",
    7: "pv.gold.daily_forecast_d7",
}

@router.get("/summary-kpi", response_model=Dict[str, Any])
def get_forecast_summary_kpi(
    horizon: int = Query(1, description="Forecast horizon in days (1, 3, 5, 7)"),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    try:
        model_name = _HORIZON_MODEL.get(horizon, "pv.gold.daily_forecast_d1")
        metrics = get_model_monitoring_metrics(model_name=model_name)
        if not metrics:
            return {"rmse": "N/A", "mae": "N/A", "r2": "N/A", "mape": "N/A", "skill_score": "N/A", "date": "N/A", "horizon": horizon}
        latest = metrics[-1]
        latest["horizon"] = horizon
        return latest
    except Exception as e:
        logger.error(f"Error fetching forecast KPI summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast KPI summary")

@router.get("/daily", response_model=List[Dict[str, Any]])
def get_forecast_daily(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    horizon: int = Query(1, description="Forecast horizon in days (1, 3, 5, 7)"),
    exclude_bad_actuals: bool = Query(
        True,
        description="Exclude facility-days flagged as bad (or outage) in Silver quality signals",
    ),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    try:
        return get_daily_forecast(
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            exclude_bad_actuals=exclude_bad_actuals,
        )
    except Exception as e:
        logger.error(f"Error fetching daily forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch daily forecast")

@router.get("/monitoring", response_model=List[Dict[str, Any]])
def get_forecast_monitoring(
    horizon: Optional[int] = Query(None, description="Filter by horizon (1, 3, 5, 7). Omit for all horizons."),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    try:
        model_name = _HORIZON_MODEL.get(horizon) if horizon else None
        return get_model_monitoring_metrics(model_name=model_name)
    except Exception as e:
        logger.error(f"Error fetching forecast monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast monitoring")

@router.get("/facility-heatmap", response_model=Dict[str, Any])
def get_facility_heatmap_route(
    horizon: Optional[int] = Query(1, description="Forecast horizon in days"),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    from app.services.databricks_service import get_facility_heatmap
    try:
        grid = get_facility_heatmap(horizon)
        return {"grid": grid}
    except Exception as e:
        logger.error(f"Error fetching heatmap: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch heatmap")

@router.get("/facility-drill/{facility_id}", response_model=Dict[str, Any])
def get_facility_drill_route(
    facility_id: str,
    horizon: Optional[int] = Query(1, description="Forecast horizon in days"),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    from app.services.databricks_service import get_facility_drill
    try:
        data = get_facility_drill(facility_id, horizon)
        return data
    except Exception as e:
        logger.error(f"Error fetching facility drill: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch facility drill")
        raise HTTPException(status_code=500, detail="Failed to fetch facility drill")
