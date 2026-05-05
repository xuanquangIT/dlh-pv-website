import re
from datetime import date as _date

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import (
    get_daily_forecast,
    get_forecast_registry_kpis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecast"])

_HORIZON_MODEL = {
    1: "pv.gold.daily_forecast_d1",
    3: "pv.gold.daily_forecast_d3",
    5: "pv.gold.daily_forecast_d5",
    7: "pv.gold.daily_forecast_d7",
}

_ALLOWED_HORIZONS = frozenset(_HORIZON_MODEL.keys())

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_horizon(horizon: int) -> int:
    if horizon not in _ALLOWED_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid horizon {horizon}. Must be one of {sorted(_ALLOWED_HORIZONS)}.",
        )
    return horizon


def _validate_date(value: str | None, param_name: str) -> str | None:
    if value is None:
        return None
    if not _DATE_RE.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {param_name}: must be YYYY-MM-DD.",
        )
    try:
        _date.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {param_name}: not a real calendar date.",
        )
    return value


@router.get("/summary-kpi", response_model=Dict[str, Any])
def get_forecast_summary_kpi(
    horizon: int = Query(1, description="Forecast horizon in days (1, 3, 5, 7)"),
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"]))
):
    horizon = _validate_horizon(horizon)
    try:
        model_name = _HORIZON_MODEL[horizon]
        metrics = get_forecast_registry_kpis(model_name=model_name, horizon=horizon)
        metrics["horizon"] = horizon
        return metrics
    except HTTPException:
        raise
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
    horizon = _validate_horizon(horizon)
    start_date = _validate_date(start_date, "start_date")
    end_date = _validate_date(end_date, "end_date")
    try:
        return get_daily_forecast(
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            exclude_bad_actuals=exclude_bad_actuals,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching daily forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch daily forecast")
