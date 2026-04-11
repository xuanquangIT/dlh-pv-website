from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from app.api.dependencies import require_role
from app.services.databricks_service import (
    get_quality_summary_metrics,
    get_facility_quality_scores,
    get_recent_quality_issues,
    get_facility_heatmap_data
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])

@router.get("/summary", response_model=Dict[str, Any])
def get_summary(_: object = Depends(require_role(["admin", "data_engineer"]))):
    try:
        return get_quality_summary_metrics()
    except Exception as e:
        logger.error(f"Error fetching quality summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quality summary")

@router.get("/facility-scores", response_model=List[Dict[str, Any]])
def get_facility_scores(_: object = Depends(require_role(["admin", "data_engineer"]))):
    try:
        return get_facility_quality_scores()
    except Exception as e:
        logger.error(f"Error fetching facility scores: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch facility scores")

@router.get("/recent-issues", response_model=List[Dict[str, Any]])
def get_recent_issues(_: object = Depends(require_role(["admin", "data_engineer"]))):
    try:
        return get_recent_quality_issues()
    except Exception as e:
        logger.error(f"Error fetching recent issues: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent issues")

@router.get("/heatmap-data", response_model=List[Dict[str, Any]])
def get_heatmap_data(_: object = Depends(require_role(["admin", "data_engineer"]))):
    try:
        return get_facility_heatmap_data()
    except Exception as e:
        logger.error(f"Error fetching heatmap data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch heatmap data")
