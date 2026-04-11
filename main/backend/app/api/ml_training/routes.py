from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import get_model_monitoring_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-training", tags=["ML Training"])

@router.get("/monitoring", response_model=List[Dict[str, Any]])
def get_ml_monitoring(_: object = Depends(require_role(["admin", "ml_engineer", "data_engineer"]))):
    try:
        return get_model_monitoring_metrics()
    except Exception as e:
        logger.error(f"Error fetching ML monitoring data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ML monitoring data")
