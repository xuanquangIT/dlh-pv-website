from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.api.dependencies import require_role
from app.services.databricks_service import get_registry_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model-registry", tags=["Model Registry"])

@router.get("/models-list", response_model=List[Dict[str, Any]])
def get_models_list(_: object = Depends(require_role(["admin", "ml_engineer", "data_engineer"]))):
    try:
        models = get_registry_models()
        return models
    except Exception as e:
        logger.error(f"Error fetching model registry data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch model registry data")
