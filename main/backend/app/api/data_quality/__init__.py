from fastapi import APIRouter

from app.api.data_quality.routes import router as data_quality_router

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])

router.include_router(data_quality_router)
