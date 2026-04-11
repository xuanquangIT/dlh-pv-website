from fastapi import APIRouter
from .routes import router as inner_router

router = APIRouter(prefix="/model-registry", tags=["Model Registry"])
router.include_router(inner_router)
