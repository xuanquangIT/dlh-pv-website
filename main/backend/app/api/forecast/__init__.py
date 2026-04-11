from fastapi import APIRouter
from .routes import router as inner_router

router = APIRouter(prefix="/forecast", tags=["Forecast"])
router.include_router(inner_router)
