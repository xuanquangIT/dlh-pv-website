from fastapi import APIRouter
from .routes import router as inner_router

router = APIRouter(prefix="/ml-training", tags=["ML Training"])
router.include_router(inner_router)
