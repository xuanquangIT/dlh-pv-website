from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/ml-training", tags=["ML Training"])


@router.get("/experiments")
def get_ml_training_experiments(
    _: object = Depends(require_role(["ml_engineer", "system"])),
) -> dict[str, str]:
    return {
        "module": "ml_training",
        "message": "ML Training API placeholder is ready.",
    }
