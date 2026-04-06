from fastapi import APIRouter

router = APIRouter(prefix="/ml-training", tags=["ML Training"])


@router.get("/experiments")
def get_ml_training_experiments() -> dict[str, str]:
    return {
        "module": "ml_training",
        "message": "ML Training API placeholder is ready.",
    }
