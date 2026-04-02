from fastapi import APIRouter

router = APIRouter(prefix="/model-registry", tags=["Model Registry"])


@router.get("/models")
def get_registered_models() -> dict[str, str]:
    return {
        "module": "model_registry",
        "message": "Model Registry API placeholder is ready.",
    }
