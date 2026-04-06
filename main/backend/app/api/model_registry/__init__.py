from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/model-registry", tags=["Model Registry"])


@router.get("/models")
def get_registered_models(
    _: object = Depends(require_role(["admin", "ml_engineer", "system"])),
) -> dict[str, str]:
    return {
        "module": "model_registry",
        "message": "Model Registry API placeholder is ready.",
    }
