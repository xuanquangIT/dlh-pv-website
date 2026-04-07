from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/data-pipeline", tags=["Data Pipeline"])


@router.get("/status")
def get_data_pipeline_status(
    _: object = Depends(require_role(["data_engineer", "system"])),
) -> dict[str, str]:
    return {
        "module": "data_pipeline",
        "message": "Data Pipeline API placeholder is ready.",
    }
