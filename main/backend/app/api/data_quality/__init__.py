from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])


@router.get("/score")
def get_data_quality_score(
    _: object = Depends(require_role(["data_engineer"])),
) -> dict[str, str]:
    return {
        "module": "data_quality",
        "message": "Data Quality API placeholder is ready.",
    }
