from fastapi import APIRouter

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])


@router.get("/score")
def get_data_quality_score() -> dict[str, str]:
    return {
        "module": "data_quality",
        "message": "Data Quality API placeholder is ready.",
    }
