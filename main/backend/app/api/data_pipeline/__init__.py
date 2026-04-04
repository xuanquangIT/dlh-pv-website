from fastapi import APIRouter

router = APIRouter(prefix="/data-pipeline", tags=["Data Pipeline"])


@router.get("/status")
def get_data_pipeline_status() -> dict[str, str]:
    return {
        "module": "data_pipeline",
        "message": "Data Pipeline API placeholder is ready.",
    }
