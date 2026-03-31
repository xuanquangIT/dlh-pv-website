from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/query-history")
def get_analytics_query_history() -> dict[str, str]:
    return {
        "module": "analytics",
        "message": "Analytics API placeholder is ready.",
    }
