from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/query-history")
def get_analytics_query_history(
    _: object = Depends(require_role(["admin", "analyst"])),
) -> dict[str, str]:
    return {
        "module": "analytics",
        "message": "Analytics API placeholder is ready.",
    }
