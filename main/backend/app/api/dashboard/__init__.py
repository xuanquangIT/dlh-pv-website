from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary")
def get_dashboard_summary() -> dict[str, str]:
    return {
        "module": "dashboard",
        "message": "Dashboard API placeholder is ready.",
    }
