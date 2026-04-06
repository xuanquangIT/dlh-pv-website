from fastapi import APIRouter, Depends

from app.api.dependencies import require_role

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("/next-72h")
def get_forecast_next_72h(
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"])),
) -> dict[str, str]:
    return {
        "module": "forecast",
        "message": "Forecast API placeholder is ready.",
    }
