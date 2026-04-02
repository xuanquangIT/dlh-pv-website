from fastapi import APIRouter

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("/next-72h")
def get_forecast_next_72h() -> dict[str, str]:
    return {
        "module": "forecast",
        "message": "Forecast API placeholder is ready.",
    }
