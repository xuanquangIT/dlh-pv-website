from fastapi import APIRouter, Depends

from app.services.powerbi_service import get_powerbi_service, PowerBIService, EmbedTokenResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary")
def get_dashboard_summary() -> dict[str, str]:
    return {
        "module": "dashboard",
        "message": "Dashboard API placeholder is ready.",
    }

@router.get("/embed-info", response_model=EmbedTokenResponse)
def get_dashboard_embed_info(powerbi_service: PowerBIService = Depends(get_powerbi_service)):
    """
    Returns the Power BI embed token and embed URL.
    This route should eventually be protected by the user auth dependency.
    """
    return powerbi_service.get_embed_info()
