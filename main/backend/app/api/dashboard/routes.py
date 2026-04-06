from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.schemas.dashboard import EmbedTokenResponse
from app.services.dashboard.powerbi_service import PowerBIService, get_powerbi_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

ALLOWED_DASHBOARD_ROLES = {
    "data_analyst",
    "data_engineer",
    "ml_engineer",
    "admin",
}


def require_dashboard_embed_access(
    x_user_role: Annotated[str | None, Header(alias="X-User-Role")] = None,
) -> str:
    if not x_user_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Role header.",
        )

    normalized_role = x_user_role.strip().lower().replace(" ", "_")
    if normalized_role not in ALLOWED_DASHBOARD_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role is not allowed to access dashboard embed information.",
        )

    return normalized_role

@router.get("/summary")
def get_dashboard_summary() -> dict[str, str]:
    return {
        "module": "dashboard",
        "message": "Dashboard API placeholder is ready.",
    }

@router.get("/embed-info", response_model=EmbedTokenResponse)
def get_dashboard_embed_info(
    _: str = Depends(require_dashboard_embed_access),
    powerbi_service: PowerBIService = Depends(get_powerbi_service),
):
    """
    Returns the Power BI embed token and embed URL for authorized roles.
    """
    return powerbi_service.get_embed_info()
