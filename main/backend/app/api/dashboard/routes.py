from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user, require_role
from app.db.database import AuthUser
from app.schemas.dashboard import EmbedTokenResponse
from app.services.dashboard.powerbi_service import PowerBIService, get_powerbi_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

ALLOWED_DASHBOARD_ROLES = {
    "analyst",
    "data_engineer",
    "ml_engineer",
    "admin",
}


def require_dashboard_embed_access(
    current_user: AuthUser = Depends(get_current_user),
) -> str:
    normalized_role = (current_user.role_id or "").strip().lower()
    if normalized_role not in ALLOWED_DASHBOARD_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role is not allowed to access dashboard embed information.",
        )

    return normalized_role

@router.get("/summary")
def get_dashboard_summary(
    _: object = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"])),
) -> dict[str, str]:
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
