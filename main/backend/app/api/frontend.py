from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.dependencies import get_current_user, require_role
from app.db.database import AuthUser
from app.schemas.solar_ai_chat import ChatRole, resolve_ui_features

router = APIRouter(tags=["Frontend"], include_in_schema=False)

project_root = Path(__file__).resolve().parents[3]
templates = Jinja2Templates(directory=str(project_root / "frontend" / "templates"))

MODULE_CARDS = [
    {
        "name": "Dashboard",
        "description": "Overall production overview and key metrics.",
        "endpoint": "/dashboard/summary",
        "ui_path": "/dashboard",
    },
    {
        "name": "Data Pipeline",
        "description": "Pipeline execution status, progress, and ETA.",
        "endpoint": "/data-pipeline/status",
        "ui_path": "/pipeline",
    },
    {
        "name": "Data Quality",
        "description": "Data quality score and issue summaries.",
        "endpoint": "/data-quality/score",
        "ui_path": "/quality",
    },
    {
        "name": "ML Training",
        "description": "Training experiments and model comparison.",
        "endpoint": "/ml-training/experiments",
        "ui_path": "/training",
    },
    {
        "name": "Model Registry",
        "description": "Model versions and release details.",
        "endpoint": "/model-registry/models",
        "ui_path": "/registry",
    },
    {
        "name": "Forecast",
        "description": "72-hour forecast outputs and confidence.",
        "endpoint": "/forecast/next-72h",
        "ui_path": "/forecast",
    },
    {
        "name": "Solar AI Chat",
        "description": "Natural language assistant for platform users.",
        "endpoint": "/solar-ai-chat/topics",
        "ui_path": "/solar-chat",
    },
]

UI_TEST_PAGES: dict[str, str] = {
    "login": "Login",
    "logout": "Logout",
    "accounts": "Account Management",
    "dashboard": "Dashboard",
    "pipeline": "Data Pipeline",
    "quality": "Data Quality",
    "training": "ML Training",
    "registry": "Model Registry",
    "forecast": "Forecast",
    "solar_chat": "Solar AI Chat",
}


def _sanitize_next_path(next_path: str | None) -> str:
    candidate = (next_path or "").strip()
    if not candidate:
        return "/dashboard"
    if not candidate.startswith("/"):
        return "/dashboard"
    if candidate.startswith("//") or candidate.startswith("/\\"):
        return "/dashboard"
    return candidate


def _to_chat_role(role_id: str) -> str:
    if role_id == "analyst":
        return "data_analyst"
    return role_id


def _render_refactored_page(
    request: Request,
    template_name: str,
    current_page: str,
    page_title: str,
    current_user: AuthUser,
) -> HTMLResponse:
    user_name = current_user.full_name or current_user.username
    chat_role_str = _to_chat_role(current_user.role_id)
    try:
        chat_role_enum = ChatRole(chat_role_str)
    except ValueError:
        chat_role_enum = None
    base_context = {
        "current_page": current_page,
        "page_title": page_title,
        "system_health": "Pipeline healthy",
        "user_name": user_name,
        "user_role": current_user.role.name if getattr(current_user, 'role') else current_user.role_id,
        "api_role": current_user.role_id,
        "chat_role": chat_role_str,
        "chat_ui_features": resolve_ui_features(chat_role_enum),
        "user_initials": "".join([part[0] for part in user_name.split()[:2]]).upper() or "U",
    }
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=base_context,
    )


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request, current_user: AuthUser = Depends(get_current_user)) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"modules": MODULE_CARDS},
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    next_path = _sanitize_next_path(request.query_params.get("next"))
    error_code = request.query_params.get("error")
    login_error_map = {
        "inactive": "Your account is inactive. Please contact an administrator.",
        "invalid_credentials": "Incorrect username or password.",
    }
    login_error = login_error_map.get(error_code, "")
    return templates.TemplateResponse(
        request=request,
        name="platform_portal/login.html",
        context={
            "page_title": UI_TEST_PAGES["login"],
            "next": next_path,
            "login_error": login_error,
        },
    )


@router.get("/logout", response_class=HTMLResponse)
def logout_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="platform_portal/logout.html",
        context={
            "page_title": UI_TEST_PAGES["logout"],
            "user_name": "Admin User",
            "user_role": "Platform Owner",
            "user_initials": "AK",
        },
    )


@router.get("/solar-ai-chat", include_in_schema=False)
def solar_ai_chat_legacy_redirect() -> RedirectResponse:
    return RedirectResponse(url="/solar-chat", status_code=307)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/dashboard.html",
        current_page="dashboard",
        page_title=UI_TEST_PAGES["dashboard"],
        current_user=current_user,
    )


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/pipeline.html",
        current_page="pipeline",
        page_title=UI_TEST_PAGES["pipeline"],
        current_user=current_user,
    )


@router.get("/quality", response_class=HTMLResponse)
def quality_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/quality.html",
        current_page="quality",
        page_title=UI_TEST_PAGES["quality"],
        current_user=current_user,
    )


@router.get("/training", response_class=HTMLResponse)
def training_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "ml_engineer"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/training.html",
        current_page="training",
        page_title=UI_TEST_PAGES["training"],
        current_user=current_user,
    )


@router.get("/registry", response_class=HTMLResponse)
def registry_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "ml_engineer"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/registry.html",
        current_page="registry",
        page_title=UI_TEST_PAGES["registry"],
        current_user=current_user,
    )


@router.get("/forecast", response_class=HTMLResponse)
def forecast_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer", "analyst"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/forecast.html",
        current_page="forecast",
        page_title=UI_TEST_PAGES["forecast"],
        current_user=current_user,
    )

@router.get("/settings/accounts", response_class=HTMLResponse)
def accounts_page(request: Request, current_user: AuthUser = Depends(get_current_user)) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/accounts.html",
        current_page="accounts",
        page_title=UI_TEST_PAGES["accounts"],
        current_user=current_user,
    )


@router.get("/solar-chat", response_class=HTMLResponse)
def solar_chat_page(
    request: Request,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/solar_chat.html",
        current_page="solar_chat",
        page_title=UI_TEST_PAGES["solar_chat"],
        current_user=current_user,
    )

