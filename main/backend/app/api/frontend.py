from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

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
        "ui_path": None,
    },
    {
        "name": "Data Quality",
        "description": "Data quality score and issue summaries.",
        "endpoint": "/data-quality/score",
        "ui_path": None,
    },
    {
        "name": "ML Training",
        "description": "Training experiments and model comparison.",
        "endpoint": "/ml-training/experiments",
        "ui_path": None,
    },
    {
        "name": "Model Registry",
        "description": "Model versions and release details.",
        "endpoint": "/model-registry/models",
        "ui_path": None,
    },
    {
        "name": "Forecast",
        "description": "72-hour forecast outputs and confidence.",
        "endpoint": "/forecast/next-72h",
        "ui_path": None,
    },
    {
        "name": "Analytics",
        "description": "Analytical query history and insights.",
        "endpoint": "/analytics/query-history",
        "ui_path": None,
    },
    {
        "name": "Solar AI Chat",
        "description": "Natural language assistant for platform users.",
        "endpoint": "/solar-ai-chat/topics",
        "ui_path": "/solar-chat",
    },
]

UI_TEST_PAGES: dict[str, str] = {
    "dashboard": "Dashboard",
    "pipeline": "Data Pipeline",
    "quality": "Data Quality",
    "training": "ML Training",
    "registry": "Model Registry",
    "forecast": "Forecast",
    "analytics": "Analytics",
    "solar_chat": "Solar AI Chat",
}


def _render_refactored_page(
    request: Request,
    template_name: str,
    current_page: str,
    page_title: str,
) -> HTMLResponse:
    base_context = {
        "current_page": current_page,
        "page_title": page_title,
        "system_health": "Pipeline healthy",
        "user_name": "Admin User",
        "user_role": "Platform Owner",
        "user_initials": "AK",
    }
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=base_context,
    )


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"modules": MODULE_CARDS},
    )


@router.get("/solar-ai-chat", include_in_schema=False)
def solar_ai_chat_legacy_redirect() -> RedirectResponse:
    return RedirectResponse(url="/solar-chat", status_code=307)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/dashboard.html",
        current_page="dashboard",
        page_title=UI_TEST_PAGES["dashboard"],
    )


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/pipeline.html",
        current_page="pipeline",
        page_title=UI_TEST_PAGES["pipeline"],
    )


@router.get("/quality", response_class=HTMLResponse)
def quality_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/quality.html",
        current_page="quality",
        page_title=UI_TEST_PAGES["quality"],
    )


@router.get("/training", response_class=HTMLResponse)
def training_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/training.html",
        current_page="training",
        page_title=UI_TEST_PAGES["training"],
    )


@router.get("/registry", response_class=HTMLResponse)
def registry_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/registry.html",
        current_page="registry",
        page_title=UI_TEST_PAGES["registry"],
    )


@router.get("/forecast", response_class=HTMLResponse)
def forecast_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/forecast.html",
        current_page="forecast",
        page_title=UI_TEST_PAGES["forecast"],
    )


@router.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/analytics.html",
        current_page="analytics",
        page_title=UI_TEST_PAGES["analytics"],
    )


@router.get("/solar-chat", response_class=HTMLResponse)
def solar_chat_page(request: Request) -> HTMLResponse:
    return _render_refactored_page(
        request=request,
        template_name="platform_portal/solar_chat.html",
        current_page="solar_chat",
        page_title=UI_TEST_PAGES["solar_chat"],
    )

