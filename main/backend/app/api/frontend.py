from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Frontend"], include_in_schema=False)

project_root = Path(__file__).resolve().parents[3]
templates = Jinja2Templates(directory=str(project_root / "frontend" / "templates"))

MODULE_CARDS = [
    {
        "name": "Dashboard",
        "description": "Overall production overview and key metrics.",
        "endpoint": "/dashboard/summary",
        "ui_path": None,
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
        "description": "Vietnamese natural language assistant for business users.",
        "endpoint": "/solar-ai-chat/topics",
        "ui_path": "/solar-ai-chat/test",
    },
]


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"modules": MODULE_CARDS},
    )


@router.get("/solar-ai-chat/test", response_class=HTMLResponse)
def solar_ai_chat_test_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="solar_ai_chat_test.html",
        context={},
    )
