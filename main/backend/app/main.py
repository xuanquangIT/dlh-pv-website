from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import (
    analytics,
    dashboard,
    data_pipeline,
    data_quality,
    frontend,
    forecast,
    ml_training,
    model_registry,
    solar_ai_chat,
)


def create_app() -> FastAPI:
    app = FastAPI(title="PV Lakehouse API", version="0.1.0")

    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "frontend" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.include_router(dashboard.router)
    app.include_router(data_pipeline.router)
    app.include_router(data_quality.router)
    app.include_router(ml_training.router)
    app.include_router(model_registry.router)
    app.include_router(forecast.router)
    app.include_router(analytics.router)
    app.include_router(solar_ai_chat.router)
    app.include_router(frontend.router)

    return app


app = create_app()
