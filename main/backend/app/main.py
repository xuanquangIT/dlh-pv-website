from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth
from app.core.security import get_password_hash
from app.db.database import AuthRole, AuthUser, Base, SessionLocal, engine

from app.api import (
    dashboard,
    data_pipeline,
    data_quality,
    frontend,
    forecast,
    ml_training,
    model_registry,
    solar_ai_chat,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup handled natively via PostgreSQL migrations (002-create-lakehouse-tables.sql)
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="PV Lakehouse API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def unauthorized_redirect_handler(request: Request, exc: HTTPException):
        if exc.status_code == 401:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse(url="/login", status_code=303)
        return await http_exception_handler(request, exc)

    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "frontend" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(data_pipeline.router)
    app.include_router(data_quality.router)
    app.include_router(ml_training.router)
    app.include_router(model_registry.router)
    app.include_router(forecast.router)
    app.include_router(solar_ai_chat.router)
    app.include_router(solar_ai_chat.stream_router)
    app.include_router(frontend.router)

    return app


app = create_app()
