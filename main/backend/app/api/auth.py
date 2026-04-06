from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, require_role
from app.core.settings import get_auth_settings
from app.db.database import AuthUser, get_db
from app.schemas.auth import LoginRequest, UserCreate, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _sanitize_next_path(next_path: str | None) -> str:
    candidate = (next_path or "").strip()
    if not candidate:
        return "/dashboard"
    if not candidate.startswith("/"):
        return "/dashboard"
    if candidate.startswith("//") or candidate.startswith("/\\"):
        return "/dashboard"
    return candidate


@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_path: str | None = Form(default="/dashboard", alias="next"),
    db: Session = Depends(get_db),
):
    service = AuthService(db)
    login_req = LoginRequest(username=username, password=password)
    token = service.authenticate_user(login_req)

    settings = get_auth_settings()
    redirect_target = _sanitize_next_path(next_path)
    response = RedirectResponse(url=redirect_target, status_code=303)
    response.set_cookie(
        key=settings.cookie_name,
        value=token.access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        expires=settings.access_token_expire_minutes * 60,
        secure=settings.cookie_secure and request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return response


@router.get("/logout")
@router.post("/logout")
def logout():
    settings = get_auth_settings()
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=settings.cookie_name, path="/")
    return response


@router.get("/me", response_model=UserRead)
def get_me(current_user: AuthUser = Depends(get_current_user)):
    return current_user


@router.post("/register", response_model=UserRead)
def register_user(
    user_in: UserCreate,
    current_user: AuthUser = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    service = AuthService(db)
    return service.create_user(user_in)
