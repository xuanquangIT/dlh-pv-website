import uuid
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.api.dependencies import get_current_user, require_role
from app.core.settings import get_auth_settings
from app.db.database import AuthUser
from app.schemas.auth import (
    AdminUserCreate,
    LoginRequest,
    UserCreate,
    UserPasswordUpdate,
    UserRead,
    UserStatusUpdate,
)
from app.services.auth.service import AuthService

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


def _login_error_redirect(next_path: str, error_code: str) -> RedirectResponse:
    query = urlencode({"next": next_path, "error": error_code})
    return RedirectResponse(url=f"/login?{query}", status_code=303)


@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_path: str | None = Form(default="/dashboard", alias="next"),
):
    service = AuthService()
    login_req = LoginRequest(username=username, password=password)
    redirect_target = _sanitize_next_path(next_path)
    try:
        token = service.authenticate_user(login_req)
    except HTTPException as exc:
        detail = str(exc.detail)
        if detail == "Inactive user":
            return _login_error_redirect(redirect_target, "inactive")
        if detail == "Incorrect username or password":
            return _login_error_redirect(redirect_target, "invalid_credentials")
        raise

    settings = get_auth_settings()
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
):
    service = AuthService()
    return service.create_user(user_in)


@router.get("/users", response_model=list[UserRead])
def list_users(
    _current_user: AuthUser = Depends(require_role(["admin"])),
):
    service = AuthService()
    return service.list_users()


@router.post("/users", response_model=UserRead)
def create_user_by_admin(
    user_in: AdminUserCreate,
    _current_user: AuthUser = Depends(require_role(["admin"])),
):
    service = AuthService()
    return service.create_user_by_admin(user_in)


@router.patch("/users/{user_id}/status", response_model=UserRead)
def update_user_status(
    user_id: uuid.UUID,
    payload: UserStatusUpdate,
    current_user: AuthUser = Depends(require_role(["admin"])),
):
    service = AuthService()
    return service.update_user_status(
        user_id=user_id,
        is_active=payload.is_active,
        current_user=current_user,
    )


@router.patch("/users/{user_id}/password", status_code=204)
def reset_user_password(
    user_id: uuid.UUID,
    payload: UserPasswordUpdate,
    _current_user: AuthUser = Depends(require_role(["admin"])),
):
    service = AuthService()
    service.reset_user_password(user_id=user_id, new_password=payload.new_password)
