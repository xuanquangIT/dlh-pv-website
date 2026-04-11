import uuid

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError

from app.core.security import decode_access_token
from app.core.settings import get_auth_settings
from app.db.database import AuthUser
from app.repositories.auth.user_repository import UserRepository


def get_token_from_cookie(request: Request) -> str | None:
    settings = get_auth_settings()
    return request.cookies.get(settings.cookie_name)


def get_current_user(request: Request) -> AuthUser:
    token = get_token_from_cookie(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    try:
        payload = decode_access_token(token)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
        user_id = uuid.UUID(user_id_str)
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc

    repo = UserRepository()
    user = repo.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user"
        )
    return user


def require_role(allowed_roles: list[str]):
    def role_checker(current_user: AuthUser = Depends(get_current_user)):
        if current_user.role_id not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted"
            )
        return current_user

    return role_checker
