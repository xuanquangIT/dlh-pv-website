import uuid
from typing import Any

from fastapi import HTTPException, status

from app.core.security import create_access_token, verify_password
from app.db.database import AuthUser
from app.repositories.auth.user_repository import DuplicateUserError, UserRepository
from app.schemas.auth import AdminUserCreate, LoginRequest, Token, UserCreate, UserRead


class AuthService:
    def __init__(self, db: Any = None):
        self.repo = UserRepository(db)

    def authenticate_user(self, login_request: LoginRequest) -> Token:
        user = self.repo.get_by_username(login_request.username)
        if not user or not verify_password(login_request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user",
            )

        access_token = create_access_token(subject=user.id)
        return Token(access_token=access_token)

    @staticmethod
    def _raise_duplicate_user_error(error: DuplicateUserError) -> None:
        field_to_message = {
            "username": "Username already exists",
            "email": "Email already exists",
        }
        detail = field_to_message.get(error.field, "User already exists")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        ) from error

    def create_user(self, user_in: UserCreate) -> UserRead:
        if self.repo.get_by_username(user_in.username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        if self.repo.get_by_email(user_in.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already exists",
            )
        role = self.repo.get_role_by_id(user_in.role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role_id",
            )

        try:
            user = self.repo.create(user_in)
        except DuplicateUserError as exc:
            self._raise_duplicate_user_error(exc)
        return UserRead.model_validate(user)

    def list_users(self) -> list[UserRead]:
        users = self.repo.list_users()
        return [UserRead.model_validate(user) for user in users]

    def create_user_by_admin(self, user_in: AdminUserCreate) -> UserRead:
        if self.repo.get_by_username(user_in.username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        if self.repo.get_by_email(user_in.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already exists",
            )
        role = self.repo.get_role_by_id(user_in.role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role_id",
            )

        try:
            user = self.repo.create(user_in)
        except DuplicateUserError as exc:
            self._raise_duplicate_user_error(exc)
        return UserRead.model_validate(user)

    def update_user_status(
        self,
        *,
        user_id: uuid.UUID,
        is_active: bool,
        current_user: AuthUser,
    ) -> UserRead:
        user = self.repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.id == current_user.id and not is_active:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")

        updated_user = self.repo.update_active_status(user, is_active)
        return UserRead.model_validate(updated_user)

    def reset_user_password(
        self,
        *,
        user_id: uuid.UUID,
        new_password: str,
    ) -> None:
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

        user = self.repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        self.repo.update_password(user, new_password)
