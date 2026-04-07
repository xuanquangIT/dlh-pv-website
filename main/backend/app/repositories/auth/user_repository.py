import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.core.security import get_password_hash
from app.db.database import AuthRole, AuthUser
from app.schemas.auth import AdminUserCreate, UserCreate


class DuplicateUserError(Exception):
    """Raised when creating a user violates a unique username/email constraint."""

    def __init__(self, field: str):
        super().__init__(f"Duplicate value for field: {field}")
        self.field = field


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_username(self, username: str) -> AuthUser | None:
        return self.db.execute(
            select(AuthUser).where(AuthUser.username == username)
        ).scalar_one_or_none()

    def get_by_email(self, email: str) -> AuthUser | None:
        return self.db.execute(
            select(AuthUser).where(AuthUser.email == email)
        ).scalar_one_or_none()

    def get_by_id(self, user_id: uuid.UUID) -> AuthUser | None:
        return self.db.get(AuthUser, user_id)

    @staticmethod
    def _extract_duplicate_field(error: IntegrityError) -> str | None:
        raw_message = str(error.orig).lower() if error.orig is not None else str(error).lower()
        if "username" in raw_message:
            return "username"
        if "email" in raw_message:
            return "email"
        return None

    def create(self, user_in: UserCreate | AdminUserCreate) -> AuthUser:
        db_obj = AuthUser(
            username=user_in.username,
            email=user_in.email,
            full_name=user_in.full_name,
            hashed_password=get_password_hash(user_in.password),
            role_id=user_in.role_id,
            is_active=getattr(user_in, "is_active", True),
        )
        self.db.add(db_obj)
        try:
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            duplicate_field = self._extract_duplicate_field(exc)
            if duplicate_field is not None:
                raise DuplicateUserError(duplicate_field) from exc
            raise
        except SQLAlchemyError:
            self.db.rollback()
            raise
        self.db.refresh(db_obj)
        return db_obj

    def get_role_by_id(self, role_id: str) -> AuthRole | None:
        return self.db.get(AuthRole, role_id)

    def list_users(self) -> list[AuthUser]:
        stmt = (
            select(AuthUser)
            .options(selectinload(AuthUser.role))
            .order_by(AuthUser.created_at.desc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def update_active_status(self, user: AuthUser, is_active: bool) -> AuthUser:
        user.is_active = is_active
        self.db.add(user)
        try:
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise
        self.db.refresh(user)
        return user

    def update_password(self, user: AuthUser, new_password: str) -> None:
        user.hashed_password = get_password_hash(new_password)
        self.db.add(user)
        try:
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise
