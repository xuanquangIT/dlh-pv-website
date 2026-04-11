import uuid
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import func

from app.core.security import get_password_hash
from app.db.database import SessionLocal
from app.db.database import AuthRole, AuthUser
from app.schemas.auth import AdminUserCreate, UserCreate


class DuplicateUserError(Exception):
    """Raised when creating a user violates a unique username/email constraint."""

    def __init__(self, field: str):
        super().__init__(f"Duplicate value for field: {field}")
        self.field = field


class UserRepository:
    def __init__(self, _db=None):
        pass

    @staticmethod
    def _as_auth_user(row: dict) -> AuthUser:
        # Use attribute-compatible object for existing service/schema code paths.
        return SimpleNamespace(
            id=uuid.UUID(str(row["id"])),
            username=row["username"],
            email=row["email"],
            hashed_password=row["hashed_password"],
            full_name=row.get("full_name"),
            is_active=bool(row.get("is_active", True)),
            role_id=row["role_id"],
            created_at=row.get("created_at") or datetime.utcnow(),
            role=None,
        )

    def get_by_username(self, username: str) -> AuthUser | None:
        with SessionLocal() as db:
            user = (
                db.query(AuthUser)
                .filter(func.lower(AuthUser.username) == username.lower())
                .first()
            )
        row = None
        if user:
            row = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "hashed_password": user.hashed_password,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "role_id": user.role_id,
                "created_at": user.created_at,
            }
        return self._as_auth_user(row) if row else None

    def get_by_email(self, email: str) -> AuthUser | None:
        with SessionLocal() as db:
            user = (
                db.query(AuthUser)
                .filter(func.lower(AuthUser.email) == email.lower())
                .first()
            )
        row = None
        if user:
            row = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "hashed_password": user.hashed_password,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "role_id": user.role_id,
                "created_at": user.created_at,
            }
        return self._as_auth_user(row) if row else None

    def get_by_id(self, user_id: uuid.UUID) -> AuthUser | None:
        with SessionLocal() as db:
            user = db.query(AuthUser).filter(AuthUser.id == user_id).first()
        row = None
        if user:
            row = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "hashed_password": user.hashed_password,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "role_id": user.role_id,
                "created_at": user.created_at,
            }
        return self._as_auth_user(row) if row else None

    def create(self, user_in: UserCreate | AdminUserCreate) -> AuthUser:
        if self.get_by_username(user_in.username):
            raise DuplicateUserError("username")
        if self.get_by_email(user_in.email):
            raise DuplicateUserError("email")

        user_id = uuid.uuid4()
        with SessionLocal() as db:
            db.add(
                AuthUser(
                    id=user_id,
                    username=user_in.username,
                    email=str(user_in.email),
                    hashed_password=get_password_hash(user_in.password),
                    full_name=user_in.full_name,
                    is_active=getattr(user_in, "is_active", True),
                    role_id=user_in.role_id,
                )
            )
            db.commit()

        created = self.get_by_id(user_id)
        if not created:
            raise RuntimeError("User insert succeeded but could not re-read inserted user.")
        return created

    def get_role_by_id(self, role_id: str) -> AuthRole | None:
        with SessionLocal() as db:
            role = db.query(AuthRole).filter(AuthRole.id == role_id).first()
        if not role:
            return None
        return SimpleNamespace(id=role.id, name=role.name, description=role.description)

    def list_users(self) -> list[AuthUser]:
        with SessionLocal() as db:
            users = db.query(AuthUser).order_by(AuthUser.created_at.desc()).all()
        rows = [
            {
                "id": row.id,
                "username": row.username,
                "email": row.email,
                "hashed_password": row.hashed_password,
                "full_name": row.full_name,
                "is_active": row.is_active,
                "role_id": row.role_id,
                "created_at": row.created_at,
            }
            for row in users
        ]
        return [self._as_auth_user(row) for row in rows]

    def update_active_status(self, user: AuthUser, is_active: bool) -> AuthUser:
        with SessionLocal() as db:
            row = db.query(AuthUser).filter(AuthUser.id == user.id).first()
            if row is None:
                raise ValueError(f"User id '{user.id}' not found")
            row.is_active = is_active
            db.commit()
        updated = self.get_by_id(uuid.UUID(str(user.id)))
        if updated is None:
            raise RuntimeError("User update succeeded but could not re-read updated user.")
        return updated

    def update_password(self, user: AuthUser, new_password: str) -> None:
        with SessionLocal() as db:
            row = db.query(AuthUser).filter(AuthUser.id == user.id).first()
            if row is None:
                raise ValueError(f"User id '{user.id}' not found")
            row.hashed_password = get_password_hash(new_password)
            db.commit()
