import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.db.database import AuthRole, AuthUser
from app.schemas.auth import UserCreate


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

    def create(self, user_in: UserCreate) -> AuthUser:
        db_obj = AuthUser(
            username=user_in.username,
            email=user_in.email,
            full_name=user_in.full_name,
            hashed_password=get_password_hash(user_in.password),
            role_id=user_in.role_id,
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def get_role_by_id(self, role_id: str) -> AuthRole | None:
        return self.db.get(AuthRole, role_id)
