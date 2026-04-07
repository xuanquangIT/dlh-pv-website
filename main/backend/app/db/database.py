import uuid
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from app.core.settings import get_db_settings

settings = get_db_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class AuthRole(Base):
    __tablename__ = "auth_roles"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)


class AuthUser(Base):
    __tablename__ = "auth_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    role_id = Column(String, ForeignKey("auth_roles.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    role = relationship("AuthRole")


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
