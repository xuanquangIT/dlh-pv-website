import uuid
from typing import Generator

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, create_engine
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


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String(12), primary_key=True)
    title = Column(String(200), nullable=False)
    role = Column(String(50), nullable=False)
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("auth_users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(12), primary_key=True)
    session_id = Column(
        String(12),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sender = Column(String(20), nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    topic = Column(String(50), nullable=True)
    sources = Column(JSON, nullable=True)
    thinking_trace = Column(JSON, nullable=True)  # ← persisted ThinkingTrace dict
    key_metrics = Column(JSON, nullable=True)  # ← compact tool outputs, used to rebuild viz on hydrate (legacy fallback)
    viz_requested = Column(Boolean, nullable=False, default=False, server_default="false")
    viz_payload = Column(JSON, nullable=True)  # ← exact viz snapshot (data_table/chart/kpi_cards) for faithful reload

    session = relationship("ChatSession", back_populates="messages")


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _apply_runtime_migrations() -> None:
    """Idempotent forward-compatible column adds for pre-existing databases.

    Keeps Solar AI Chat working when the operator hasn't re-run
    ``002-create-lakehouse-tables.sql`` after pulling new columns. Safe to
    call multiple times — the SQL uses IF NOT EXISTS.
    """
    from sqlalchemy import text
    statements = (
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS key_metrics JSONB",
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS viz_requested BOOLEAN DEFAULT FALSE NOT NULL",
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS viz_payload JSONB",
    )
    try:
        with engine.begin() as conn:
            for stmt in statements:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    # Best-effort — not Postgres, or permissions denied. Don't block startup.
                    pass
    except Exception:
        pass


_apply_runtime_migrations()
