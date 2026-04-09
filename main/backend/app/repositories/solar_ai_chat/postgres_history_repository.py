import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from app.db.database import ChatMessage as ChatMessageModel
from app.db.database import ChatSession as ChatSessionModel
from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTopic,
    SourceMetadata,
)

logger = logging.getLogger(__name__)


class PostgresChatHistoryRepository:
    """PostgreSQL-backed persistence for chat sessions and messages."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._invalid_role_log_keys: set[str] = set()

    def create_session(self, role: ChatRole, title: str, owner_user_id: str) -> ChatSessionSummary:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)

        with self._session_factory() as db:
            db_session = ChatSessionModel(
                session_id=session_id,
                title=title,
                role=role.value,
                owner_user_id=owner_user_id,
                created_at=now,
                updated_at=now,
            )
            db.add(db_session)
            db.commit()

        return ChatSessionSummary(
            session_id=session_id,
            title=title,
            role=role,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_sessions(self, owner_user_id: str) -> list[ChatSessionSummary]:
        with self._session_factory() as db:
            rows = (
                db.query(
                    ChatSessionModel.session_id,
                    ChatSessionModel.title,
                    ChatSessionModel.role,
                    ChatSessionModel.created_at,
                    ChatSessionModel.updated_at,
                    func.count(ChatMessageModel.id).label("message_count"),
                )
                .filter(ChatSessionModel.owner_user_id == owner_user_id)
                .outerjoin(ChatMessageModel, ChatMessageModel.session_id == ChatSessionModel.session_id)
                .group_by(
                    ChatSessionModel.session_id,
                    ChatSessionModel.title,
                    ChatSessionModel.role,
                    ChatSessionModel.created_at,
                    ChatSessionModel.updated_at,
                )
                .order_by(ChatSessionModel.updated_at.desc())
                .all()
            )

        sessions: list[ChatSessionSummary] = []
        for row in rows:
            try:
                role = self._parse_role(row.role)
            except ValueError:
                self._warn_invalid_role_once(f"session:{row.session_id}", session_id=row.session_id)
                continue

            sessions.append(
                ChatSessionSummary(
                    session_id=row.session_id,
                    title=row.title,
                    role=role,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    message_count=int(row.message_count or 0),
                )
            )

        return sessions

    def get_session(self, session_id: str, owner_user_id: str) -> ChatSessionDetail | None:
        with self._session_factory() as db:
            db_session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_user_id,
                )
                .one_or_none()
            )
            if db_session is None:
                return None

            try:
                role = self._parse_role(db_session.role)
            except ValueError:
                self._warn_invalid_role_once(f"session:{session_id}", session_id=session_id)
                return None

            rows = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .order_by(ChatMessageModel.timestamp.asc())
                .all()
            )

        return ChatSessionDetail(
            session_id=db_session.session_id,
            title=db_session.title,
            role=role,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            messages=[self._deserialize_message(row) for row in rows],
        )

    def delete_session(self, session_id: str, owner_user_id: str) -> bool:
        with self._session_factory() as db:
            deleted_count = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_user_id,
                )
                .delete(synchronize_session=False)
            )
            if deleted_count == 0:
                db.rollback()
                return False
            db.commit()
            return True

    def update_session_title(
        self,
        session_id: str,
        title: str,
        owner_user_id: str,
    ) -> ChatSessionSummary | None:
        now = datetime.now(tz=timezone.utc)
        with self._session_factory() as db:
            db_session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_user_id,
                )
                .one_or_none()
            )
            if db_session is None:
                return None

            db_session.title = title
            db_session.updated_at = now
            db.commit()

            message_count = (
                db.query(func.count(ChatMessageModel.id))
                .filter(ChatMessageModel.session_id == session_id)
                .scalar()
            )

            try:
                role = self._parse_role(db_session.role)
            except ValueError:
                self._warn_invalid_role_once(f"session:{session_id}", session_id=session_id)
                return None

            return ChatSessionSummary(
                session_id=db_session.session_id,
                title=db_session.title,
                role=role,
                created_at=db_session.created_at,
                updated_at=db_session.updated_at,
                message_count=int(message_count or 0),
            )

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        topic: ChatTopic | None = None,
        sources: list[SourceMetadata] | None = None,
    ) -> ChatMessage | None:
        message_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)

        with self._session_factory() as db:
            db_session = (
                db.query(ChatSessionModel)
                .filter(ChatSessionModel.session_id == session_id)
                .one_or_none()
            )
            if db_session is None:
                return None

            db_message = ChatMessageModel(
                id=message_id,
                session_id=session_id,
                sender=sender,
                content=content,
                timestamp=now,
                topic=topic.value if topic else None,
                sources=[s.model_dump() for s in sources] if sources else None,
            )
            db_session.updated_at = now
            db.add(db_message)
            db.commit()

        return ChatMessage(
            id=message_id,
            session_id=session_id,
            sender=sender,
            content=content,
            timestamp=now,
            topic=topic,
            sources=sources,
        )

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[ChatMessage]:
        with self._session_factory() as db:
            rows = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .order_by(ChatMessageModel.timestamp.desc())
                .limit(limit)
                .all()
            )

        rows.reverse()
        return [self._deserialize_message(row) for row in rows]

    def fork_session(
        self,
        source_session_id: str,
        new_title: str,
        owner_user_id: str,
        new_role: ChatRole | None = None,
    ) -> ChatSessionSummary | None:
        with self._session_factory() as db:
            source_session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == source_session_id,
                    ChatSessionModel.owner_user_id == owner_user_id,
                )
                .one_or_none()
            )
            if source_session is None:
                return None

            try:
                role = new_role if new_role else self._parse_role(source_session.role)
            except ValueError:
                self._warn_invalid_role_once(
                    f"fork_source_session:{source_session_id}",
                    session_id=source_session_id,
                )
                return None

            title = new_title or f"Fork of {source_session.title}"
            new_session_id = uuid.uuid4().hex[:12]
            now = datetime.now(tz=timezone.utc)

            fork_session = ChatSessionModel(
                session_id=new_session_id,
                title=title,
                role=role.value,
                owner_user_id=owner_user_id,
                created_at=now,
                updated_at=now,
            )
            db.add(fork_session)

            source_messages = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == source_session_id)
                .order_by(ChatMessageModel.timestamp.asc())
                .all()
            )

            copied_count = 0
            for source_message in source_messages:
                db.add(
                    ChatMessageModel(
                        id=uuid.uuid4().hex[:12],
                        session_id=new_session_id,
                        sender=source_message.sender,
                        content=source_message.content,
                        timestamp=source_message.timestamp,
                        topic=source_message.topic,
                        sources=source_message.sources,
                    )
                )
                copied_count += 1

            if copied_count > 0:
                fork_session.updated_at = datetime.now(tz=timezone.utc)

            db.commit()

            return ChatSessionSummary(
                session_id=new_session_id,
                title=title,
                role=role,
                created_at=fork_session.created_at,
                updated_at=fork_session.updated_at,
                message_count=copied_count,
            )

    def _warn_invalid_role_once(self, key: str, **context: str) -> None:
        if key in self._invalid_role_log_keys:
            return
        self._invalid_role_log_keys.add(key)
        logger.warning("invalid_session_role key=%s context=%s", key, context)

    @staticmethod
    def _parse_role(role_value: str) -> ChatRole:
        normalized_value = str(role_value).strip().lower().replace(" ", "_")
        return ChatRole(normalized_value)

    @staticmethod
    def _deserialize_message(row: ChatMessageModel) -> ChatMessage:
        topic = ChatTopic(row.topic) if row.topic else None
        sources = None
        if row.sources:
            sources = [SourceMetadata(**source) for source in row.sources]

        return ChatMessage(
            id=row.id,
            session_id=row.session_id,
            sender=row.sender,
            content=row.content,
            timestamp=row.timestamp,
            topic=topic,
            sources=sources,
        )
