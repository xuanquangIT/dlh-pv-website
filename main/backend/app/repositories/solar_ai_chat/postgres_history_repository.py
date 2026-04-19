import uuid
from datetime import datetime, timezone

from sqlalchemy import func

from app.db.database import ChatMessage as ChatMessageModel
from app.db.database import ChatSession as ChatSessionModel
from app.db.database import AuthUser as AuthUserModel
from app.db.database import SessionLocal
from app.repositories.auth.user_repository import UserRepository
from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTopic,
    SourceMetadata,
    ThinkingTrace,
)


class PostgresChatHistoryRepository:
    """PostgreSQL-backed persistence for chat sessions and messages."""

    @staticmethod
    def _to_uuid(value: str) -> uuid.UUID:
        return uuid.UUID(value)

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _to_message(model: ChatMessageModel) -> ChatMessage:
        topic = ChatTopic(model.topic) if model.topic else None
        sources = None
        if model.sources:
            sources = [SourceMetadata.model_validate(row) for row in model.sources]
        thinking_trace = None
        if model.thinking_trace:
            try:
                thinking_trace = ThinkingTrace.model_validate(model.thinking_trace)
            except Exception:
                thinking_trace = None

        # Viz reload strategy:
        #   1) Prefer the exact rendered snapshot stored in `viz_payload`
        #      (guarantees live↔reload fidelity — no re-picking the "best"
        #      list on different input ordering).
        #   2) Fall back to rebuilding from `key_metrics` for legacy rows
        #      written before viz_payload existed.
        data_table = None
        chart = None
        kpi_cards = None
        snapshot = getattr(model, "viz_payload", None)
        if snapshot and isinstance(snapshot, dict):
            from app.schemas.solar_ai_chat.visualization import (
                ChartPayload,
                DataTablePayload,
                KpiCardsPayload,
            )
            try:
                if snapshot.get("data_table"):
                    data_table = DataTablePayload.model_validate(snapshot["data_table"])
            except Exception:
                data_table = None
            try:
                if snapshot.get("chart") and bool(getattr(model, "viz_requested", False)):
                    chart = ChartPayload.model_validate(snapshot["chart"])
            except Exception:
                chart = None
            try:
                if snapshot.get("kpi_cards"):
                    kpi_cards = KpiCardsPayload.model_validate(snapshot["kpi_cards"])
            except Exception:
                kpi_cards = None
        else:
            raw_metrics = getattr(model, "key_metrics", None)
            if raw_metrics:
                try:
                    from app.services.solar_ai_chat.chart_service import ChartSpecBuilder
                    data_table, chart, kpi_cards = ChartSpecBuilder().build(
                        raw_metrics, topic=model.topic,
                    )
                    if not bool(getattr(model, "viz_requested", False)):
                        chart = None
                except Exception:
                    data_table = None
                    chart = None
                    kpi_cards = None

        return ChatMessage(
            id=model.id,
            session_id=model.session_id,
            sender=model.sender,
            content=model.content,
            timestamp=model.timestamp,
            topic=topic,
            sources=sources,
            thinking_trace=thinking_trace,
            data_table=data_table,
            chart=chart,
            kpi_cards=kpi_cards,
        )

    def _ensure_local_auth_user(self, owner_uuid: uuid.UUID) -> None:
        with SessionLocal() as db:
            local_user = db.query(AuthUserModel).filter(AuthUserModel.id == owner_uuid).first()
            if local_user is not None:
                return

            remote_user = UserRepository().get_by_id(owner_uuid)
            if remote_user is None:
                raise ValueError(f"Owner user id '{owner_uuid}' does not exist in auth source.")

            # Neon bootstrap may have pre-seeded users with random UUIDs.
            # Reconcile by username/email so the authenticated Databricks UUID can satisfy FK constraints.
            local_conflict = (
                db.query(AuthUserModel)
                .filter(
                    (AuthUserModel.username == remote_user.username)
                    | (AuthUserModel.email == remote_user.email)
                )
                .first()
            )
            if local_conflict is not None:
                local_conflict.id = owner_uuid
                local_conflict.username = remote_user.username
                local_conflict.email = remote_user.email
                local_conflict.hashed_password = remote_user.hashed_password
                local_conflict.full_name = remote_user.full_name
                local_conflict.is_active = bool(remote_user.is_active)
                local_conflict.role_id = remote_user.role_id
                local_conflict.created_at = remote_user.created_at
                db.commit()
                return

            db.add(
                AuthUserModel(
                    id=owner_uuid,
                    username=remote_user.username,
                    email=remote_user.email,
                    hashed_password=remote_user.hashed_password,
                    full_name=remote_user.full_name,
                    is_active=bool(remote_user.is_active),
                    role_id=remote_user.role_id,
                    created_at=remote_user.created_at,
                )
            )
            db.commit()

    def create_session(self, role: ChatRole, title: str, owner_user_id: str) -> ChatSessionSummary:
        now = self._now_utc()
        session_id = uuid.uuid4().hex[:12]
        owner_uuid = self._to_uuid(owner_user_id)
        self._ensure_local_auth_user(owner_uuid)
        with SessionLocal() as db:
            session = ChatSessionModel(
                session_id=session_id,
                title=title,
                role=role.value,
                owner_user_id=owner_uuid,
                created_at=now,
                updated_at=now,
            )
            db.add(session)
            db.commit()
        return ChatSessionSummary(
            session_id=session_id,
            title=title,
            role=role,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_sessions(self, owner_user_id: str, limit: int = 50, offset: int = 0) -> list[ChatSessionSummary]:
        owner_uuid = self._to_uuid(owner_user_id)
        with SessionLocal() as db:
            query = (
                db.query(
                    ChatSessionModel,
                    func.count(ChatMessageModel.id).label("message_count"),
                )
                .outerjoin(
                    ChatMessageModel,
                    ChatMessageModel.session_id == ChatSessionModel.session_id,
                )
                .filter(ChatSessionModel.owner_user_id == owner_uuid)
                .group_by(ChatSessionModel.session_id)
                .order_by(ChatSessionModel.updated_at.desc())
            )
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)
            rows = query.all()
            results: list[ChatSessionSummary] = []
            for session, message_count in rows:
                results.append(
                    ChatSessionSummary(
                        session_id=session.session_id,
                        title=session.title,
                        role=ChatRole(session.role),
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                        message_count=int(message_count or 0),
                    )
                )
            return results

    def session_exists(self, session_id: str, owner_user_id: str) -> bool:
        owner_uuid = self._to_uuid(owner_user_id)
        with SessionLocal() as db:
            row = (
                db.query(ChatSessionModel.session_id)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_uuid,
                )
                .first()
            )
        return row is not None

    def get_session(self, session_id: str, owner_user_id: str) -> ChatSessionDetail | None:
        owner_uuid = self._to_uuid(owner_user_id)
        with SessionLocal() as db:
            session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_uuid,
                )
                .first()
            )
            if session is None:
                return None

            messages = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .order_by(ChatMessageModel.timestamp.asc())
                .all()
            )

            return ChatSessionDetail(
                session_id=session.session_id,
                title=session.title,
                role=ChatRole(session.role),
                created_at=session.created_at,
                updated_at=session.updated_at,
                messages=[self._to_message(row) for row in messages],
            )

    def delete_session(self, session_id: str, owner_user_id: str) -> bool:
        owner_uuid = self._to_uuid(owner_user_id)
        with SessionLocal() as db:
            session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_uuid,
                )
                .first()
            )
            if session is None:
                return False
            db.delete(session)
            db.commit()
            return True

    def update_session_title(self, session_id: str, title: str, owner_user_id: str) -> ChatSessionSummary | None:
        owner_uuid = self._to_uuid(owner_user_id)
        now = self._now_utc()
        with SessionLocal() as db:
            session = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == session_id,
                    ChatSessionModel.owner_user_id == owner_uuid,
                )
                .first()
            )
            if session is None:
                return None
            session.title = title
            session.updated_at = now
            db.commit()
            message_count = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .count()
            )
            return ChatSessionSummary(
                session_id=session.session_id,
                title=session.title,
                role=ChatRole(session.role),
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=message_count,
            )

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        topic: ChatTopic | None = None,
        sources: list[SourceMetadata] | None = None,
        thinking_trace: ThinkingTrace | None = None,
        key_metrics: dict | None = None,
        viz_requested: bool = False,
        viz_payload: dict | None = None,
    ) -> ChatMessage | None:
        with SessionLocal() as db:
            session = (
                db.query(ChatSessionModel)
                .filter(ChatSessionModel.session_id == session_id)
                .first()
            )
            if session is None:
                return None

            now = self._now_utc()
            message = ChatMessageModel(
                id=uuid.uuid4().hex[:12],
                session_id=session_id,
                sender=sender,
                content=content,
                timestamp=now,
                topic=topic.value if topic else None,
                sources=[row.model_dump() for row in sources] if sources else None,
                thinking_trace=thinking_trace.model_dump() if thinking_trace else None,
                key_metrics=key_metrics if key_metrics else None,
                viz_requested=bool(viz_requested),
                viz_payload=viz_payload if viz_payload else None,
            )
            session.updated_at = now
            db.add(message)
            db.commit()
            db.refresh(message)
            return self._to_message(message)

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[ChatMessage]:
        with SessionLocal() as db:
            rows = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .order_by(ChatMessageModel.timestamp.desc())
                .limit(limit)
                .all()
            )
        rows.reverse()
        return [self._to_message(row) for row in rows]

    def fork_session(
        self,
        source_session_id: str,
        new_title: str,
        new_role: ChatRole,
        owner_user_id: str,
    ) -> ChatSessionSummary | None:
        owner_uuid = self._to_uuid(owner_user_id)
        with SessionLocal() as db:
            source = (
                db.query(ChatSessionModel)
                .filter(
                    ChatSessionModel.session_id == source_session_id,
                    ChatSessionModel.owner_user_id == owner_uuid,
                )
                .first()
            )
            if source is None:
                return None

            new_session_id = uuid.uuid4().hex[:12]
            now = self._now_utc()
            title = new_title.strip() if new_title.strip() else f"Fork of {source.title}"

            cloned_session = ChatSessionModel(
                session_id=new_session_id,
                title=title,
                role=new_role.value,
                owner_user_id=owner_uuid,
                created_at=now,
                updated_at=now,
            )
            db.add(cloned_session)

            source_messages = (
                db.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == source_session_id)
                .order_by(ChatMessageModel.timestamp.asc())
                .all()
            )
            for message in source_messages:
                db.add(
                    ChatMessageModel(
                        id=uuid.uuid4().hex[:12],
                        session_id=new_session_id,
                        sender=message.sender,
                        content=message.content,
                        timestamp=message.timestamp,
                        topic=message.topic,
                        sources=message.sources,
                        thinking_trace=message.thinking_trace,
                    )
                )

            db.commit()
            return ChatSessionSummary(
                session_id=new_session_id,
                title=title,
                role=new_role,
                created_at=now,
                updated_at=now,
                message_count=len(source_messages),
            )