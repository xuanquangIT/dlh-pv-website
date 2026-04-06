import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTopic,
    SourceMetadata,
)

logger = logging.getLogger(__name__)


class ChatHistoryRepository:
    """File-based persistence for chat sessions and messages."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._invalid_role_log_keys: set[str] = set()

    def create_session(self, role: ChatRole, title: str) -> ChatSessionSummary:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)
        session_data: dict[str, Any] = {
            "session_id": session_id,
            "title": title,
            "role": role.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "messages": [],
        }
        self._write_session(session_id, session_data)
        return ChatSessionSummary(
            session_id=session_id,
            title=title,
            role=role,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_sessions(self) -> list[ChatSessionSummary]:
        sessions: list[ChatSessionSummary] = []
        for file_path in sorted(self._storage_dir.glob("*.json"), reverse=True):
            data = self._read_file(file_path)
            if data is None:
                continue
            try:
                sessions.append(self._deserialize_session_summary(data))
            except ValueError:
                self._warn_invalid_role_once(f"path:{file_path}", path=str(file_path))
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def get_session(self, session_id: str) -> ChatSessionDetail | None:
        data = self._load_session(session_id)
        if data is None:
            return None
        try:
            role = self._parse_role(data.get("role", ""))
        except ValueError:
            self._warn_invalid_role_once(f"session:{session_id}", session_id=session_id)
            return None
        messages = [self._deserialize_message(msg) for msg in data.get("messages", [])]
        return ChatSessionDetail(
            session_id=data["session_id"],
            title=data["title"],
            role=role,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            messages=messages,
        )

    def delete_session(self, session_id: str) -> bool:
        file_path = self._session_path(session_id)
        if not file_path.exists():
            return False
        file_path.unlink()
        return True

    def update_session_title(
        self,
        session_id: str,
        title: str,
    ) -> ChatSessionSummary | None:
        data = self._load_session(session_id)
        if data is None:
            return None

        now = datetime.now(tz=timezone.utc)
        data["title"] = title
        data["updated_at"] = now.isoformat()
        self._write_session(session_id, data)
        return self._deserialize_session_summary(data)

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        topic: ChatTopic | None = None,
        sources: list[SourceMetadata] | None = None,
    ) -> ChatMessage | None:
        data = self._load_session(session_id)
        if data is None:
            return None

        message_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)
        msg_data: dict[str, Any] = {
            "id": message_id,
            "session_id": session_id,
            "sender": sender,
            "content": content,
            "timestamp": now.isoformat(),
            "topic": topic.value if topic else None,
            "sources": [s.model_dump() for s in sources] if sources else None,
        }
        data["messages"].append(msg_data)
        data["updated_at"] = now.isoformat()
        self._write_session(session_id, data)

        return self._deserialize_message(msg_data)

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[ChatMessage]:
        data = self._load_session(session_id)
        if data is None:
            return []
        raw_messages = data.get("messages", [])[-limit:]
        return [self._deserialize_message(msg) for msg in raw_messages]

    def fork_session(
        self,
        source_session_id: str,
        new_title: str,
        new_role: ChatRole | None = None,
    ) -> ChatSessionSummary | None:
        source_data = self._load_session(source_session_id)
        if source_data is None:
            return None

        # C6: new_role is already a ChatRole, no need to unwrap and re-wrap
        try:
            role = new_role if new_role else self._parse_role(source_data.get("role", ""))
        except ValueError:
            self._warn_invalid_role_once(
                f"fork_source_session:{source_session_id}",
                session_id=source_session_id,
            )
            return None
        title = new_title or f"Fork of {source_data['title']}"
        new_session = self.create_session(role=role, title=title)

        # C7: source_data is still valid; create_session writes to a different file
        new_data = self._load_session(new_session.session_id)
        if new_data is None:
            return new_session

        copied_messages: list[dict[str, Any]] = [
            {**msg, "id": uuid.uuid4().hex[:12], "session_id": new_session.session_id}
            for msg in source_data.get("messages", [])
        ]
        new_data["messages"] = copied_messages
        new_data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._write_session(new_session.session_id, new_data)

        return ChatSessionSummary(
            session_id=new_session.session_id,
            title=title,
            role=role,
            created_at=new_session.created_at,
            updated_at=datetime.fromisoformat(new_data["updated_at"]),
            message_count=len(copied_messages),
        )

    def _session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self._storage_dir / f"{safe_id}.json"

    def _load_session(self, session_id: str) -> dict[str, Any] | None:
        return self._read_file(self._session_path(session_id))

    def _write_session(self, session_id: str, data: dict[str, Any]) -> None:
        file_path = self._session_path(session_id)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _warn_invalid_role_once(self, key: str, **context: str) -> None:
        if key in self._invalid_role_log_keys:
            return
        self._invalid_role_log_keys.add(key)
        logger.warning("invalid_session_role key=%s context=%s", key, context)

    @staticmethod
    def _deserialize_session_summary(data: dict[str, Any]) -> ChatSessionSummary:
        """D11: Shared deserializer used by list_sessions and get_session."""
        return ChatSessionSummary(
            session_id=data["session_id"],
            title=data["title"],
            role=ChatHistoryRepository._parse_role(data.get("role", "")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            message_count=len(data.get("messages", [])),
        )

    @staticmethod
    def _parse_role(role_value: str) -> ChatRole:
        normalized_value = str(role_value).strip().lower().replace(" ", "_")
        return ChatRole(normalized_value)

    @staticmethod
    def _read_file(file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            return None
        try:
            # Accept UTF-8 with or without BOM for legacy session files.
            return json.loads(file_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            logger.warning("corrupted_session_file", extra={"path": str(file_path)})
            return None

    @staticmethod
    def _deserialize_message(msg: dict[str, Any]) -> ChatMessage:
        sources = None
        if msg.get("sources"):
            sources = [SourceMetadata(**s) for s in msg["sources"]]
        return ChatMessage(
            id=msg["id"],
            session_id=msg["session_id"],
            sender=msg["sender"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"]),
            topic=ChatTopic(msg["topic"]) if msg.get("topic") else None,
            sources=sources,
        )
