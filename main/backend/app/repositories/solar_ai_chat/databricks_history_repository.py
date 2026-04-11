import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from databricks import sql as databricks_sql

from app.core.settings import SolarChatSettings
from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTopic,
    SourceMetadata,
)

logger = logging.getLogger(__name__)


class DatabricksChatHistoryRepository:
    """Databricks SQL-backed persistence for chat sessions and messages."""

    def __init__(self, settings: SolarChatSettings) -> None:
        self._settings = settings
        self._app_schema = (settings.uc_app_schema or "app").strip().lower()

        configured_catalog = (settings.uc_app_catalog or settings.uc_catalog or "dlh_web").strip()
        fallback_catalog = configured_catalog.replace("-", "_")
        self._catalog_candidates = [configured_catalog]
        if fallback_catalog not in self._catalog_candidates:
            self._catalog_candidates.append(fallback_catalog)

        self._resolved_catalog: str | None = None
        self._invalid_role_log_keys: set[str] = set()

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return f"`{identifier.replace('`', '``')}`"

    @staticmethod
    def _quote_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _datetime_sql(value: datetime) -> str:
        utc = value.astimezone(timezone.utc).replace(tzinfo=None)
        return utc.strftime("%Y-%m-%d %H:%M:%S")

    @contextmanager
    def _connection(self):
        host = (self._settings.databricks_host or "").strip()
        token = (self._settings.databricks_token or "").strip()
        http_path = (self._settings.resolved_databricks_http_path or "").strip()

        if not host or not token or not http_path:
            raise ValueError(
                "Missing Databricks connection settings. Required: DATABRICKS_HOST, DATABRICKS_TOKEN, "
                "and DATABRICKS_SQL_HTTP_PATH (or DATABRICKS_WAREHOUSE_ID)."
            )

        parsed = urlparse(host)
        server_hostname = parsed.netloc if parsed.scheme else host
        if not server_hostname:
            raise ValueError("Invalid DATABRICKS_HOST value.")

        conn = databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _table_name(self, catalog: str, table: str) -> str:
        return (
            f"{self._quote_identifier(catalog)}."
            f"{self._quote_identifier(self._app_schema)}."
            f"{self._quote_identifier(table)}"
        )

    def _resolve_catalog(self) -> str:
        if self._resolved_catalog:
            return self._resolved_catalog

        last_error: Exception | None = None
        with self._connection() as conn:
            cursor = conn.cursor()
            for catalog in self._catalog_candidates:
                try:
                    cursor.execute(
                        f"SELECT 1 FROM {self._table_name(catalog, 'chat_sessions')} LIMIT 1"
                    )
                    self._resolved_catalog = catalog
                    return catalog
                except Exception as exc:
                    last_error = exc
                    continue

        if last_error:
            raise RuntimeError(
                "Cannot resolve app catalog for chat history tables. "
                f"Tried: {', '.join(self._catalog_candidates)}"
            ) from last_error
        raise RuntimeError("Cannot resolve app catalog for chat history tables.")

    def _execute(self, sql: str) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)

    def _fetch_all(self, sql: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    def create_session(self, role: ChatRole, title: str, owner_user_id: str) -> ChatSessionSummary:
        catalog = self._resolve_catalog()
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)
        now_sql = self._datetime_sql(now)

        safe_session_id = self._quote_literal(session_id)
        safe_title = self._quote_literal(title)
        safe_role = self._quote_literal(role.value)
        safe_owner = self._quote_literal(owner_user_id)

        self._execute(
            f"""
            INSERT INTO {self._table_name(catalog, 'chat_sessions')}
              (session_id, title, role, owner_user_id, created_at, updated_at)
            VALUES
              ('{safe_session_id}', '{safe_title}', '{safe_role}', '{safe_owner}',
               TIMESTAMP '{now_sql}', TIMESTAMP '{now_sql}')
            """
        )

        return ChatSessionSummary(
            session_id=session_id,
            title=title,
            role=role,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_sessions(self, owner_user_id: str) -> list[ChatSessionSummary]:
        catalog = self._resolve_catalog()
        safe_owner = self._quote_literal(owner_user_id)

        rows = self._fetch_all(
            f"""
            SELECT s.session_id,
                   s.title,
                   s.role,
                   s.created_at,
                   s.updated_at,
                   COALESCE(COUNT(m.id), 0) AS message_count
            FROM {self._table_name(catalog, 'chat_sessions')} s
            LEFT JOIN {self._table_name(catalog, 'chat_messages')} m
              ON m.session_id = s.session_id
            WHERE s.owner_user_id = '{safe_owner}'
            GROUP BY s.session_id, s.title, s.role, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
        )

        sessions: list[ChatSessionSummary] = []
        for row in rows:
            try:
                role = self._parse_role(row.get("role", ""))
            except ValueError:
                self._warn_invalid_role_once(f"session:{row.get('session_id', '')}", session_id=str(row.get("session_id", "")))
                continue

            sessions.append(
                ChatSessionSummary(
                    session_id=str(row["session_id"]),
                    title=str(row["title"]),
                    role=role,
                    created_at=self._normalize_datetime(row.get("created_at")),
                    updated_at=self._normalize_datetime(row.get("updated_at")),
                    message_count=int(row.get("message_count") or 0),
                )
            )

        return sessions

    def session_exists(self, session_id: str, owner_user_id: str) -> bool:
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)
        safe_owner = self._quote_literal(owner_user_id)

        rows = self._fetch_all(
            f"""
            SELECT 1 AS found
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_session_id}'
              AND owner_user_id = '{safe_owner}'
            LIMIT 1
            """
        )
        return bool(rows)

    def get_session(self, session_id: str, owner_user_id: str) -> ChatSessionDetail | None:
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)
        safe_owner = self._quote_literal(owner_user_id)

        session_rows = self._fetch_all(
            f"""
            SELECT session_id, title, role, created_at, updated_at
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_session_id}'
              AND owner_user_id = '{safe_owner}'
            LIMIT 1
            """
        )
        if not session_rows:
            return None

        session_row = session_rows[0]
        try:
            role = self._parse_role(session_row.get("role", ""))
        except ValueError:
            self._warn_invalid_role_once(f"session:{session_id}", session_id=session_id)
            return None

        message_rows = self._fetch_all(
            f"""
            SELECT id, session_id, sender, content, timestamp, topic, sources
            FROM {self._table_name(catalog, 'chat_messages')}
            WHERE session_id = '{safe_session_id}'
            ORDER BY timestamp ASC
            """
        )

        return ChatSessionDetail(
            session_id=str(session_row["session_id"]),
            title=str(session_row["title"]),
            role=role,
            created_at=self._normalize_datetime(session_row.get("created_at")),
            updated_at=self._normalize_datetime(session_row.get("updated_at")),
            messages=[self._deserialize_message(row) for row in message_rows],
        )

    def delete_session(self, session_id: str, owner_user_id: str) -> bool:
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)
        safe_owner = self._quote_literal(owner_user_id)

        rows = self._fetch_all(
            f"""
            SELECT 1 AS exists_flag
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_session_id}'
              AND owner_user_id = '{safe_owner}'
            LIMIT 1
            """
        )
        if not rows:
            return False

        self._execute(
            f"DELETE FROM {self._table_name(catalog, 'chat_messages')} "
            f"WHERE session_id = '{safe_session_id}'"
        )
        self._execute(
            f"DELETE FROM {self._table_name(catalog, 'chat_sessions')} "
            f"WHERE session_id = '{safe_session_id}'"
        )
        return True

    def update_session_title(
        self,
        session_id: str,
        title: str,
        owner_user_id: str,
    ) -> ChatSessionSummary | None:
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)
        safe_owner = self._quote_literal(owner_user_id)
        safe_title = self._quote_literal(title)
        now = datetime.now(tz=timezone.utc)
        now_sql = self._datetime_sql(now)

        session_rows = self._fetch_all(
            f"""
            SELECT session_id, role, created_at
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_session_id}'
              AND owner_user_id = '{safe_owner}'
            LIMIT 1
            """
        )
        if not session_rows:
            return None

        self._execute(
            f"""
            UPDATE {self._table_name(catalog, 'chat_sessions')}
            SET title = '{safe_title}',
                updated_at = TIMESTAMP '{now_sql}'
            WHERE session_id = '{safe_session_id}'
              AND owner_user_id = '{safe_owner}'
            """
        )

        message_count_rows = self._fetch_all(
            f"""
            SELECT COALESCE(COUNT(id), 0) AS message_count
            FROM {self._table_name(catalog, 'chat_messages')}
            WHERE session_id = '{safe_session_id}'
            """
        )

        role_value = session_rows[0].get("role", "")
        try:
            role = self._parse_role(role_value)
        except ValueError:
            self._warn_invalid_role_once(f"session:{session_id}", session_id=session_id)
            return None

        created_at = self._normalize_datetime(session_rows[0].get("created_at"))
        message_count = int(message_count_rows[0].get("message_count") or 0)
        return ChatSessionSummary(
            session_id=session_id,
            title=title,
            role=role,
            created_at=created_at,
            updated_at=now,
            message_count=message_count,
        )

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        topic: ChatTopic | None = None,
        sources: list[SourceMetadata] | None = None,
    ) -> ChatMessage | None:
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)

        exists_rows = self._fetch_all(
            f"""
            SELECT 1 AS exists_flag
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_session_id}'
            LIMIT 1
            """
        )
        if not exists_rows:
            return None

        message_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)
        now_sql = self._datetime_sql(now)

        safe_message_id = self._quote_literal(message_id)
        safe_sender = self._quote_literal(sender)
        safe_content = self._quote_literal(content)
        topic_expr = "NULL"
        if topic is not None:
            topic_expr = f"'{self._quote_literal(topic.value)}'"
        sources_expr = self._sources_to_sql_expression(sources)

        self._execute(
            f"""
            INSERT INTO {self._table_name(catalog, 'chat_messages')}
              (id, session_id, sender, content, timestamp, topic, sources, created_at)
            VALUES
              ('{safe_message_id}', '{safe_session_id}', '{safe_sender}', '{safe_content}',
               TIMESTAMP '{now_sql}', {topic_expr}, {sources_expr}, TIMESTAMP '{now_sql}')
            """
        )
        self._execute(
            f"""
            UPDATE {self._table_name(catalog, 'chat_sessions')}
            SET updated_at = TIMESTAMP '{now_sql}'
            WHERE session_id = '{safe_session_id}'
            """
        )

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
        catalog = self._resolve_catalog()
        safe_session_id = self._quote_literal(session_id)
        safe_limit = max(1, int(limit))

        rows = self._fetch_all(
            f"""
            SELECT id, session_id, sender, content, timestamp, topic, sources
            FROM {self._table_name(catalog, 'chat_messages')}
            WHERE session_id = '{safe_session_id}'
            ORDER BY timestamp DESC
            LIMIT {safe_limit}
            """
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
        catalog = self._resolve_catalog()
        safe_source_session_id = self._quote_literal(source_session_id)
        safe_owner = self._quote_literal(owner_user_id)

        source_rows = self._fetch_all(
            f"""
            SELECT session_id, title, role
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_source_session_id}'
              AND owner_user_id = '{safe_owner}'
            LIMIT 1
            """
        )
        if not source_rows:
            return None

        source_row = source_rows[0]
        try:
            role = new_role if new_role else self._parse_role(source_row.get("role", ""))
        except ValueError:
            self._warn_invalid_role_once(
                f"fork_source_session:{source_session_id}",
                session_id=source_session_id,
            )
            return None

        title = new_title or f"Fork of {source_row['title']}"
        new_session_id = uuid.uuid4().hex[:12]
        now = datetime.now(tz=timezone.utc)
        now_sql = self._datetime_sql(now)

        safe_new_session_id = self._quote_literal(new_session_id)
        safe_title = self._quote_literal(title)
        safe_role = self._quote_literal(role.value)

        self._execute(
            f"""
            INSERT INTO {self._table_name(catalog, 'chat_sessions')}
              (session_id, title, role, owner_user_id, created_at, updated_at)
            VALUES
              ('{safe_new_session_id}', '{safe_title}', '{safe_role}', '{safe_owner}',
               TIMESTAMP '{now_sql}', TIMESTAMP '{now_sql}')
            """
        )

        source_messages = self._fetch_all(
            f"""
            SELECT sender, content, timestamp, topic, sources
            FROM {self._table_name(catalog, 'chat_messages')}
            WHERE session_id = '{safe_source_session_id}'
            ORDER BY timestamp ASC
            """
        )

        copied_count = 0
        for row in source_messages:
            message_timestamp = self._normalize_datetime(row.get("timestamp"))
            message_timestamp_sql = self._datetime_sql(message_timestamp)
            copied_message_id = self._quote_literal(uuid.uuid4().hex[:12])
            copied_sender = self._quote_literal(str(row.get("sender") or "assistant"))
            copied_content = self._quote_literal(str(row.get("content") or ""))
            topic_value = row.get("topic")
            topic_expr = "NULL"
            if topic_value:
                topic_expr = f"'{self._quote_literal(str(topic_value))}'"
            sources_expr = self._sources_to_sql_expression(
                self._normalize_sources(row.get("sources"))
            )

            self._execute(
                f"""
                INSERT INTO {self._table_name(catalog, 'chat_messages')}
                  (id, session_id, sender, content, timestamp, topic, sources, created_at)
                VALUES
                  ('{copied_message_id}', '{safe_new_session_id}', '{copied_sender}', '{copied_content}',
                   TIMESTAMP '{message_timestamp_sql}', {topic_expr}, {sources_expr}, TIMESTAMP '{now_sql}')
                """
            )
            copied_count += 1

        if copied_count > 0:
            updated_now_sql = self._datetime_sql(datetime.now(tz=timezone.utc))
            self._execute(
                f"""
                UPDATE {self._table_name(catalog, 'chat_sessions')}
                SET updated_at = TIMESTAMP '{updated_now_sql}'
                WHERE session_id = '{safe_new_session_id}'
                """
            )

        summary_rows = self._fetch_all(
            f"""
            SELECT created_at, updated_at
            FROM {self._table_name(catalog, 'chat_sessions')}
            WHERE session_id = '{safe_new_session_id}'
            LIMIT 1
            """
        )
        created_at = now
        updated_at = now
        if summary_rows:
            created_at = self._normalize_datetime(summary_rows[0].get("created_at"))
            updated_at = self._normalize_datetime(summary_rows[0].get("updated_at"))

        return ChatSessionSummary(
            session_id=new_session_id,
            title=title,
            role=role,
            created_at=created_at,
            updated_at=updated_at,
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
    def _normalize_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                pass

        return datetime.now(tz=timezone.utc)

    def _sources_to_sql_expression(self, sources: list[SourceMetadata] | None) -> str:
        if not sources:
            return "NULL"

        struct_sqls: list[str] = []
        for source in sources:
            layer = self._quote_literal(source.layer)
            dataset = self._quote_literal(source.dataset)
            data_source = self._quote_literal(source.data_source)
            struct_sqls.append(
                "named_struct("
                f"'layer', '{layer}', "
                f"'dataset', '{dataset}', "
                f"'data_source', '{data_source}'"
                ")"
            )

        return f"array({', '.join(struct_sqls)})"

    def _normalize_sources(self, raw_sources: Any) -> list[SourceMetadata] | None:
        if raw_sources is None:
            return None

        source_items = raw_sources
        if isinstance(raw_sources, str):
            text = raw_sources.strip()
            if not text:
                return None
            try:
                source_items = json.loads(text)
            except json.JSONDecodeError:
                return None

        if not isinstance(source_items, list):
            return None

        normalized: list[SourceMetadata] = []
        for item in source_items:
            if isinstance(item, SourceMetadata):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                try:
                    normalized.append(SourceMetadata(**item))
                except Exception:
                    continue
                continue
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                try:
                    normalized.append(
                        SourceMetadata(
                            layer=str(item[0]),
                            dataset=str(item[1]),
                            data_source=str(item[2]),
                        )
                    )
                except Exception:
                    continue
                continue

            try:
                layer = str(getattr(item, "layer"))
                dataset = str(getattr(item, "dataset"))
                data_source = str(getattr(item, "data_source"))
                normalized.append(
                    SourceMetadata(layer=layer, dataset=dataset, data_source=data_source)
                )
            except Exception:
                continue

        return normalized or None

    def _deserialize_message(self, row: dict[str, Any]) -> ChatMessage:
        topic_value = row.get("topic")
        topic: ChatTopic | None = None
        if topic_value:
            try:
                topic = ChatTopic(str(topic_value))
            except ValueError:
                topic = None

        return ChatMessage(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            sender=str(row["sender"]),
            content=str(row["content"]),
            timestamp=self._normalize_datetime(row.get("timestamp")),
            topic=topic,
            sources=self._normalize_sources(row.get("sources")),
        )
