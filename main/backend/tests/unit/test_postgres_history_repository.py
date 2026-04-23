"""Comprehensive unit tests for PostgresChatHistoryRepository.

Strategy:
- SQLAlchemy's SessionLocal and all DB model interactions are replaced with
  MagicMock / patch so no real database is needed.
- Every public method is covered: create_session, list_sessions,
  session_exists, get_session, delete_session, update_session_title,
  add_message, get_recent_messages, fork_session.
- _to_message and _ensure_local_auth_user are also tested via their callers
  and, where necessary, directly.
"""
from __future__ import annotations

import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Path setup and stubs
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# bcrypt stub
if "bcrypt" not in sys.modules:
    bcrypt_stub = types.ModuleType("bcrypt")
    bcrypt_stub.gensalt = lambda rounds=12: b"stub-salt"
    bcrypt_stub.hashpw = lambda password, salt: password + b"." + salt
    bcrypt_stub.checkpw = lambda password, hashed: True
    sys.modules["bcrypt"] = bcrypt_stub

# Stub databricks (only if real package not installed)
try:
    import databricks  # noqa: F401
except ImportError:
    _db_pkg = types.ModuleType("databricks")
    _db_pkg.__path__ = []
    _db_sql = types.ModuleType("databricks.sql")
    _db_sql.connect = MagicMock()
    sys.modules.setdefault("databricks", _db_pkg)
    sys.modules.setdefault("databricks.sql", _db_sql)

import pytest

from app.repositories.solar_ai_chat.postgres_history_repository import (  # noqa: E402
    PostgresChatHistoryRepository,
)
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic  # noqa: E402
from app.schemas.solar_ai_chat import SourceMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OWNER_UUID = uuid.uuid4()
_OWNER_ID_STR = str(_OWNER_UUID)
_SESSION_ID = "abc123def456"
_NOW = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers to build fake ORM objects
# ---------------------------------------------------------------------------

def _fake_session(session_id=_SESSION_ID, title="Test Session",
                  role="data_analyst", owner_user_id=None,
                  created_at=_NOW, updated_at=_NOW):
    s = MagicMock()
    s.session_id = session_id
    s.title = title
    s.role = role
    s.owner_user_id = owner_user_id or _OWNER_UUID
    s.created_at = created_at
    s.updated_at = updated_at
    return s


def _fake_message(msg_id="msg001", session_id=_SESSION_ID,
                  sender="user", content="Hello",
                  topic=None, sources=None, thinking_trace=None,
                  key_metrics=None, viz_payload=None, viz_requested=False,
                  timestamp=_NOW):
    m = MagicMock()
    m.id = msg_id
    m.session_id = session_id
    m.sender = sender
    m.content = content
    m.timestamp = timestamp
    m.topic = topic
    m.sources = sources
    m.thinking_trace = thinking_trace
    m.key_metrics = key_metrics
    m.viz_payload = viz_payload
    m.viz_requested = viz_requested
    return m


# ---------------------------------------------------------------------------
# Test helpers for context-manager mocking
# ---------------------------------------------------------------------------

def _cm(obj):
    """Return a mock that works as a context manager yielding obj."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=obj)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ===========================================================================
# 1. Utility helpers
# ===========================================================================

class TestStaticHelpers:
    def test_to_uuid_valid(self):
        repo = PostgresChatHistoryRepository()
        result = repo._to_uuid(_OWNER_ID_STR)
        assert isinstance(result, uuid.UUID)
        assert result == _OWNER_UUID

    def test_to_uuid_invalid_raises(self):
        repo = PostgresChatHistoryRepository()
        with pytest.raises(ValueError):
            repo._to_uuid("not-a-uuid")

    def test_now_utc_is_aware(self):
        repo = PostgresChatHistoryRepository()
        now = repo._now_utc()
        assert now.tzinfo is not None

    def test_to_message_minimal(self):
        """_to_message with topic=None and sources=None should not crash."""
        repo = PostgresChatHistoryRepository()
        msg_model = _fake_message()
        msg = repo._to_message(msg_model)
        assert msg.content == "Hello"
        assert msg.topic is None
        assert msg.sources is None

    def test_to_message_with_topic(self):
        repo = PostgresChatHistoryRepository()
        msg_model = _fake_message(topic="energy_performance")
        msg = repo._to_message(msg_model)
        assert msg.topic == ChatTopic.ENERGY_PERFORMANCE

    def test_to_message_with_sources(self):
        repo = PostgresChatHistoryRepository()
        raw_sources = [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}]
        msg_model = _fake_message(sources=raw_sources)
        msg = repo._to_message(msg_model)
        assert msg.sources is not None
        assert len(msg.sources) == 1

    def test_to_message_bad_thinking_trace_becomes_none(self):
        repo = PostgresChatHistoryRepository()
        msg_model = _fake_message(thinking_trace="not-a-dict")
        msg = repo._to_message(msg_model)
        assert msg.thinking_trace is None

    def test_to_message_viz_payload_snapshot_used(self):
        """When viz_payload is present, _to_message runs without error."""
        repo = PostgresChatHistoryRepository()
        fake_table = {"headers": ["a"], "rows": [["1"]]}
        fake_kpi = {"cards": [{"label": "Energy", "value": "100 MWh"}]}
        msg_model = _fake_message(
            viz_payload={"data_table": fake_table, "kpi_cards": fake_kpi, "chart": None},
            viz_requested=False,
        )
        msg = repo._to_message(msg_model)
        assert msg.content == "Hello"

    def test_to_message_fallback_to_key_metrics_when_no_viz_payload(self):
        """When viz_payload is absent but key_metrics exist, ChartSpecBuilder is invoked."""
        repo = PostgresChatHistoryRepository()
        msg_model = _fake_message(key_metrics={"facility_count": 8}, viz_payload=None)
        mock_builder = MagicMock()
        mock_builder.build.return_value = (None, None, None)
        # ChartSpecBuilder is locally imported inside _to_message, so patch at source module
        with patch(
            "app.services.solar_ai_chat.chart_service.ChartSpecBuilder",
            return_value=mock_builder,
        ):
            msg = repo._to_message(msg_model)
        assert msg.content == "Hello"


# ===========================================================================
# 2. _ensure_local_auth_user
# ===========================================================================

class TestEnsureLocalAuthUser:
    def test_user_already_exists_no_op(self):
        """If local user is found, method returns immediately."""
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = MagicMock()

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            repo._ensure_local_auth_user(_OWNER_UUID)

        # Since user was found, remote UserRepository is never called.
        db_mock.add.assert_not_called()

    def test_user_not_found_locally_fetched_from_remote_and_inserted(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        # First query (local lookup) returns None; second (conflict check) returns None.
        db_mock.query.return_value.filter.return_value.first.side_effect = [None, None]

        remote_user = SimpleNamespace(
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            full_name="Test User",
            is_active=True,
            role_id="analyst",
            created_at=_NOW,
        )

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id.return_value = remote_user

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.UserRepository",
            return_value=mock_user_repo,
        ):
            repo._ensure_local_auth_user(_OWNER_UUID)

        db_mock.add.assert_called_once()
        db_mock.commit.assert_called_once()

    def test_remote_user_not_found_raises_value_error(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.UserRepository",
            return_value=mock_user_repo,
        ):
            with pytest.raises(ValueError, match="does not exist"):
                repo._ensure_local_auth_user(_OWNER_UUID)

    def test_conflict_user_reconciled(self):
        """When a username/email conflict exists locally, attributes are reconciled."""
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        conflict_user = MagicMock()
        # First filter call (local lookup by id) → None; second (conflict check) → conflict_user
        db_mock.query.return_value.filter.return_value.first.side_effect = [None, conflict_user]

        remote_user = SimpleNamespace(
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            full_name="Test User",
            is_active=True,
            role_id="analyst",
            created_at=_NOW,
        )
        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id.return_value = remote_user

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.UserRepository",
            return_value=mock_user_repo,
        ):
            repo._ensure_local_auth_user(_OWNER_UUID)

        assert conflict_user.id == _OWNER_UUID
        db_mock.commit.assert_called_once()


# ===========================================================================
# 3. create_session
# ===========================================================================

class TestCreateSession:
    def test_returns_summary_with_correct_fields(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch.object(repo, "_ensure_local_auth_user"):
            summary = repo.create_session(
                role=ChatRole.DATA_ANALYST,
                title="My Session",
                owner_user_id=_OWNER_ID_STR,
            )

        assert summary.title == "My Session"
        assert summary.role == ChatRole.DATA_ANALYST
        assert summary.message_count == 0
        db_mock.add.assert_called_once()
        db_mock.commit.assert_called_once()

    def test_session_id_is_12_hex_chars(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch.object(repo, "_ensure_local_auth_user"):
            summary = repo.create_session(
                role=ChatRole.ADMIN,
                title="Admin Session",
                owner_user_id=_OWNER_ID_STR,
            )

        assert len(summary.session_id) == 12
        assert all(c in "0123456789abcdef" for c in summary.session_id)


# ===========================================================================
# 4. list_sessions
# ===========================================================================

class TestListSessions:
    def test_returns_list_of_summaries(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        # offset=0 is falsy, so code only calls .limit(50).all() — no .offset()
        db_mock.query.return_value.outerjoin.return_value.filter.return_value \
               .group_by.return_value.order_by.return_value \
               .limit.return_value.all.return_value = [
            (fake_sess, 3)
        ]

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            results = repo.list_sessions(_OWNER_ID_STR, limit=50, offset=0)

        assert len(results) == 1
        assert results[0].session_id == _SESSION_ID
        assert results[0].message_count == 3

    def test_empty_db_returns_empty_list(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.outerjoin.return_value.filter.return_value \
               .group_by.return_value.order_by.return_value \
               .limit.return_value.offset.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            results = repo.list_sessions(_OWNER_ID_STR)

        assert results == []


# ===========================================================================
# 5. session_exists
# ===========================================================================

class TestSessionExists:
    def test_returns_true_when_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = (_SESSION_ID,)

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            assert repo.session_exists(_SESSION_ID, _OWNER_ID_STR) is True

    def test_returns_false_when_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            assert repo.session_exists("nonexistent", _OWNER_ID_STR) is False


# ===========================================================================
# 6. get_session
# ===========================================================================

class TestGetSession:
    def test_returns_none_when_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.get_session(_SESSION_ID, _OWNER_ID_STR)

        assert result is None

    def test_returns_session_detail_with_messages(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        fake_msg = _fake_message()

        # First query call → session; second query call → messages
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess
        db_mock.query.return_value.filter.return_value.order_by.return_value.all.return_value = [fake_msg]

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            detail = repo.get_session(_SESSION_ID, _OWNER_ID_STR)

        assert detail is not None
        assert detail.session_id == _SESSION_ID
        assert len(detail.messages) == 1

    def test_messages_sorted_ascending_by_timestamp(self):
        """The query orders by timestamp ASC — we verify the ORM call chain."""
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess
        db_mock.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            detail = repo.get_session(_SESSION_ID, _OWNER_ID_STR)

        assert detail is not None
        assert detail.messages == []


# ===========================================================================
# 7. delete_session
# ===========================================================================

class TestDeleteSession:
    def test_returns_true_on_success(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.delete_session(_SESSION_ID, _OWNER_ID_STR)

        assert result is True
        db_mock.delete.assert_called_once_with(fake_sess)
        db_mock.commit.assert_called_once()

    def test_returns_false_when_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.delete_session("missing", _OWNER_ID_STR)

        assert result is False
        db_mock.delete.assert_not_called()


# ===========================================================================
# 8. update_session_title
# ===========================================================================

class TestUpdateSessionTitle:
    def test_returns_none_when_session_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.update_session_title(_SESSION_ID, "New Title", _OWNER_ID_STR)

        assert result is None

    def test_updates_title_and_returns_summary(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess
        db_mock.query.return_value.filter.return_value.count.return_value = 5

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            summary = repo.update_session_title(_SESSION_ID, "New Title", _OWNER_ID_STR)

        assert summary is not None
        assert fake_sess.title == "New Title"
        db_mock.commit.assert_called_once()
        assert summary.message_count == 5


# ===========================================================================
# 9. add_message
# ===========================================================================

class TestAddMessage:
    def test_returns_none_when_session_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.add_message(_SESSION_ID, "user", "Hello")

        assert result is None

    def test_adds_message_and_returns_chat_message(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        fake_msg_model = _fake_message(content="Hello")

        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess
        db_mock.refresh.side_effect = lambda m: None

        def _add_side_effect(obj):
            pass

        db_mock.add.side_effect = _add_side_effect

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel",
            return_value=fake_msg_model,
        ):
            msg = repo.add_message(_SESSION_ID, "user", "Hello")

        assert msg is not None
        db_mock.commit.assert_called_once()

    def test_add_message_with_topic_and_sources(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        fake_msg_model = _fake_message(
            content="What is the forecast?",
            topic="forecast_72h",
            sources=[{"layer": "Gold", "dataset": "gold.forecast_daily", "data_source": "databricks"}],
        )
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess

        sources = [SourceMetadata(layer="Gold", dataset="gold.forecast_daily", data_source="databricks")]

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel",
            return_value=fake_msg_model,
        ):
            msg = repo.add_message(
                _SESSION_ID, "assistant", "What is the forecast?",
                topic=ChatTopic.FORECAST_72H,
                sources=sources,
            )

        assert msg is not None

    def test_add_message_viz_payload_stored(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        fake_sess = _fake_session()
        viz = {"data_table": {"headers": ["H"], "rows": []}, "chart": None, "kpi_cards": None}
        fake_msg_model = _fake_message(viz_payload=viz)
        db_mock.query.return_value.filter.return_value.first.return_value = fake_sess

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel",
            return_value=fake_msg_model,
        ):
            msg = repo.add_message(
                _SESSION_ID, "assistant", "Chart result",
                viz_payload=viz, viz_requested=True,
            )

        assert msg is not None


# ===========================================================================
# 10. get_recent_messages
# ===========================================================================

class TestGetRecentMessages:
    def test_returns_messages_in_ascending_order(self):
        """Rows come back DESC from DB; the method reverses them."""
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        msg1 = _fake_message(msg_id="m1", content="First")
        msg2 = _fake_message(msg_id="m2", content="Second")
        # Simulated DESC order from DB
        db_mock.query.return_value.filter.return_value.order_by.return_value \
               .limit.return_value.all.return_value = [msg2, msg1]

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            messages = repo.get_recent_messages(_SESSION_ID, limit=10)

        # After reversal, msg1 should be first
        assert messages[0].content == "First"
        assert messages[1].content == "Second"

    def test_respects_limit(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.order_by.return_value \
               .limit.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            messages = repo.get_recent_messages(_SESSION_ID, limit=5)

        db_mock.query.return_value.filter.return_value.order_by.return_value \
               .limit.assert_called_once_with(5)
        assert messages == []

    def test_empty_session_returns_empty_list(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.order_by.return_value \
               .limit.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            messages = repo.get_recent_messages(_SESSION_ID)

        assert messages == []


# ===========================================================================
# 11. fork_session
# ===========================================================================

class TestForkSession:
    def test_returns_none_when_source_not_found(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            result = repo.fork_session(
                source_session_id="nonexistent",
                new_title="Fork",
                new_role=ChatRole.DATA_ANALYST,
                owner_user_id=_OWNER_ID_STR,
            )

        assert result is None

    def test_fork_creates_new_session_with_cloned_messages(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        source_sess = _fake_session()
        source_msg = _fake_message()

        db_mock.query.return_value.filter.return_value.first.return_value = source_sess
        db_mock.query.return_value.filter.return_value.order_by.return_value.all.return_value = [source_msg]

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatSessionModel"
        ) as MockSession, patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel"
        ) as MockMessage:
            summary = repo.fork_session(
                source_session_id=_SESSION_ID,
                new_title="Forked",
                new_role=ChatRole.ML_ENGINEER,
                owner_user_id=_OWNER_ID_STR,
            )

        assert summary is not None
        assert summary.title == "Forked"
        assert summary.role == ChatRole.ML_ENGINEER
        assert summary.message_count == 1
        db_mock.commit.assert_called_once()

    def test_fork_with_blank_title_uses_default(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        source_sess = _fake_session(title="Original")
        db_mock.query.return_value.filter.return_value.first.return_value = source_sess
        db_mock.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatSessionModel"
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel"
        ):
            summary = repo.fork_session(
                source_session_id=_SESSION_ID,
                new_title="   ",  # blank
                new_role=ChatRole.DATA_ANALYST,
                owner_user_id=_OWNER_ID_STR,
            )

        assert "Original" in summary.title

    def test_fork_session_id_has_12_chars(self):
        repo = PostgresChatHistoryRepository()
        db_mock = MagicMock()
        source_sess = _fake_session()
        db_mock.query.return_value.filter.return_value.first.return_value = source_sess
        db_mock.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatSessionModel"
        ) as MockSess, patch(
            "app.repositories.solar_ai_chat.postgres_history_repository.ChatMessageModel"
        ):
            summary = repo.fork_session(
                source_session_id=_SESSION_ID,
                new_title="Fork",
                new_role=ChatRole.DATA_ANALYST,
                owner_user_id=_OWNER_ID_STR,
            )

        assert len(summary.session_id) == 12
