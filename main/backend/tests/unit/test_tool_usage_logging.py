"""Task 0.1 — ToolExecutor telemetry hook.

We verify the contract (success + failure both log; a logger failure must
not break the caller) without touching a real database by injecting a
lightweight fake logger.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.services.solar_ai_chat.tool_executor import ToolExecutor


class _RecordingLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def log_usage(self, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))


class _ExplodingLogger:
    def log_usage(self, **kwargs: object) -> None:
        raise RuntimeError("telemetry backend down")


def _make_repo() -> MagicMock:
    repo = MagicMock(spec=SolarChatRepository)
    repo._resolve_latest_date.return_value = date(2025, 5, 5)
    repo.fetch_topic_metrics.return_value = (
        {"facility_count": 8},
        [{"layer": "Gold", "dataset": "g", "data_source": "trino"}],
    )
    return repo


class TestUsageLoggingHappyPath:
    def test_success_logs_one_row_with_latency_and_role(self) -> None:
        tracker = _RecordingLogger()
        ex = ToolExecutor(_make_repo(), usage_logger=tracker)

        ex.execute("get_system_overview", {}, ChatRole.ADMIN)

        assert len(tracker.rows) == 1
        row = tracker.rows[0]
        assert row["tool_name"] == "get_system_overview"
        assert row["role"] == "admin"
        assert row["success"] is True
        assert row["error_message"] is None
        assert isinstance(row["latency_ms"], int) and row["latency_ms"] >= 0

    def test_request_context_is_attributed(self) -> None:
        tracker = _RecordingLogger()
        ex = ToolExecutor(_make_repo(), usage_logger=tracker)
        ex.set_request_context(session_id="abcd1234efgh", user_id="u-123")

        ex.execute("get_system_overview", {}, ChatRole.ADMIN)

        row = tracker.rows[0]
        assert row["session_id"] == "abcd1234efgh"
        assert row["user_id"] == "u-123"


class TestUsageLoggingFailurePath:
    def test_permission_error_logs_failure_and_reraises(self) -> None:
        tracker = _RecordingLogger()
        ex = ToolExecutor(_make_repo(), usage_logger=tracker)

        with pytest.raises(PermissionError):
            # data_engineer cannot access get_ml_model_info (topic ML_MODEL)
            ex.execute("get_ml_model_info", {}, ChatRole.DATA_ENGINEER)

        assert len(tracker.rows) == 1
        row = tracker.rows[0]
        assert row["tool_name"] == "get_ml_model_info"
        assert row["success"] is False
        assert "PermissionError" in str(row["error_message"])

    def test_unknown_tool_raises_value_error_and_logs(self) -> None:
        tracker = _RecordingLogger()
        ex = ToolExecutor(_make_repo(), usage_logger=tracker)

        with pytest.raises(ValueError):
            ex.execute("nonexistent_tool", {}, ChatRole.ADMIN)

        assert tracker.rows[-1]["success"] is False


class TestUsageLoggingDefensive:
    def test_logger_exception_does_not_break_caller(self) -> None:
        ex = ToolExecutor(_make_repo(), usage_logger=_ExplodingLogger())
        # Must not raise: telemetry is best-effort.
        metrics, _ = ex.execute("get_system_overview", {}, ChatRole.ADMIN)
        assert "facility_count" in metrics

    def test_logger_none_is_pure_passthrough(self) -> None:
        ex = ToolExecutor(_make_repo(), usage_logger=None)
        metrics, _ = ex.execute("get_system_overview", {}, ChatRole.ADMIN)
        assert "facility_count" in metrics
