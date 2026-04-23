"""Unit coverage for ToolUsageRepository (Task 0.1).

We mock SessionLocal so the test runs without a live Postgres — the repo's
job is (a) produce correctly-shaped insert rows, (b) never raise on DB
errors, (c) aggregate the right columns.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.repositories.solar_ai_chat.tool_usage_repository import ToolUsageRepository


class _FakeSessionCtx:
    """Minimal stand-in for ``SessionLocal()`` context manager."""

    def __init__(self, session: MagicMock) -> None:
        self._session = session

    def __enter__(self) -> MagicMock:
        return self._session

    def __exit__(self, *args: object) -> None:
        self._session.close()


class TestLogUsage:
    def test_happy_path_commits_one_row(self) -> None:
        session = MagicMock()
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            ToolUsageRepository().log_usage(
                tool_name="get_system_overview",
                role="admin",
                latency_ms=123,
                success=True,
                session_id="sess-abcd",
                user_id="00000000-0000-0000-0000-000000000010",
                error_message=None,
            )
        session.add.assert_called_once()
        session.commit.assert_called_once()
        added = session.add.call_args.args[0]
        assert added.tool_name == "get_system_overview"
        assert added.role == "admin"
        assert added.latency_ms == 123
        assert added.success is True

    def test_negative_latency_clamped_to_zero(self) -> None:
        session = MagicMock()
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            ToolUsageRepository().log_usage(
                tool_name="t", role=None, latency_ms=-50, success=False,
            )
        added = session.add.call_args.args[0]
        assert added.latency_ms == 0

    def test_long_error_message_truncated_to_500(self) -> None:
        session = MagicMock()
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            ToolUsageRepository().log_usage(
                tool_name="t", role=None, latency_ms=1, success=False,
                error_message="x" * 2000,
            )
        added = session.add.call_args.args[0]
        assert added.error_message is not None
        assert len(added.error_message) <= 500

    def test_db_exception_is_swallowed(self) -> None:
        def boom(*_a, **_kw):
            raise RuntimeError("db down")
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            side_effect=boom,
        ):
            # Must not raise — telemetry is best-effort.
            ToolUsageRepository().log_usage(
                tool_name="t", role=None, latency_ms=1, success=True,
            )


class TestGetStats:
    def _build_session(self, total, by_tool, by_role):
        """Return a MagicMock whose .query(...).filter(...).group_by(...).all()
        chains return the expected aggregate rows."""
        session = MagicMock()

        # total-count query path — .query(count).filter(...).scalar()
        count_q = MagicMock()
        count_q.filter.return_value.scalar.return_value = total

        # by_tool aggregate
        tool_q = MagicMock()
        tool_q.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = by_tool

        # by_role aggregate
        role_q = MagicMock()
        role_q.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = by_role

        # query() is called 3 times: count → tool → role
        session.query.side_effect = [count_q, tool_q, role_q]
        return session

    def test_default_window_is_7_days(self) -> None:
        session = self._build_session(
            total=5,
            by_tool=[("get_system_overview", 5, 120.0, 5)],
            by_role=[("admin", 5)],
        )
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            stats = ToolUsageRepository().get_stats()
        assert stats["window_days"] == 7
        assert stats["total_calls"] == 5
        assert stats["by_tool"][0]["tool_name"] == "get_system_overview"
        assert stats["by_tool"][0]["success_rate"] == 1.0
        assert stats["by_role"] == [{"role": "admin", "count": 5}]

    def test_custom_days_passed_through(self) -> None:
        session = self._build_session(total=0, by_tool=[], by_role=[])
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            stats = ToolUsageRepository().get_stats(days=30)
        assert stats["window_days"] == 30
        assert stats["total_calls"] == 0

    def test_null_role_rendered_as_unknown(self) -> None:
        session = self._build_session(
            total=3,
            by_tool=[("t", 3, 50.0, 2)],
            by_role=[(None, 3)],
        )
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            return_value=_FakeSessionCtx(session),
        ):
            stats = ToolUsageRepository().get_stats(days=1)
        assert stats["by_role"][0]["role"] == "unknown"
        # success_rate = 2/3
        assert stats["by_tool"][0]["success_rate"] == pytest.approx(0.6667, abs=1e-4)

    def test_db_exception_returns_empty_shape(self) -> None:
        with patch(
            "app.repositories.solar_ai_chat.tool_usage_repository.SessionLocal",
            side_effect=RuntimeError("down"),
        ):
            stats = ToolUsageRepository().get_stats(days=14)
        assert stats == {"window_days": 14, "total_calls": 0, "by_tool": [], "by_role": []}
