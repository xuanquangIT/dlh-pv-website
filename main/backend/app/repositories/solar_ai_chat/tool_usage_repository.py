"""Persistence + aggregation for chat_tool_usage (Task 0.1).

Every insert is best-effort — a DB failure here must never surface to the
user-facing chat flow. Aggregates are used only by the admin analytics
endpoint to answer "which tools are actually being called, by whom,
how fast, how often?" so we can make evidence-based cuts in later phases.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func

from app.db.database import ChatToolUsage, SessionLocal

logger = logging.getLogger(__name__)


class ToolUsageRepository:
    """Insert + aggregate chat_tool_usage rows."""

    def log_usage(
        self,
        *,
        tool_name: str,
        role: str | None,
        latency_ms: int,
        success: bool,
        session_id: str | None = None,
        user_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Insert one telemetry row. Never raises."""
        try:
            with SessionLocal() as db:
                row = ChatToolUsage(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    user_id=uuid.UUID(user_id) if user_id else None,
                    role=role,
                    tool_name=tool_name,
                    latency_ms=max(0, int(latency_ms)),
                    success=bool(success),
                    error_message=(error_message or "")[:500] or None,
                )
                db.add(row)
                db.commit()
        except Exception as exc:  # pragma: no cover - pure best-effort
            logger.warning(
                "chat_tool_usage_log_failed tool=%s error_type=%s error=%s",
                tool_name, type(exc).__name__, exc,
            )

    def get_stats(self, *, days: int = 7) -> dict[str, Any]:
        """Return aggregated usage for the last ``days`` days.

        Shape:
            {
              "window_days": 7,
              "total_calls": 1234,
              "by_tool": [{"tool_name": "...", "count": N,
                           "avg_latency_ms": float, "success_rate": float}, ...],
              "by_role": [{"role": "...", "count": N}, ...],
            }
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max(1, int(days)))
        result: dict[str, Any] = {
            "window_days": int(days),
            "total_calls": 0,
            "by_tool": [],
            "by_role": [],
        }
        try:
            with SessionLocal() as db:
                total = (
                    db.query(func.count(ChatToolUsage.id))
                    .filter(ChatToolUsage.created_at >= cutoff)
                    .scalar()
                ) or 0

                by_tool_rows = (
                    db.query(
                        ChatToolUsage.tool_name,
                        func.count(ChatToolUsage.id).label("cnt"),
                        func.avg(ChatToolUsage.latency_ms).label("avg_latency"),
                        func.sum(
                            # success → 1, failure → 0
                            func.cast(ChatToolUsage.success, ChatToolUsage.latency_ms.type)
                        ).label("success_count"),
                    )
                    .filter(ChatToolUsage.created_at >= cutoff)
                    .group_by(ChatToolUsage.tool_name)
                    .order_by(func.count(ChatToolUsage.id).desc())
                    .all()
                )
                by_tool = []
                for tool_name, cnt, avg_latency, success_count in by_tool_rows:
                    c = int(cnt or 0)
                    by_tool.append({
                        "tool_name": tool_name,
                        "count": c,
                        "avg_latency_ms": round(float(avg_latency or 0.0), 2),
                        "success_rate": round(
                            (float(success_count or 0) / c) if c else 0.0, 4,
                        ),
                    })

                by_role_rows = (
                    db.query(
                        ChatToolUsage.role,
                        func.count(ChatToolUsage.id).label("cnt"),
                    )
                    .filter(ChatToolUsage.created_at >= cutoff)
                    .group_by(ChatToolUsage.role)
                    .order_by(func.count(ChatToolUsage.id).desc())
                    .all()
                )
                by_role = [
                    {"role": role or "unknown", "count": int(cnt or 0)}
                    for role, cnt in by_role_rows
                ]

                result["total_calls"] = int(total)
                result["by_tool"] = by_tool
                result["by_role"] = by_role
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "chat_tool_usage_stats_failed error_type=%s error=%s",
                type(exc).__name__, exc,
            )
        return result
