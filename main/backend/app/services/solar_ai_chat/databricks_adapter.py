"""Solar AI Chat — Databricks adapter that wires BaseRepository's
connection into the engine primitives.

Why an adapter: BaseRepository owns connection management. Rather than
duplicating that logic, we wrap an existing repository instance and
expose its `_execute_query` as a `Callable[[str], list[dict]]` the
primitives accept.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.base_repository import BaseRepository

logger = logging.getLogger(__name__)


def make_sql_executor(
    settings: SolarChatSettings,
) -> Callable[[str], list[dict[str, Any]]]:
    """Return a callable suitable for primitives.execute_sql.

    The callable runs SQL via BaseRepository._databricks_connection and
    returns column-keyed row dicts. Errors propagate as exceptions so the
    primitive can format them for the LLM.
    """
    repo = BaseRepository(settings=settings)

    def _executor(sql: str) -> list[dict[str, Any]]:
        return repo._execute_query(sql)  # noqa: SLF001 — intentional re-use

    return _executor


def make_sample_executor(
    settings: SolarChatSettings,
    *,
    max_sample_rows: int = 3,
) -> Callable[[str], list[dict[str, Any]]]:
    """Specialised executor for inspect_table sample queries.
    Limits sample row count and swallows errors (returning empty list)
    so an offline warehouse doesn't break schema inspection."""
    base_exec = make_sql_executor(settings)

    def _sample(sql: str) -> list[dict[str, Any]]:
        try:
            rows = base_exec(sql)
            return rows[:max_sample_rows]
        except Exception as exc:  # noqa: BLE001
            logger.info("inspect_table_sample_skipped error=%s", exc)
            return []

    return _sample
