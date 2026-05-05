"""SolarChatRepository — public facade for Solar AI Chat data access.

Phase 4 cleanup: the v1 mixin chain (ExtremeRepository / TopicRepository /
ReportRepository / KpiRepository) was removed because the engine no
longer dispatches via topic-bound `fetch_*` helpers — it composes
`execute_sql` over the YAML semantic layer instead. Only `BaseRepository`
infrastructure (Databricks SQL connection + safe row materialisation)
remains, which the engine's `databricks_adapter.py` consumes.
"""
from __future__ import annotations

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.base_repository import BaseRepository


class SolarChatRepository(BaseRepository):
    """Thin facade exposing BaseRepository to the engine dispatcher.

    Kept as a class (rather than aliasing BaseRepository directly) so callers
    can construct it with `SolarChatRepository(settings=settings)` per the
    existing factory in `api/solar_ai_chat/routes.py`.
    """

    def __init__(self, settings: SolarChatSettings) -> None:
        super().__init__(settings)
