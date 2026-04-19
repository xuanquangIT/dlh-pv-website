"""SolarChatRepository — public facade combining all repository layers.

Architecture:
    BaseRepository          shared infrastructure (Trino, CSV, helpers)
        ExtremeRepository   fetch_extreme_aqi/energy/weather
        TopicRepository     fetch_topic_metrics + 6 topic handlers
        ReportRepository    fetch_station_daily_report
        KpiRepository       fetch_gold_kpi for dynamic mart tables

All business-facing code should depend on SolarChatRepository only.
"""
from __future__ import annotations

from typing import Any

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.extreme_repository import ExtremeRepository
from app.repositories.solar_ai_chat.report_repository import ReportRepository
from app.repositories.solar_ai_chat.topic_repository import TopicRepository
from app.repositories.solar_ai_chat.kpi_repository import KpiRepository
from app.schemas.solar_ai_chat.enums import ChatTopic


class SolarChatRepository(ExtremeRepository, TopicRepository, ReportRepository, KpiRepository):
    """Read-only analytics repository for the Solar AI Chat module.

    Queries Silver and Gold data layers via Trino, falling back to local
    CSV files when Trino is unreachable.
    """

    def __init__(self, settings: SolarChatSettings) -> None:
        super().__init__(settings)

    def fetch_topic_metrics(
        self, topic: ChatTopic, arguments: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if topic is ChatTopic.GENERAL:
            return self._general_greeting()
        handlers = {
            ChatTopic.SYSTEM_OVERVIEW: lambda: self._system_overview(arguments),
            ChatTopic.ENERGY_PERFORMANCE: lambda: self._energy_performance(arguments),
            ChatTopic.ML_MODEL: self._ml_model,
            ChatTopic.PIPELINE_STATUS: self._pipeline_status,
            ChatTopic.FORECAST_72H: self._forecast_72h,
            ChatTopic.DATA_QUALITY_ISSUES: lambda: self._data_quality_issues(arguments),
            ChatTopic.FACILITY_INFO: self._facility_info,
        }
        return handlers[topic]()