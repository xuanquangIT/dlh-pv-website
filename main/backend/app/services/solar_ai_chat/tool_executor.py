import logging
from datetime import date
from typing import Any

from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.tools import TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient

logger = logging.getLogger(__name__)

WEATHER_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "temperature_2m": ("temperature", "C"),
    "wind_speed_10m": ("wind speed", "m/s"),
    "wind_gusts_10m": ("wind gust", "m/s"),
    "shortwave_radiation": ("shortwave radiation", "W/m2"),
    "cloud_cover": ("cloud cover", "%"),
}

_TOOL_PERMISSIONS: dict[ChatRole, set[str]] = {
    ChatRole.DATA_ENGINEER: {
        "get_system_overview", "get_energy_performance",
        "get_pipeline_status", "get_forecast_72h", "get_data_quality_issues",
        "get_extreme_aqi", "get_extreme_energy", "get_extreme_weather",
        "get_station_daily_report", "search_documents",
    },
    ChatRole.ML_ENGINEER: {
        "get_system_overview", "get_energy_performance",
        "get_ml_model_info", "get_forecast_72h",
        "get_extreme_energy", "get_extreme_weather",
        "get_station_daily_report", "search_documents",
    },
    ChatRole.DATA_ANALYST: {
        "get_system_overview", "get_energy_performance",
        "get_ml_model_info", "get_forecast_72h", "get_data_quality_issues",
        "get_extreme_aqi", "get_extreme_energy", "get_extreme_weather",
        "get_station_daily_report", "search_documents",
    },
    ChatRole.VIEWER: {
        "get_system_overview", "get_energy_performance",
        "get_forecast_72h",
        "get_extreme_energy", "get_extreme_weather",
        "get_station_daily_report", "search_documents",
    },
    ChatRole.ADMIN: {
        "get_system_overview", "get_energy_performance",
        "get_ml_model_info", "get_pipeline_status",
        "get_forecast_72h", "get_data_quality_issues",
        "get_extreme_aqi", "get_extreme_energy", "get_extreme_weather",
        "get_station_daily_report", "search_documents",
    },
}


class ToolExecutor:
    """Dispatcher that maps Gemini function calls to repository methods."""

    def __init__(
        self,
        repository: SolarChatRepository,
        vector_repo: VectorRepository | None = None,
        embedding_client: GeminiEmbeddingClient | None = None,
    ) -> None:
        self._repository = repository
        self._vector_repo = vector_repo
        self._embedding_client = embedding_client

    def execute(
        self,
        function_name: str,
        arguments: dict[str, Any],
        role: ChatRole,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        topic_handlers = {
            "get_system_overview": self._get_topic_metrics,
            "get_energy_performance": self._get_topic_metrics,
            "get_ml_model_info": self._get_topic_metrics,
            "get_pipeline_status": self._get_topic_metrics,
            "get_forecast_72h": self._get_topic_metrics,
            "get_data_quality_issues": self._get_topic_metrics,
            "get_extreme_aqi": self._get_extreme_aqi,
            "get_extreme_energy": self._get_extreme_energy,
            "get_extreme_weather": self._get_extreme_weather,
            "get_station_daily_report": self._get_station_daily_report,
            "search_documents": self._search_documents,
        }

        handler = topic_handlers.get(function_name)
        if handler is None:
            raise ValueError(f"Unknown tool: '{function_name}'.")

        self._validate_permission(function_name, role)

        if function_name in TOOL_NAME_TO_TOPIC and handler == self._get_topic_metrics:
            topic = ChatTopic(TOOL_NAME_TO_TOPIC[function_name])
            return self._repository.fetch_topic_metrics(topic)

        return handler(arguments)

    @staticmethod
    def _validate_permission(function_name: str, role: ChatRole) -> None:
        allowed = _TOOL_PERMISSIONS.get(role, set())
        if function_name not in allowed:
            topic_value = TOOL_NAME_TO_TOPIC.get(function_name, function_name)
            raise PermissionError(
                f"Role '{role.value}' is not allowed to access topic '{topic_value}'."
            )

    def _get_topic_metrics(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        raise RuntimeError("Should not be called directly.")

    def _get_extreme_aqi(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._repository.fetch_extreme_aqi(
            query_type=arguments["query_type"],
            timeframe=arguments["timeframe"],
            anchor_date=self._parse_date(arguments.get("anchor_date")),
            specific_hour=arguments.get("specific_hour"),
        )

    def _get_extreme_energy(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        return self._repository.fetch_extreme_energy(
            query_type=arguments["query_type"],
            timeframe=arguments["timeframe"],
            anchor_date=self._parse_date(arguments.get("anchor_date")),
            specific_hour=arguments.get("specific_hour"),
        )

    def _get_extreme_weather(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        metric_key = arguments.get("weather_metric", "temperature_2m")
        label, unit = WEATHER_METRIC_LABELS.get(metric_key, (metric_key, ""))
        return self._repository.fetch_extreme_weather(
            query_type=arguments["query_type"],
            timeframe=arguments["timeframe"],
            anchor_date=self._parse_date(arguments.get("anchor_date")),
            weather_metric=metric_key,
            weather_metric_label=label,
            weather_unit=unit,
            specific_hour=arguments.get("specific_hour"),
        )

    def _get_station_daily_report(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        anchor = self._parse_date(arguments.get("anchor_date"))
        if not anchor:
            anchor = date.today()
        metrics = arguments.get("metrics")
        return self._repository.fetch_station_daily_report(
            anchor_date=anchor,
            metrics=metrics,
        )

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    def _search_documents(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if not self._vector_repo or not self._embedding_client:
            return (
                {"message": "RAG is not configured. Document search is unavailable."},
                [{"layer": "rag", "dataset": "rag_documents", "data_source": "unavailable"}],
            )

        query_text = arguments.get("query", "")
        doc_type = arguments.get("doc_type")

        query_embedding = self._embedding_client.embed_text(query_text)
        chunks = self._vector_repo.search_similar(
            query_embedding=query_embedding,
            top_k=5,
            doc_type=doc_type,
        )

        if not chunks:
            return (
                {"message": "Khong tim thay tai lieu phu hop.", "chunks": []},
                [{"layer": "rag", "dataset": "rag_documents", "data_source": "pgvector"}],
            )

        metrics: dict[str, Any] = {
            "chunks": [
                {
                    "content": c.content,
                    "source_file": c.source_file,
                    "doc_type": c.doc_type,
                    "score": round(c.similarity_score, 4),
                }
                for c in chunks
            ],
            "total_results": len(chunks),
        }
        sources = [
            {
                "layer": "rag",
                "dataset": c.source_file,
                "data_source": "pgvector",
            }
            for c in chunks
        ]
        return metrics, sources
