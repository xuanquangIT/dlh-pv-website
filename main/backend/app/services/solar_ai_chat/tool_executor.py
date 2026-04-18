import logging
from datetime import date
from typing import Any

from app.repositories.solar_ai_chat.base_repository import DatabricksDataUnavailableError
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat.agent import ToolResultEnvelope
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.tools import TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient
from app.services.solar_ai_chat.permissions import ROLE_TOOL_PERMISSIONS

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Weather metric label/unit lookup — shared with chat_service._WEATHER_METRIC_CATALOG
WEATHER_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "temperature_2m": ("temperature", "C"),
    "wind_speed_10m": ("wind speed", "m/s"),
    "wind_gusts_10m": ("wind gust", "m/s"),
    "shortwave_radiation": ("shortwave radiation", "W/m2"),
    "cloud_cover": ("cloud cover", "%"),
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

    def execute_envelope(
        self,
        function_name: str,
        arguments: dict[str, Any],
        role: ChatRole,
    ) -> ToolResultEnvelope:
        """Execute a tool and return a standardised ToolResultEnvelope.

        This is the preferred method for the new orchestrator path.
        The legacy ``execute()`` method delegates here for backward compatibility.
        """
        try:
            data, sources = self.execute(function_name, arguments, role)
            return ToolResultEnvelope(
                status="ok",
                data=data,
                sources=sources,
                confidence=1.0,
                errors=[],
                tool_name=function_name,
            )
        except PermissionError as exc:
            return ToolResultEnvelope(
                status="error",
                data={},
                sources=[],
                confidence=0.0,
                errors=[str(exc)],
                tool_name=function_name,
            )
        except DatabricksDataUnavailableError:
            raise
        except Exception as exc:
            logger.exception(
                "tool_executor_envelope_error tool=%s error=%s", function_name, exc
            )
            return ToolResultEnvelope(
                status="error",
                data={},
                sources=[],
                confidence=0.0,
                errors=[str(exc)],
                tool_name=function_name,
            )

    def execute(
        self,
        function_name: str,
        arguments: dict[str, Any],
        role: ChatRole,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        logger.info(
            "solar_chat_tool_executor_start tool=%s role=%s args=%s",
            function_name,
            role.value,
            self._short(arguments, 300),
        )
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
            "get_station_hourly_report": self._get_station_hourly_report,
            "get_facility_info": self._get_facility_info,
            "search_documents": self._search_documents,
        }

        handler = topic_handlers.get(function_name)
        if handler is None:
            raise ValueError(f"Unknown tool: '{function_name}'.")

        self._validate_permission(function_name, role)

        if function_name in TOOL_NAME_TO_TOPIC and handler == self._get_topic_metrics:
            topic = ChatTopic(TOOL_NAME_TO_TOPIC[function_name])
            metrics, sources = self._repository.fetch_topic_metrics(topic)
            logger.info(
                "solar_chat_tool_executor_done tool=%s topic=%s metric_keys=%s source_count=%d",
                function_name,
                topic.value,
                sorted(list(metrics.keys())),
                len(sources),
            )
            return metrics, sources

        metrics, sources = handler(arguments)
        logger.info(
            "solar_chat_tool_executor_done tool=%s metric_keys=%s source_count=%d",
            function_name,
            sorted(list(metrics.keys())),
            len(sources),
        )
        return metrics, sources

    @staticmethod
    def _short(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _validate_permission(function_name: str, role: ChatRole) -> None:
        allowed = ROLE_TOOL_PERMISSIONS.get(role, set())
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

    def _get_station_hourly_report(
        self, arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        anchor = self._parse_date(arguments.get("anchor_date"))
        station_name = arguments.get("station_name")
        return self._repository.fetch_station_hourly_report(
            station_name=station_name,
            anchor_date=anchor,
        )

    def _get_station_daily_report(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        anchor = self._parse_date(arguments.get("anchor_date"))
        if not anchor:
            try:
                anchor = self._repository._resolve_latest_date("silver.energy_readings")
            except Exception:
                anchor = date.today()
        metrics = arguments.get("metrics")
        station_name = arguments.get("station_name")
        return self._repository.fetch_station_daily_report(
            anchor_date=anchor,
            metrics=metrics,
            station_name=station_name,
        )

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    def _get_facility_info(
        self, arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Return facility details including location and capacity."""
        facility_name = arguments.get("facility_name")
        result = None
        if hasattr(self._repository, "_facility_info"):
            result = self._repository._facility_info(
                facility_name=facility_name,
            )

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], dict)
            and isinstance(result[1], list)
        ):
            return result

        # Backward compatibility for tests/mocks that expose only topic-level metrics.
        return self._repository.fetch_topic_metrics(ChatTopic.FACILITY_INFO)

    def _search_documents(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if not self._vector_repo or not self._embedding_client:
            return (
                {"message": "RAG is not configured. Document search is unavailable."},
                [{"layer": "rag", "dataset": "rag_documents", "data_source": "unavailable"}],
            )

        query_text = str(arguments.get("query", "") or "").strip()
        doc_type_raw = arguments.get("doc_type")
        doc_type = str(doc_type_raw).strip() if isinstance(doc_type_raw, str) else None

        if not query_text:
            # Avoid embedding API calls with empty input; treat as an empty retrieval result.
            return (
                {
                    "message": "Empty search query. Skipped document search.",
                    "chunks": [],
                    "total_results": 0,
                },
                [{"layer": "rag", "dataset": "rag_documents", "data_source": "pgvector"}],
            )

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
