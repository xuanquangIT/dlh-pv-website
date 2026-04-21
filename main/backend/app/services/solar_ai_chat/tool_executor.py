import logging
import re
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


_CORRELATION_KEYWORDS = (
    "mối liên hệ", "liên hệ giữa", "mối tương quan", "tương quan giữa",
    "correlation", "relationship between",
    " vs ", " vs.", "versus",
    "ảnh hưởng", "affects", "affect",
    "so sánh", "compared to",
    "how does", "how do",
)
_METRIC_KEYWORDS = (
    "performance ratio", "performance_ratio", "pr ", " pr",
    "capacity factor", "capacity_factor",
    "hiệu suất",        # Vietnamese: efficiency / performance (covers "hiệu suất phát điện")
    "phát điện",        # Vietnamese: power generation / electricity output
    "sản xuất điện",    # Vietnamese: electricity production
    "energy", "năng lượng", "sản lượng",
    "temperature", "nhiệt độ",
    "aqi", "air quality", "chất lượng không khí",
    "cloud", "mây", "bức xạ", "radiation",
    "humidity", "độ ẩm",
    "wind", "gió", "tốc độ gió",   # covers "tốc độ và hướng gió"
)


_TOP_N_PATTERN = re.compile(
    r"top\s*(\d+)"                  # "top 5", "top5"
    r"|bottom\s*(\d+)"              # "bottom 3"
    r"|(\d+)\s*(?:trạm|cơ sở|facility|facilities|nhà máy|stations?)"  # "5 trạm"
    r"|(?:trạm|facility|facilities|nhà máy)\s*(?:top\s*)?(\d+)",       # "facility top 5"
    re.IGNORECASE,
)


def _extract_top_n_from_query(text: str) -> int | None:
    """Return the N from 'top N', 'bottom N', 'N trạm' patterns, or None."""
    m = _TOP_N_PATTERN.search(str(text or ""))
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            try:
                return max(1, int(g))
            except ValueError:
                pass
    return None


def _is_correlation_query(text: str) -> bool:
    """True when the user clearly asks for correlation/relationship between
    two domain metrics. Used to gate tool selection so only the one tool that
    returns paired X-Y data (query_gold_kpi on mart_energy_daily) is allowed.
    """
    if not text:
        return False
    t = str(text).lower()
    has_corr = any(k in t for k in _CORRELATION_KEYWORDS)
    if not has_corr:
        return False
    metric_hits = sum(1 for k in _METRIC_KEYWORDS if k in t)
    return metric_hits >= 2  # at least two metrics named → correlation


def _infer_correlation_table(text: str) -> str:
    """Pick the gold KPI mart table most suited for a correlation query.

    Routing rules (checked in order):
    - AQI / air quality       → aqi_impact    (mart_aqi_impact_daily)
    - Wind speed / humidity   → weather_impact (mart_weather_impact_daily)
      mart_energy_daily has NO avg_wind_speed_ms or avg_humidity_pct — only the
      weather_impact mart carries these alongside avg_capacity_factor_pct.
    - Cloud / temperature / radiation vs PR → energy (mart_energy_daily)
      has performance_ratio_pct + avg_cloud_cover_pct + avg_temperature_c.
    """
    t = str(text or "").lower()
    if any(k in t for k in (
        "aqi", "air quality", "chất lượng không khí", "không khí",
        "pm2.5", "pm10", "bụi mịn",
    )):
        return "aqi_impact"
    if any(k in t for k in (
        "wind", "gió", "tốc độ gió", "hướng gió",
        "humidity", "độ ẩm",
    )):
        return "weather_impact"
    return "energy"


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
        self._current_user_query: str = ""

    def set_user_query(self, query: str) -> None:
        """Called by chat_service at the start of each request so per-tool
        dispatch can gate routing based on the user's original question."""
        self._current_user_query = str(query or "")

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
            "query_gold_kpi": self._query_gold_kpi,
        }

        handler = topic_handlers.get(function_name)
        if handler is None:
            raise ValueError(f"Unknown tool: '{function_name}'.")

        self._validate_permission(function_name, role)

        # Top-level safety net: a correlation query MUST go through
        # query_gold_kpi('energy'). Every other data tool returns a shape
        # that can't produce a meaningful X-Y scatter, and the chart ends
        # up decoupled from the narrative. Instead of raising (which the LLM
        # tends to give up on), we transparently re-route the call so the
        # right data is fetched regardless of what the LLM picked.
        if (
            _is_correlation_query(self._current_user_query)
            and function_name not in ("query_gold_kpi",)
        ):
            function_name = "query_gold_kpi"
            _corr_table = _infer_correlation_table(self._current_user_query)
            arguments = {"table_name": _corr_table, "limit": 100}
            logger.warning(
                "tool_routing_rerouted original_tool=%s -> query_gold_kpi('%s') "
                "reason=correlation_query user_query=%r original_args=%s",
                function_name,
                _corr_table,
                self._current_user_query[:200],
                self._short(arguments, 200),
            )
            handler = topic_handlers.get(function_name)
            # Re-validate permission under the rerouted tool name — the user
            # may have had access to the original tool but not to query_gold_kpi.
            self._validate_permission(function_name, role)

        # Safety net: inject `limit` for get_energy_performance when the user
        # says "top N" but the LLM forgot to pass limit in the tool call.
        if function_name == "get_energy_performance" and not arguments.get("limit"):
            inferred = _extract_top_n_from_query(self._current_user_query)
            if inferred is not None:
                arguments = {**arguments, "limit": inferred}
                logger.info(
                    "tool_limit_injected tool=get_energy_performance limit=%d "
                    "reason=llm_omitted_limit user_query=%r",
                    inferred,
                    self._current_user_query[:200],
                )

        # Safety net: block `get_station_daily_report` when the LLM requests
        # performance_ratio (the tool doesn't return PR and the chart comes
        # out unrelated). Force the LLM to retry with the correct tool.
        if function_name == "get_station_daily_report":
            requested_metrics = arguments.get("metrics") or []
            if isinstance(requested_metrics, str):
                requested_metrics = [requested_metrics]
            if any(
                "performance_ratio" in str(m).lower() or str(m).lower() in {"pr", "pr_pct", "pr_ratio"}
                for m in requested_metrics
            ):
                logger.warning(
                    "tool_routing_block tool=get_station_daily_report "
                    "reason=performance_ratio_not_supported args=%s",
                    self._short(arguments, 200),
                )
                raise ValueError(
                    "get_station_daily_report does not return performance_ratio. "
                    "For PR correlation / relationship queries call "
                    "query_gold_kpi(table_name='energy', limit=100) instead — "
                    "that table has per-facility-per-day rows with "
                    "performance_ratio_pct + avg_temperature_c."
                )

        # Safety net: block `search_documents` when the query clearly asks for
        # numeric/correlation data. RAG returns text chunks — no chart will be
        # generated and the answer ends up hand-wavy.
        if function_name == "search_documents":
            q = str(arguments.get("query") or "").lower()
            # Also check original user query — LLM sometimes passes a reworded
            # search string that drops the correlation keyword.
            user_q = self._current_user_query.lower()
            correlation_terms = (
                "mối liên hệ", "tương quan", "correlation",
                " vs ", " vs.", "versus",
                "so sánh", "compare",
                "top", "ranking", "ranked",
                "cao nhất", "thấp nhất", "highest", "lowest",
                "ảnh hưởng", "affects", "affect", "how does", "how do",
            )
            metric_terms = (
                "performance ratio", "performance_ratio",
                "capacity factor", "capacity_factor",
                "hiệu suất",    # Vietnamese for efficiency/performance
                "phát điện",    # power generation
                "energy_mwh", "mwh", "năng lượng",
                "wind", "gió", "tốc độ gió",
                "temperature", "nhiệt độ",
                "cloud", "mây", "humidity", "độ ẩm",
                "aqi", "radiation", "bức xạ",
                "trạm nào",
            )
            if (
                any(t in q for t in correlation_terms) or any(t in user_q for t in correlation_terms)
            ) and (
                any(t in q for t in metric_terms) or any(t in user_q for t in metric_terms)
            ):
                logger.warning(
                    "tool_routing_block tool=search_documents "
                    "reason=correlation_or_ranking_of_metrics query=%r",
                    q[:200],
                )
                raise ValueError(
                    "search_documents only returns text chunks from manuals / "
                    "incident reports / model changelogs — it has no numeric "
                    "data and cannot produce a chart. For correlation or "
                    "ranking of metrics call query_gold_kpi(table_name='energy', "
                    "limit=100) instead. The `energy` mart has per-facility-"
                    "per-day rows with performance_ratio_pct, capacity_factor_pct, "
                    "energy_mwh_daily, avg_temperature_c."
                )

        if function_name in TOOL_NAME_TO_TOPIC and handler == self._get_topic_metrics:
            topic = ChatTopic(TOOL_NAME_TO_TOPIC[function_name])
            if arguments:
                metrics, sources = self._repository.fetch_topic_metrics(topic, arguments)
            else:
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

    def _query_gold_kpi(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if not hasattr(self._repository, "fetch_gold_kpi"):
             return {"message": "KPI Mart querying is not supported by the current repository."}, []

        raw_limit = arguments.get("limit", 30)
        try:
            safe_limit = int(raw_limit) if raw_limit is not None else 30
        except (TypeError, ValueError):
            safe_limit = 30

        return self._repository.fetch_gold_kpi(
            table_short_name=arguments.get("table_name", ""),
            anchor_date=arguments.get("anchor_date"),
            station_name=arguments.get("station_name"),
            limit=safe_limit,
        )
