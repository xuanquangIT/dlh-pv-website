"""Solar AI Chat service — orchestration layer.

Coordinates agentic tool-calling, RBAC, data retrieval, LLM response generation,
and response finalization.  NLP parsing is in nlp_parser.py; prompt and
architecture context is in prompt_builder.py.

Architecture: Native LLM tool-calling (ReAct pattern)
------------------------------------------------------
1. build_agentic_messages() injects the full Lakehouse architecture context.
2. The LLM (generate_with_tools) decides which tools to call.
3. Each tool response is appended to the conversation thread.
4. The LLM synthesises the final answer after collecting all tool results.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any
from unicodedata import normalize

from app.repositories.solar_ai_chat.base_repository import DatabricksDataUnavailableError
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatTopic,
    SolarChatRequest,
    SolarChatResponse,
    SourceMetadata,
    ThinkingStep,
    ThinkingTrace,
    resolve_ui_features,
    tool_label,
)
from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS, TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.llm_client import (
    LLMModelRouter,
    ModelUnavailableError,
    ToolCallNotSupportedError,
)
from app.services.solar_ai_chat.intent_service import (
    VietnameseIntentService,
    normalize_vietnamese_text,
)
from app.services.solar_ai_chat.nlp_parser import (
    ExtremeMetricQuery,
    extract_extreme_metric_query,
    extract_query_date,
    extract_timeframe,
    expand_facility_codes_in_message,
)
from app.services.solar_ai_chat.permissions import ROLE_TOPIC_PERMISSIONS, ROLE_TOOL_PERMISSIONS
from app.services.solar_ai_chat.prompt_builder import (
    build_agentic_messages,
    build_data_only_summary,
    build_synthesis_prompt,
    build_insufficient_data_response,
    format_source_text,
)
from app.services.solar_ai_chat.query_rewriter import QueryRewriter
from app.services.solar_ai_chat.chart_service import ChartSpecBuilder
from app.services.solar_ai_chat.tool_executor import ToolExecutor, ToolRoutingBlockedError
from app.services.solar_ai_chat.answer_verifier import AnswerVerifier
from app.services.solar_ai_chat.deep_planner import DeepPlanner
from app.schemas.solar_ai_chat.agent import EvidenceItem, EvidenceStore

if TYPE_CHECKING:
    from app.repositories.solar_ai_chat.vector_repository import VectorRepository
    from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

TOOL_PATH_CONFIDENCE = 0.95
MAX_TOOL_STEPS = 6  # Increased from 2 to allow multi-tool agentic queries
MAX_PLANNED_ACTIONS = 5
MAX_REFLECTION_PASSES = 1

# Keywords that signal the user explicitly wants a live web search.
_WEB_SEARCH_KEYWORDS = (
    "internet", "web", "google", "tìm kiếm", "search", "tra cứu",
    "tìm trên", "online",
)

# Prefixes to strip when extracting the actual search query (longest first).
_WEB_SEARCH_STRIP_PREFIXES: tuple[str, ...] = (
    "search internet", "tìm trên internet", "tim tren internet",
    "tìm kiếm trên internet", "tim kiem tren internet",
    "tìm kiếm trên web", "tim kiem tren web",
    "search web", "tìm trên web", "tim tren web",
    "google search", "tra cứu trên web", "tìm online", "tim online",
    "tìm kiếm", "tim kiem", "google", "tra cứu", "search",
)

# Vietnamese filler words to strip from the front of the extracted query.
# Ordered longest-first so compound forms match before their fragments.
_QUERY_FILLER_PREFIXES: tuple[str, ...] = (
    "để cho tôi", "de cho toi", "để tôi xem", "de toi xem",
    "để tôi", "de toi", "xem về", "xem",
    "về", "cho tôi", "cho toi", "cho mình", "cho minh",
    "để", "de",
)

_PROMPT_INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous", "ignore all", "bo qua huong dan", "bo qua chi dan",
    "reveal system", "system prompt", "developer prompt", "secret key",
    "api key", "token", "jailbreak", "override instructions",
)


def _needs_web_search(message: str) -> bool:
    """Web search is disabled. Stub kept so existing ``not _needs_web_search(...)``
    guards still compile; they all now evaluate to True."""
    _ = message  # noqa: B007
    return False


def _merge_tool_result(
    all_metrics: dict[str, Any],
    tool_name: str,
    data: dict[str, Any],
) -> None:
    """Accumulate tool results across repeated calls instead of overwriting.

    The agentic loop can call the same tool several times with different
    arguments (e.g. daily reports for each day of a week). A naive
    ``all_metrics.update(data)`` drops every prior day because the top-level
    keys (``stations``, ``report_date``, ``has_data``, …) collide.

    For tools that return a per-date slice, we keep a flat ``stations`` list
    with ``report_date`` stamped on each row, plus a ``daily_station_reports``
    list of the raw per-date payloads for downstream rendering.
    """
    if tool_name == "get_station_daily_report":
        incoming_date = data.get("report_date")
        incoming_stations = list(data.get("stations") or [])
        incoming_has_data = bool(data.get("has_data"))

        history = all_metrics.get("daily_station_reports")
        prior_date = all_metrics.get("report_date")
        prior_stations = all_metrics.get("stations")

        if history is None and prior_date and prior_stations is not None and prior_date != incoming_date:
            history = [{
                "report_date": prior_date,
                "stations": list(prior_stations) if isinstance(prior_stations, list) else [],
                "has_data": bool(all_metrics.get("has_data", True)),
            }]

        if history is not None:
            history.append({
                "report_date": incoming_date,
                "stations": incoming_stations,
                "has_data": incoming_has_data,
            })
            flat: list[dict[str, Any]] = []
            unique_facilities: set[str] = set()
            for entry in history:
                rd = entry.get("report_date")
                for s in entry.get("stations") or []:
                    flat.append({**s, "report_date": rd})
                    name = s.get("facility") or s.get("facility_name")
                    if name:
                        unique_facilities.add(str(name))
            all_metrics["daily_station_reports"] = history
            all_metrics["stations"] = flat
            all_metrics["station_count"] = len(unique_facilities)
            all_metrics["day_count"] = sum(1 for e in history if e.get("has_data"))
            all_metrics["row_count"] = len(flat)
            all_metrics["has_data"] = any(e.get("has_data") for e in history)
            for key, value in data.items():
                if key not in {"stations", "station_count", "has_data", "report_date", "no_data_reason"}:
                    all_metrics[key] = value
            all_metrics["report_date_range"] = sorted(
                {e["report_date"] for e in history if e.get("report_date")}
            )
            return

    all_metrics.update(data)


_VISUALIZE_KEYWORDS: tuple[str, ...] = (
    # Explicit viz requests
    "visualize", "visualise", "visualization", "visualisation",
    "chart", "charts", "plot", "graph", "graphs", "diagram",
    "biểu đồ", "bieu do", "vẽ", "ve ",
    "trực quan", "truc quan", "hình ảnh hóa", "hinh anh hoa",
    "đồ thị", "do thi",
)

# Implicit viz signals — queries that clearly benefit from a chart even if
# the user didn't type "chart/plot/graph" explicitly. Comparison, ranking,
# distribution, and trend-over-time questions all expect a visual payoff.
_IMPLICIT_VIZ_KEYWORDS: tuple[str, ...] = (
    # Comparison / ranking
    "compare", "comparison", "versus", " vs ", " vs.",
    "top ", "bottom ", "best ", "worst ", "highest", "lowest",
    "ranking", "rank ", "rankings",
    "per facility", "by facility", "across facilities", "across all",
    "all facilities", "each facility",
    "so sánh", "so sanh", "xếp hạng", "xep hang",
    "cao nhất", "cao nhat", "thấp nhất", "thap nhat",
    # Time series / trend
    "trend", "over time", "by hour", "hourly", "daily", "weekly", "monthly",
    "theo giờ", "theo gio", "theo ngày", "theo ngay", "theo tuần", "theo tuan",
    # Distribution
    "distribution", "spread", "histogram", "phân bố", "phan bo",
    # Summary implying breakdown
    "summarize", "summary", "breakdown", "tổng hợp", "tong hop",
)


def _should_visualize(message: str, tool_hints: list[str] | None) -> bool:
    """Return True when we should attach a chart payload to the response.

    Triggers:
    - the "visualize" tool hint pill, OR
    - explicit viz keywords (chart, plot, graph, ...), OR
    - implicit viz signals (compare, top, per facility, over time, ...).

    Only suppressed for genuinely non-visual queries (single-fact lookups,
    definitions, yes/no questions).
    """
    if tool_hints and "visualize" in tool_hints:
        return True
    lowered = (message or "").lower()
    if any(kw in lowered for kw in _VISUALIZE_KEYWORDS):
        return True
    if any(kw in lowered for kw in _IMPLICIT_VIZ_KEYWORDS):
        return True
    return False


def _apply_tool_hints(request: "SolarChatRequest") -> "SolarChatRequest":
    """Kept as a pass-through for compatibility. Tool hints are now carried
    via ``request.tool_hints`` straight through to the prompt builder (see
    :func:`build_agentic_messages`) so the user's original message is never
    mutated (keeps the persisted transcript clean).
    """
    return request


def _has_any_marker(message: str, markers: tuple[str, ...]) -> bool:
    normalized = normalize_vietnamese_text(message)
    return any(normalize_vietnamese_text(m) in normalized for m in markers)


def _contains_scope_refusal_signal(text: str) -> bool:
    normalized = normalize_vietnamese_text(text)
    markers = (
        "toi chi",
        "i can only",
        "solar",
        "solar energy",
        "nang luong mat troi",
        "outside",
    )
    return any(m in normalized for m in markers)


def _is_prompt_injection_request(message: str) -> bool:
    normalized = normalize_vietnamese_text(message)
    token_groups: tuple[tuple[str, ...], ...] = (
        ("ignore", "previous"),
        ("ignore", "instructions"),
        ("bo qua", "huong dan"),
        ("bo qua", "chi dan"),
        ("reveal", "system"),
        ("system", "prompt"),
        ("developer", "prompt"),
        ("authentication", "token"),
    )
    if any(all(tok in normalized for tok in group) for group in token_groups):
        return True
    return _has_any_marker(message, _PROMPT_INJECTION_MARKERS)


def _build_scope_refusal(language: str) -> str:
    if language == "vi":
        return (
            "Tôi chỉ hỗ trợ các câu hỏi liên quan đến hệ thống năng lượng mặt trời "
            "(solar energy). Vui lòng đặt câu hỏi về dữ liệu, dự báo, hoặc hiệu suất "
            "năng lượng mặt trời."
        )
    return (
        "I can only assist with questions related to solar energy systems and "
        "the PV Lakehouse platform. Please ask about solar energy data or "
        "solar system performance."
    )


def _last_assistant_topic(history: list[ChatMessage]) -> ChatTopic | None:
    for msg in reversed(history):
        if str(getattr(msg, "sender", "")).lower() == "assistant" and getattr(msg, "topic", None):
            return msg.topic
    return None


def _is_capacity_or_station_query(message: str) -> bool:
    norm = normalize_vietnamese_text(message)
    markers = (
        "cong suat lap dat",
        "installed capacity",
        "largest capacity",
        "largest installed",
        "capacity of the station",
        "bao nhieu tram",
        "so tram",
        "tong so tram",
        "how many stations",
        "station count",
        "list all stations",
        "liet ke",
        "mui gio cua tram do",
        "timezone of that station",
        "we just discussed",
        "luc dau",
        "nhac lai chinh xac cong suat",
        "vua noi den",
    )
    return any(marker in norm for marker in markers)


def _is_implicit_followup(message: str) -> bool:
    norm = normalize_vietnamese_text(message)
    markers = (
        "chi so do",
        "chi so do tinh theo",
        "that figure",
        "that metric",
        "that number",
        "cai do",
        "thong so do",
        "chi so o",
        "chi so o tinh theo",
    )
    return any(marker in norm for marker in markers)


def _is_cross_topic_summary_query(message: str) -> bool:
    norm = normalize_vietnamese_text(message)
    markers = ("tom tat", "tom tat lai", "summarise", "summarize", "summary")
    return any(marker in norm for marker in markers)


_MULTI_DAY_STATION_REPORT_MARKERS: tuple[str, ...] = (
    "daily station report",
    "daily report",
    "station report",
    "bao cao tram",
    "bao cao theo tram",
    "bao cao hang ngay",
    "bao cao hang ngay tuan",
)

_MULTI_DAY_WINDOW_MARKERS: dict[str, int] = {
    "this week": 5,
    "tuan nay": 5,
    "past week": 7,
    "last week": 7,
    "tuan vua roi": 7,
    "tuan truoc": 7,
    "last 7 days": 7,
    "last seven days": 7,
    "7 ngay gan day": 7,
    "last 3 days": 3,
    "last three days": 3,
    "3 ngay gan day": 3,
    "last 5 days": 5,
    "5 ngay gan day": 5,
    "last 10 days": 10,
    "10 ngay gan day": 10,
    "last 14 days": 14,
    "two weeks": 14,
    "past fortnight": 14,
}


def _detect_multi_day_station_window(message: str) -> int | None:
    """Return the number of days to fetch when the user explicitly asks for a
    multi-day station report; return None otherwise.

    This exists because the LLM is occasionally flaky at driving the tool
    loop for phrases like "daily station report for this week" — it sometimes
    drops into a generic clarification response instead of calling the tool.
    A deterministic pre-fetch guarantees the data is on the table before
    synthesis runs, regardless of how the LLM behaves."""
    norm = normalize_vietnamese_text(message)
    if not any(m in norm for m in _MULTI_DAY_STATION_REPORT_MARKERS):
        return None
    for marker, days in _MULTI_DAY_WINDOW_MARKERS.items():
        if marker in norm:
            return days
    return None


def _has_in_domain_history(history: list[ChatMessage]) -> bool:
    """A prior assistant turn on a non-GENERAL topic establishes conversational
    context — the next user message is almost certainly a contextual follow-up
    even if it lacks explicit domain keywords (e.g. "phân tích sâu hơn",
    "list từ cao đến thấp"). Used to bypass the deterministic scope guard
    that would otherwise refuse such follow-ups."""
    last_topic = _last_assistant_topic(history)
    return last_topic is not None and last_topic != ChatTopic.GENERAL


def _is_in_domain_query(message: str) -> bool:
    """Best-effort detector for PV-Lakehouse in-domain requests.

    Used only to avoid false-positive deterministic scope refusals when intent
    detection falls back to GENERAL for short/ambiguous but relevant prompts.
    Meta questions about the assistant itself ("who are you", "what can you do",
    greetings) are also treated as in-domain so the model can introduce itself
    instead of being shut down by the deterministic scope guard.
    """
    norm = normalize_vietnamese_text(message)
    markers = (
        "solar",
        "pv",
        "energy",
        "nang luong",
        "facility",
        "facilities",
        "station",
        "stations",
        "tram",
        "co so",
        "pipeline",
        "forecast",
        "du bao",
        "model",
        "ml",
        "aqi",
        "weather",
        "thoi tiet",
        "data quality",
        "chat luong du lieu",
    )
    if any(m in norm for m in markers):
        return True
    return _is_meta_assistant_query(message)


_META_ASSISTANT_MARKERS: tuple[str, ...] = (
    "who are you",
    "what are you",
    "what is your name",
    "introduce yourself",
    "tell me about yourself",
    "what can you do",
    "what you can do",
    "what can you help",
    "what do you do",
    "what do you offer",
    "what do you support",
    "your capabilities",
    "your abilities",
    "your features",
    "how can you help",
    "how do you work",
    "ban la ai",
    "ban ten gi",
    "gioi thieu ban than",
    "gioi thieu ve ban",
    "ban co the lam gi",
    "ban giup duoc gi",
    "ban lam duoc gi",
    "tinh nang cua ban",
    "kha nang cua ban",
    "huong dan su dung",
    "hello",
    "hi there",
    "hey there",
    "xin chao",
    "chao ban",
    "chao",
)


def _is_meta_assistant_query(message: str) -> bool:
    """Detect self-introduction / capability / greeting questions."""
    norm = normalize_vietnamese_text(message).strip()
    if not norm:
        return False
    if any(m in norm for m in _META_ASSISTANT_MARKERS):
        return True
    # Bare one-word greetings (≤ 2 tokens) — e.g. "hi", "chao".
    tokens = norm.split()
    if len(tokens) <= 2 and tokens and tokens[0] in {"hi", "hello", "hey", "chao", "yo"}:
        return True
    return False


def _extract_search_query(
    message: str,
    history: "list | None",
) -> str:
    """Extract a focused search query from a web-search request.

    Strips trigger prefixes and Vietnamese filler words, then enriches
    short/vague queries with context from recent history.  The enrichment
    scans both user and assistant turns, skipping trivial header lines
    (e.g. "Definitions:") to find actionable topic text.
    """
    query = message.strip()
    lower = query.lower()

    # Remove leading web-search trigger phrase (longest match first)
    for prefix in _WEB_SEARCH_STRIP_PREFIXES:
        if lower.startswith(prefix):
            query = query[len(prefix):].lstrip(" ,;:").strip()
            lower = query.lower()
            break

    # Remove leading filler words (longest compound forms first)
    for filler in _QUERY_FILLER_PREFIXES:
        if lower.startswith(filler + " ") or lower == filler:
            query = query[len(filler):].lstrip(" ,;:").strip()
            lower = query.lower()
            break

    # Enrich short/vague queries with topic context from recent history.
    # Scans the last 6 messages (both roles) for the first non-trivial
    # line (>= 20 chars) — this skips short headers like "Definitions:"
    # and picks up meaningful topic text like "Performance Ratio (PR) la..."
    if len(query) < 50 and history:
        for msg in reversed(history[-6:]):
            content = (getattr(msg, "content", "") or "")
            for line in content.strip().split("\n"):
                cleaned = line.strip().lstrip("#*-•$`> ").strip()
                if len(cleaned) >= 20:
                    query = f"{query} {cleaned[:120]}"
                    break
            else:
                continue
            break

    return query.strip() or message.strip()


def _extract_top_facility_names(metrics: dict[str, Any], top_n: int = 2) -> list[str]:
    """Return top N facility names by capacity from pre-fetched get_facility_info data.

    Only uses the ``facilities`` key (from get_facility_info) — not
    ``top_facilities`` (from get_energy_performance) — to avoid replacing
    unrelated search queries (e.g. "how to calculate PR") with facility names.
    """
    facilities = metrics.get("facilities", [])
    if not isinstance(facilities, list) or not facilities:
        return []
    try:
        sorted_f = sorted(
            [f for f in facilities if isinstance(f, dict)],
            key=lambda x: float(x.get("capacity_mw") or x.get("total_capacity_mw") or 0),
            reverse=True,
        )
        names = [
            f.get("facility_name") or f.get("name") or ""
            for f in sorted_f[:top_n]
        ]
        return [n for n in names if n]
    except Exception:
        return []


# Primary tool to pre-execute when intent is clear and the LLM does not call tools
_TOPIC_TO_PRIMARY_TOOL: dict[ChatTopic, str] = {
    ChatTopic.SYSTEM_OVERVIEW: "get_system_overview",
    ChatTopic.ENERGY_PERFORMANCE: "get_energy_performance",
    ChatTopic.ML_MODEL: "get_ml_model_info",
    ChatTopic.PIPELINE_STATUS: "get_pipeline_status",
    ChatTopic.FORECAST_72H: "get_forecast_72h",
    ChatTopic.DATA_QUALITY_ISSUES: "get_data_quality_issues",
    ChatTopic.FACILITY_INFO: "get_facility_info",
}


class SolarAIChatService:
    """Solar AI Chat — agentic tool-calling architecture.

    The LLM receives the full Lakehouse architecture context and reasons
    about which tools to call.  Python contains no query-routing heuristics.
    """

    def __init__(
        self,
        repository: SolarChatRepository,
        intent_service: VietnameseIntentService,
        model_router: LLMModelRouter | None,
        history_repository: Any | None = None,
        vector_repo: "VectorRepository | None" = None,
        embedding_client: "GeminiEmbeddingClient | None" = None,
        # Accept-and-ignore for callers that still pass a legacy web search client.
        web_search_client: Any | None = None,
        tool_usage_logger: Any | None = None,
        *,
        planner_enabled: bool = True,           # kept for route-compat, ignored
        orchestrator_enabled: bool = True,       # kept for route-compat, ignored
        verifier_enabled: bool = True,
        hybrid_retrieval_enabled: bool = True,   # kept for route-compat, ignored
        max_tool_steps: int = MAX_TOOL_STEPS,
        planner_max_output_tokens: int | None = None,
        synthesis_max_output_tokens: int | None = None,
        verifier_max_output_tokens: int | None = None,
        deep_planner_enabled: bool = False,
    ) -> None:
        self._repository = repository
        self._intent_service = intent_service
        self._model_router = model_router
        self._history_repository = history_repository
        self._tool_executor = ToolExecutor(
            repository,
            vector_repo=vector_repo,
            embedding_client=embedding_client,
            usage_logger=tool_usage_logger,
        )
        self._web_search_client = web_search_client
        self._verifier_enabled = verifier_enabled
        self._max_tool_steps = max(1, max_tool_steps)
        self._synthesis_max_output_tokens = synthesis_max_output_tokens
        self._query_rewriter = QueryRewriter()
        self._chart_spec_builder = ChartSpecBuilder()
        self._answer_verifier = (
            AnswerVerifier(model_router, max_output_tokens=verifier_max_output_tokens)
            if verifier_enabled and model_router is not None
            else AnswerVerifier(None)
        )
        self._deep_planner = DeepPlanner(
            model_router,
            max_output_tokens=planner_max_output_tokens or 700,
            enabled=deep_planner_enabled and model_router is not None,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def handle_query(self, request: SolarChatRequest) -> SolarChatResponse:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex[:8]
        request = _apply_tool_hints(request)
        # Share the raw user message with the tool executor so it can gate
        # routing (e.g., force correlation queries through query_gold_kpi).
        try:
            self._tool_executor.set_user_query(request.message)
        except AttributeError:
            pass
        try:
            self._tool_executor.set_request_context(
                session_id=getattr(request, "session_id", None),
                user_id=getattr(request, "user_id", None),
            )
        except AttributeError:
            pass
        logger.info(
            "solar_chat_trace_start trace_id=%s session_id=%s role=%s tool_mode=%s hints=%s message=%s",
            trace_id,
            request.session_id,
            request.role.value if request.role else "unknown",
            getattr(request, "tool_mode", "auto"),
            getattr(request, "tool_hints", None),
            self._short(request.message, 200),
        )
        history_messages = self._load_history(request.session_id)
        logger.info(
            "solar_chat_trace_history trace_id=%s history_count=%d",
            trace_id,
            len(history_messages),
        )
        # v2 engine cutover: when SOLAR_CHAT_ENGINE=v2, route the request
        # through the slim primitive-based loop instead of the v1 ToolExecutor
        # path. v1 stays untouched for safe rollback.
        if self._is_v2_engine_enabled():
            try:
                return self._handle_with_v2_engine(
                    request, started, history_messages, trace_id,
                )
            except DatabricksDataUnavailableError:
                raise
            except Exception as err:
                logger.exception(
                    "solar_chat_v2_engine_failed trace_id=%s error=%s",
                    trace_id, err,
                )
                language = self._query_rewriter.rewrite(request.message).language
                return self._finalize_response(
                    request=request,
                    answer=build_insufficient_data_response(language),
                    topic=ChatTopic.GENERAL,
                    metrics={},
                    sources=[],
                    model_used="v2-engine-error",
                    fallback_used=True,
                    started=started,
                    trace_id=trace_id,
                    intent_confidence=0.0,
                    warning_message="v2 engine failed unexpectedly.",
                    thinking_trace=ThinkingTrace(
                        summary="v2 engine failed.",
                        steps=[ThinkingStep(step="v2 failure", detail=str(err), status="warning")],
                        trace_id=trace_id,
                    ),
                )
        try:
            return self._handle_with_agentic_loop(request, started, history_messages, trace_id)
        except DatabricksDataUnavailableError:
            raise
        except Exception as err:
            logger.exception(
                "solar_chat_agentic_path_failed trace_id=%s error=%s",
                trace_id,
                err,
            )
            language = self._query_rewriter.rewrite(request.message).language
            return self._finalize_response(
                request=request,
                answer=build_insufficient_data_response(language),
                topic=ChatTopic.GENERAL,
                metrics={},
                sources=[],
                model_used="agentic-error",
                fallback_used=True,
                started=started,
                trace_id=trace_id,
                intent_confidence=0.0,
                warning_message="Agentic loop failed unexpectedly.",
                thinking_trace=ThinkingTrace(
                    summary="Agentic loop failed. Returned unavailability notice.",
                    steps=[ThinkingStep(step="Loop failure", detail=str(err), status="warning")],
                    trace_id=trace_id,
                ),
            )

    # ------------------------------------------------------------------
    # v2 engine bridge
    # ------------------------------------------------------------------

    def _is_v2_engine_enabled(self) -> bool:
        try:
            from app.core.settings import SolarChatSettings
            return SolarChatSettings().engine_version.lower() == "v2"
        except Exception:  # noqa: BLE001 - never break v1 path on settings glitches
            return False

    def _handle_with_v2_engine(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
    ) -> SolarChatResponse:
        from app.core.settings import SolarChatSettings
        from app.services.solar_ai_chat.v2.dispatcher import V2Dispatcher
        from app.services.solar_ai_chat.v2.engine import V2ChatEngine

        if self._model_router is None:
            raise RuntimeError("v2 engine requires an LLMModelRouter")

        settings = SolarChatSettings()
        # role_id mirrors v1 chat-role mapping (analyst → data_analyst handled
        # at the request layer; here we just pass through whatever's set).
        role_id = (request.role.value if request.role else "admin").lower()

        dispatcher = V2Dispatcher(settings, role_id=role_id)
        engine = V2ChatEngine(self._model_router, dispatcher)
        rewrite = self._query_rewriter.rewrite(request.message)

        result = engine.run(
            user_message=request.message,
            history=history_messages,
            language=rewrite.language or "en",
        )

        from app.schemas.solar_ai_chat.visualization import (
            ChartPayload,
            DataTableColumn,
            DataTablePayload,
            KpiCard,
            KpiCardsPayload,
        )

        chart_payload: ChartPayload | None = None
        if result.chart and result.chart.get("format") == "vega-lite":
            spec = result.chart.get("spec") or {}
            mark = spec.get("mark")
            mark_type = mark if isinstance(mark, str) else (
                (mark or {}).get("type") if isinstance(mark, dict) else "bar"
            )
            chart_payload = ChartPayload(
                chart_type=mark_type or "bar",
                title=str(result.chart.get("title") or ""),
                format="vega-lite",
                spec=spec,
                row_count=result.chart.get("row_count"),
            )

        data_table_payload: DataTablePayload | None = None
        if result.data_table:
            data_table_payload = DataTablePayload(
                title=(result.chart or {}).get("title", "Query result") or "Query result",
                columns=[DataTableColumn(**c) for c in result.data_table.get("columns", [])],
                rows=result.data_table.get("rows", []),
                row_count=int(result.data_table.get("row_count", 0)),
            )

        kpi_payload: KpiCardsPayload | None = None
        if result.key_metrics and not chart_payload:
            cards = [
                KpiCard(label=str(k), value=v if isinstance(v, (int, float, str)) else str(v))
                for k, v in result.key_metrics.items()
                if v is not None
            ]
            if cards:
                kpi_payload = KpiCardsPayload(cards=cards)

        viz_snapshot: dict | None = None
        if chart_payload or data_table_payload or kpi_payload:
            viz_snapshot = {
                "chart": chart_payload.model_dump() if chart_payload else None,
                "data_table": data_table_payload.model_dump() if data_table_payload else None,
                "kpi_cards": kpi_payload.model_dump() if kpi_payload else None,
            }

        thinking_trace = ThinkingTrace(
            summary=f"v2 engine ran {len(result.trace_steps)} primitive call(s).",
            steps=[
                ThinkingStep(
                    step=f"{s['step']}. {s['primitive']}",
                    detail=f"args={s['args_preview']} duration={s['duration_ms']}ms",
                    status="success" if s["ok"] else "warning",
                )
                for s in result.trace_steps
            ],
            trace_id=trace_id,
        )

        try:
            self._persist_exchange(
                session_id=request.session_id,
                user_message=request.message,
                answer=result.answer,
                topic=ChatTopic.GENERAL,
                sources=result.sources,
                thinking_trace=thinking_trace,
                key_metrics=result.key_metrics or None,
                viz_requested=bool(result.chart),
                viz_payload=viz_snapshot,
            )
        except Exception as persist_err:  # noqa: BLE001
            logger.warning(
                "v2_engine_persist_failed trace_id=%s err=%s",
                trace_id, persist_err,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "solar_chat_v2_done trace_id=%s steps=%d model=%s latency_ms=%d chart=%s",
            trace_id, len(result.trace_steps), result.model_used, latency_ms,
            "vega" if chart_payload else "none",
        )
        return SolarChatResponse(
            answer=result.answer,
            topic=ChatTopic.GENERAL,
            role=request.role,
            key_metrics=result.key_metrics or {},
            sources=result.sources,
            model_used=result.model_used,
            fallback_used=result.fallback_used,
            latency_ms=latency_ms,
            intent_confidence=TOOL_PATH_CONFIDENCE,
            warning_message=result.error,
            thinking_trace=thinking_trace,
            ui_features=resolve_ui_features(request.role),
            data_table=data_table_payload,
            chart=chart_payload,
            kpi_cards=kpi_payload,
        )

    @staticmethod
    def _select_tool_declarations(request: SolarChatRequest) -> list:
        """Filter TOOL_DECLARATIONS based on the caller's tool_mode.

        - auto      → full palette (default)
        - selected  → only tools whose name is in request.allowed_tools
        - none      → no tools (caller should short-circuit before calling this)
        """
        mode = getattr(request, "tool_mode", "auto") or "auto"
        if mode != "selected":
            return list(TOOL_DECLARATIONS)
        allowed = set(request.allowed_tools or [])
        if not allowed:
            return list(TOOL_DECLARATIONS)

        def _tool_name(decl: Any) -> str:
            if isinstance(decl, dict):
                return str(decl.get("name") or decl.get("function", {}).get("name", ""))
            return str(getattr(decl, "name", ""))

        filtered = [d for d in TOOL_DECLARATIONS if _tool_name(d) in allowed]
        return filtered or list(TOOL_DECLARATIONS)

    def handle_query_stream(self, request: SolarChatRequest):
        """Streaming variant — yields SSE-ready JSON strings.

        Each yielded item is a ``data: <json>\\n\\n``-formatted SSE line.
        The caller (FastAPI StreamingResponse) sends these directly to the
        browser.  The final event is always ``done`` or ``error``.

        Design: runs the full synchronous agentic loop but emits events at
        every decision boundary so the frontend can show a live Task Tracker.
        No async/await; the generator is wrapped by the SSE route handler.
        """
        import json as _json
        from app.schemas.solar_ai_chat.stream import (
            ThinkingStepEvent,
            ToolResultEvent,
            StatusUpdateEvent,
            DoneEvent,
            ErrorEvent,
        )

        def _sse(event_obj) -> str:
            return "data: " + _json.dumps(event_obj.model_dump()) + "\n\n"

        started = time.perf_counter()
        trace_id = uuid.uuid4().hex[:8]
        request = _apply_tool_hints(request)
        try:
            self._tool_executor.set_user_query(request.message)
        except AttributeError:
            pass
        try:
            self._tool_executor.set_request_context(
                session_id=getattr(request, "session_id", None),
                user_id=getattr(request, "user_id", None),
            )
        except AttributeError:
            pass

        try:
            yield _sse(StatusUpdateEvent(text="Analyzing your request…"))

            history_messages = self._load_history(request.session_id)

            # Facility code expansion
            expanded_message = expand_facility_codes_in_message(request.message)
            if expanded_message != request.message:
                request = request.model_copy(update={"message": expanded_message})

            language = self._query_rewriter.rewrite(request.message).language

            if self._model_router is None:
                yield _sse(ErrorEvent(message="No LLM configured.", code="no_llm"))
                return

            # Prompt injection check
            if _is_prompt_injection_request(request.message):
                final_resp = self._finalize_response(
                    request=request,
                    answer=_build_scope_refusal(language),
                    topic=ChatTopic.GENERAL,
                    metrics={},
                    sources=[],
                    model_used="scope-guard",
                    fallback_used=True,
                    started=started,
                    trace_id=trace_id,
                    intent_confidence=0.0,
                    warning_message="Prompt-injection request refused.",
                )
                yield _sse(DoneEvent(
                    answer=final_resp.answer,
                    topic=final_resp.topic.value,
                    role=final_resp.role.value,
                    sources=[s.model_dump() for s in final_resp.sources],
                    key_metrics=final_resp.key_metrics,
                    model_used=final_resp.model_used,
                    fallback_used=final_resp.fallback_used,
                    latency_ms=final_resp.latency_ms,
                    intent_confidence=final_resp.intent_confidence,
                    warning_message=final_resp.warning_message,
                    ui_features=final_resp.ui_features,
                    trace_id=trace_id,
                ))
                return

            # Intent detection
            intent_result = self._intent_service.detect_intent(request.message)
            try:
                intent_confidence = float(intent_result.confidence)
            except (TypeError, ValueError):
                intent_confidence = 0.0
            intent_topic = intent_result.topic

            if _is_cross_topic_summary_query(request.message):
                intent_topic = ChatTopic.SYSTEM_OVERVIEW
                intent_confidence = max(intent_confidence, 0.8)
            elif _is_capacity_or_station_query(request.message):
                intent_topic = ChatTopic.FACILITY_INFO
                intent_confidence = max(intent_confidence, 0.85)
            elif _is_implicit_followup(request.message):
                previous_topic = _last_assistant_topic(history_messages)
                intent_topic = previous_topic or ChatTopic.ENERGY_PERFORMANCE
                intent_confidence = max(intent_confidence, 0.8)

            # General scope guard
            if (
                intent_topic == ChatTopic.GENERAL
                and not _needs_web_search(request.message)
                and not _is_in_domain_query(request.message)
                and not _has_in_domain_history(history_messages)
            ):
                final_resp = self._finalize_response(
                    request=request,
                    answer=_build_scope_refusal(language),
                    topic=ChatTopic.GENERAL,
                    metrics={},
                    sources=[],
                    model_used="scope-guard",
                    fallback_used=True,
                    started=started,
                    trace_id=trace_id,
                    intent_confidence=intent_confidence,
                )
                yield _sse(DoneEvent(
                    answer=final_resp.answer,
                    topic=final_resp.topic.value,
                    role=final_resp.role.value,
                    sources=[],
                    key_metrics={},
                    model_used=final_resp.model_used,
                    fallback_used=final_resp.fallback_used,
                    latency_ms=final_resp.latency_ms,
                    intent_confidence=final_resp.intent_confidence,
                    ui_features=final_resp.ui_features,
                    trace_id=trace_id,
                ))
                return

            try:
                latest_mart_date = self._repo._resolve_latest_date("gold.mart_energy_daily")
                today_str = latest_mart_date.isoformat()
            except Exception:
                latest_mart_date = None
                today_str = None
                
            messages = build_agentic_messages(
                request.message, language, history_messages, today_str=today_str,
                tool_hints=getattr(request, "tool_hints", None),
            )
            all_sources: list[dict[str, str]] = []
            all_metrics: dict[str, Any] = {}
            topic = ChatTopic.GENERAL
            step_events: list[dict[str, Any]] = []
            answer: str | None = None
            model_used = "llm-agentic"
            fallback_used = False
            locked_topic: ChatTopic | None = None
            user_query_date = extract_query_date(request.message, base_date=latest_mart_date)

            # ----------------------------------------------------------
            # Tool-mode override: user explicitly selected "no tools"
            # in the UI — answer directly from LLM knowledge. Matches
            # ChatGPT/Claude "Default" mode without web/tools.
            # ----------------------------------------------------------
            if getattr(request, "tool_mode", "auto") == "none":
                _NO_TOOL_STEP = 0
                yield _sse(StatusUpdateEvent(text="Answering without tools…"))
                yield _sse(ThinkingStepEvent(
                    step=_NO_TOOL_STEP, tool_name="answer_directly",
                    label="Answering without tools", trace_id=trace_id,
                ))
                t_dir = time.perf_counter()
                try:
                    synthesis_prompt = build_synthesis_prompt(
                        user_message=request.message,
                        evidence_text="",
                        language=language,
                        history=history_messages or None,
                    )
                    gen = self._model_router.generate(
                        synthesis_prompt,
                        max_output_tokens=self._synthesis_max_output_tokens,
                        temperature=0.2,
                    )
                    answer = gen.text
                    model_used = gen.model_used
                    fallback_used = gen.fallback_used
                    dur_dir = int((time.perf_counter() - t_dir) * 1000)
                    step_events.append({"step": _NO_TOOL_STEP, "tool": "answer_directly",
                                        "status": "ok", "duration_ms": dur_dir, "detail": f"via {model_used}"})
                    yield _sse(ToolResultEvent(
                        step=_NO_TOOL_STEP, tool_name="answer_directly",
                        status="ok", duration_ms=dur_dir, trace_id=trace_id,
                    ))
                except Exception as direct_err:
                    dur_dir = int((time.perf_counter() - t_dir) * 1000)
                    logger.warning("solar_chat_no_tools_direct_failed trace_id=%s err=%s", trace_id, direct_err)
                    answer = build_insufficient_data_response(language)
                    fallback_used = True
                    step_events.append({"step": _NO_TOOL_STEP, "tool": "answer_directly",
                                        "status": "error", "duration_ms": dur_dir})
                    yield _sse(ToolResultEvent(
                        step=_NO_TOOL_STEP, tool_name="answer_directly",
                        status="error", duration_ms=dur_dir, trace_id=trace_id,
                    ))

                thinking_trace = ThinkingTrace(
                    summary="Direct answer (no tools).",
                    steps=[ThinkingStep(step="answer_directly", detail=f"duration_ms={dur_dir}",
                                         status="success" if answer else "warning")],
                    trace_id=trace_id,
                )
                final_resp = self._finalize_response(
                    request=request, answer=answer or "",
                    topic=ChatTopic.GENERAL, metrics={}, sources=[],
                    model_used=model_used, fallback_used=fallback_used,
                    started=started, trace_id=trace_id, intent_confidence=0.0,
                    thinking_trace=thinking_trace,
                )
                yield _sse(DoneEvent(
                    answer=final_resp.answer, topic=final_resp.topic.value,
                    role=final_resp.role.value, sources=[],
                    key_metrics={}, model_used=final_resp.model_used,
                    fallback_used=final_resp.fallback_used, latency_ms=final_resp.latency_ms,
                    intent_confidence=0.0, warning_message=final_resp.warning_message,
                    thinking_trace=final_resp.thinking_trace.model_dump() if final_resp.thinking_trace else None,
                    ui_features=final_resp.ui_features,
                    trace_id=trace_id,
                ))
                return
            
            # ----------------------------------------------------------
            # Deep planner (stream path): structured plan with all tools
            # needed for compound/multi-intent prompts.  Each planned tool
            # is executed and emitted as a thinking step so the web UI
            # shows progress for every call.
            # ----------------------------------------------------------
            planner_used_stream = False
            if self._deep_planner.enabled and not _needs_web_search(request.message):
                try:
                    plan = self._deep_planner.plan(
                        request.message, language, history_messages or None
                    )
                except Exception as plan_err:
                    logger.warning("solar_chat_stream_planner_error trace_id=%s error=%s", trace_id, plan_err)
                    plan = None

                if plan and plan.actions:
                    allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
                    yield _sse(StatusUpdateEvent(text="Planning tool calls…"))
                    for idx, action in enumerate(list(plan.actions)[:MAX_PLANNED_ACTIONS]):
                        step_num = len(step_events)
                        tool_name = str(getattr(action, "tool", "") or "")
                        tool_args = dict(getattr(action, "arguments", {}) or {})
                        if not tool_name or tool_name == "answer_directly":
                            step_events.append({"step": step_num, "tool": tool_name or "answer_directly", "status": "skipped"})
                            continue

                        yield _sse(ThinkingStepEvent(
                            step=step_num, tool_name=tool_name,
                            label=tool_label(tool_name), trace_id=trace_id,
                        ))
                        t0 = time.perf_counter()

                        if tool_name == "web_lookup":
                            web_response = self._execute_web_lookup(tool_args)
                            dur = int((time.perf_counter() - t0) * 1000)
                            for wr in web_response.get("results", []):
                                wr_url = wr.get("url", "")
                                if wr_url:
                                    all_sources.append({
                                        "layer": "Web", "dataset": wr.get("title", wr_url),
                                        "data_source": "web_search", "url": wr_url,
                                    })
                            messages.append({
                                "role": "model",
                                "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
                            })
                            messages.append({
                                "role": "user",
                                "parts": [{"functionResponse": {"name": tool_name, "response": web_response}}],
                            })
                            step_events.append({
                                "step": step_num, "tool": tool_name, "status": "ok",
                                "result_count": len(web_response.get("results", [])),
                            })
                            yield _sse(ToolResultEvent(
                                step=step_num, tool_name=tool_name, status="ok",
                                duration_ms=dur, trace_id=trace_id,
                            ))
                            continue

                        if tool_name not in allowed_tools:
                            step_events.append({"step": step_num, "tool": tool_name, "status": "denied"})
                            yield _sse(ToolResultEvent(
                                step=step_num, tool_name=tool_name, status="denied", trace_id=trace_id,
                            ))
                            continue

                        try:
                            data, sources = self._tool_executor.execute(tool_name, tool_args, request.role)
                            dur = int((time.perf_counter() - t0) * 1000)
                            _merge_tool_result(all_metrics, tool_name, data)
                            all_sources.extend(sources)
                            if tool_name in TOOL_NAME_TO_TOPIC and locked_topic is None:
                                topic = ChatTopic(TOOL_NAME_TO_TOPIC[tool_name])
                            messages.append({
                                "role": "model",
                                "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
                            })
                            messages.append({
                                "role": "user",
                                "parts": [{"functionResponse": {"name": tool_name, "response": data}}],
                            })
                            step_events.append({
                                "step": step_num, "tool": tool_name, "status": "ok",
                                "metric_keys": sorted(data.keys()), "source_count": len(sources),
                            })
                            yield _sse(ToolResultEvent(
                                step=step_num, tool_name=tool_name, status="ok",
                                metric_keys=sorted(data.keys()), duration_ms=dur, trace_id=trace_id,
                            ))
                        except DatabricksDataUnavailableError:
                            raise
                        except Exception as tool_err:
                            dur = int((time.perf_counter() - t0) * 1000)
                            step_events.append({
                                "step": step_num, "tool": tool_name, "status": "error",
                                "detail": str(tool_err)[:120],
                            })
                            yield _sse(ToolResultEvent(
                                step=step_num, tool_name=tool_name, status="error",
                                duration_ms=dur, trace_id=trace_id,
                            ))

                    if all_metrics:
                        planner_used_stream = True

            # Deterministic multi-day station-report pre-fetch.
            # Fires on explicit phrases like "daily station report for this week"
            # so the agent always has per-day data on the table before synthesis,
            # even in runs where the LLM flakes out and replies with a generic
            # clarification question instead of driving the tool loop.
            multi_day_window = (
                None if planner_used_stream
                else _detect_multi_day_station_window(request.message)
            )
            if multi_day_window and "get_station_daily_report" in ROLE_TOOL_PERMISSIONS.get(request.role, set()):
                try:
                    anchor_latest = self._repo._resolve_latest_date("gold.mart_energy_daily")
                except Exception:
                    anchor_latest = date.today()
                window_start = anchor_latest - timedelta(days=multi_day_window - 1)
                day = window_start
                while day <= anchor_latest:
                    step_idx = len(step_events)
                    yield _sse(ThinkingStepEvent(
                        step=step_idx, tool_name="get_station_daily_report",
                        label=tool_label("get_station_daily_report"), trace_id=trace_id,
                    ))
                    t0 = time.perf_counter()
                    try:
                        tool_args = {"anchor_date": day.isoformat()}
                        data, sources = self._tool_executor.execute(
                            "get_station_daily_report", tool_args, request.role,
                        )
                        dur = int((time.perf_counter() - t0) * 1000)
                        _merge_tool_result(all_metrics, "get_station_daily_report", data)
                        all_sources.extend(sources)
                        messages.append({
                            "role": "model",
                            "parts": [{"functionCall": {"name": "get_station_daily_report", "args": tool_args}}],
                        })
                        messages.append({
                            "role": "user",
                            "parts": [{"functionResponse": {"name": "get_station_daily_report", "response": data}}],
                        })
                        step_events.append({
                            "step": step_idx, "tool": "get_station_daily_report", "status": "ok",
                            "metric_keys": sorted(data.keys()), "source_count": len(sources),
                        })
                        yield _sse(ToolResultEvent(
                            step=step_idx, tool_name="get_station_daily_report",
                            status="ok", metric_keys=sorted(data.keys()),
                            duration_ms=dur, trace_id=trace_id,
                        ))
                    except Exception as tool_err:
                        dur = int((time.perf_counter() - t0) * 1000)
                        step_events.append({
                            "step": step_idx, "tool": "get_station_daily_report",
                            "status": "error", "detail": str(tool_err)[:120],
                        })
                        yield _sse(ToolResultEvent(
                            step=step_idx, tool_name="get_station_daily_report",
                            status="error", duration_ms=dur, trace_id=trace_id,
                        ))
                    day = day + timedelta(days=1)
                topic = ChatTopic.ENERGY_PERFORMANCE
                planner_used_stream = True

            # Intent-based pre-fetch (step 0) — legacy fallback when planner
            # returned no actionable plan.
            prefetch_tool: str | None = None
            if planner_used_stream:
                prefetch_tool = None
            elif user_query_date is not None:
                prefetch_tool = "get_station_daily_report"
            elif intent_topic != ChatTopic.GENERAL and intent_confidence >= 0.6:
                prefetch_tool = _TOPIC_TO_PRIMARY_TOOL.get(intent_topic)

            if prefetch_tool:
                allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
                if prefetch_tool in allowed_tools:
                    yield _sse(ThinkingStepEvent(
                        step=0,
                        tool_name=prefetch_tool,
                        label=tool_label(prefetch_tool),
                        trace_id=trace_id,
                    ))
                    t0 = time.perf_counter()
                    try:
                        tool_args = (
                            {"anchor_date": user_query_date.isoformat()}
                            if user_query_date and prefetch_tool == "get_station_daily_report"
                            else {}
                        )
                        data, sources = self._tool_executor.execute(
                            prefetch_tool, tool_args, request.role
                        )
                        dur = int((time.perf_counter() - t0) * 1000)
                        _merge_tool_result(all_metrics, prefetch_tool, data)
                        all_sources.extend(sources)
                        topic = intent_topic if user_query_date is None else ChatTopic.ENERGY_PERFORMANCE
                        messages.append({
                            "role": "model",
                            "parts": [{"functionCall": {"name": prefetch_tool, "args": tool_args}}],
                        })
                        messages.append({
                            "role": "user",
                            "parts": [{"functionResponse": {
                                "name": prefetch_tool,
                                "response": data,
                            }}],
                        })
                        step_events.append({"step": 0, "tool": prefetch_tool, "status": "ok",
                                            "metric_keys": sorted(data.keys()), "source_count": len(sources)})
                        yield _sse(ToolResultEvent(
                            step=0,
                            tool_name=prefetch_tool,
                            status="ok",
                            metric_keys=sorted(data.keys()),
                            duration_ms=dur,
                            trace_id=trace_id,
                        ))
                    except DatabricksDataUnavailableError:
                        raise
                    except Exception as prefetch_err:
                        dur = int((time.perf_counter() - t0) * 1000)
                        step_events.append({"step": 0, "tool": prefetch_tool, "status": "error",
                                            "detail": str(prefetch_err)[:80]})
                        yield _sse(ToolResultEvent(
                            step=0, tool_name=prefetch_tool, status="error", duration_ms=dur, trace_id=trace_id,
                        ))
                        logger.warning("solar_chat_stream_prefetch_failed trace_id=%s error=%s", trace_id, prefetch_err)

            # Fast-path synthesis
            if all_metrics and not _needs_web_search(request.message):
                _SYNTH_STEP = len(step_events)  # place after tool steps
                yield _sse(StatusUpdateEvent(text="Synthesizing answer…"))
                yield _sse(ThinkingStepEvent(
                    step=_SYNTH_STEP, tool_name="synthesize",
                    label="Synthesizing answer", trace_id=trace_id,
                ))
                evidence_text = json.dumps(all_metrics, ensure_ascii=False)
                synthesis_prompt = build_synthesis_prompt(
                    user_message=request.message,
                    evidence_text=evidence_text,
                    language=language,
                    history=history_messages or None,
                )
                t_syn = time.perf_counter()
                try:
                    gen = self._model_router.generate(
                        synthesis_prompt,
                        max_output_tokens=self._synthesis_max_output_tokens,
                        temperature=0.1,
                    )
                    answer = gen.text
                    model_used = gen.model_used
                    fallback_used = gen.fallback_used
                    dur_syn = int((time.perf_counter() - t_syn) * 1000)
                    step_events.append({"step": _SYNTH_STEP, "tool": "synthesize", "status": "ok", "duration_ms": dur_syn, "detail": f"via {model_used}"})
                    yield _sse(ToolResultEvent(step=_SYNTH_STEP, tool_name="synthesize", status="ok", duration_ms=dur_syn, trace_id=trace_id))
                except Exception:
                    dur_syn = int((time.perf_counter() - t_syn) * 1000)
                    answer = build_data_only_summary(all_metrics, all_sources, language)
                    fallback_used = True
                    step_events.append({"step": _SYNTH_STEP, "tool": "synthesize", "status": "error", "duration_ms": dur_syn})
                    yield _sse(ToolResultEvent(step=_SYNTH_STEP, tool_name="synthesize", status="error", duration_ms=dur_syn, trace_id=trace_id))

            # Agentic loop
            for step_num in range(1, self._max_tool_steps + 1):
                if answer is not None:
                    break
                yield _sse(StatusUpdateEvent(text=f"Step {step_num}: calling AI agent…"))
                try:
                    result = self._model_router.generate_with_tools(
                        messages,
                        self._select_tool_declarations(request),
                        max_output_tokens=self._synthesis_max_output_tokens,
                        require_function_call=(step_num == 1 and not all_metrics),
                    )
                except ToolCallNotSupportedError:
                    evidence_text = json.dumps(all_metrics, ensure_ascii=False)
                    synthesis_prompt = build_synthesis_prompt(
                        user_message=request.message,
                        evidence_text=evidence_text,
                        language=language,
                        history=history_messages or None,
                    )
                    _SYNTH_STEP_F = len(step_events)
                    yield _sse(StatusUpdateEvent(text="Synthesizing answer (fallback)…"))
                    yield _sse(ThinkingStepEvent(
                        step=_SYNTH_STEP_F, tool_name="synthesize",
                        label="Synthesizing answer", trace_id=trace_id,
                    ))
                    t_syn = time.perf_counter()
                    try:
                        gen = self._model_router.generate(
                            synthesis_prompt,
                            max_output_tokens=self._synthesis_max_output_tokens,
                            temperature=0.1,
                        )
                        answer = gen.text
                        model_used = gen.model_used
                        fallback_used = gen.fallback_used
                        dur_syn = int((time.perf_counter() - t_syn) * 1000)
                        step_events.append({"step": _SYNTH_STEP_F, "tool": "synthesize", "status": "ok", "duration_ms": dur_syn, "detail": f"via {model_used}"})
                        yield _sse(ToolResultEvent(step=_SYNTH_STEP_F, tool_name="synthesize", status="ok", duration_ms=dur_syn, trace_id=trace_id))
                    except Exception:
                        dur_syn = int((time.perf_counter() - t_syn) * 1000)
                        answer = build_data_only_summary(all_metrics, all_sources, language)
                        fallback_used = True
                        step_events.append({"step": _SYNTH_STEP_F, "tool": "synthesize", "status": "error", "duration_ms": dur_syn})
                        yield _sse(ToolResultEvent(step=_SYNTH_STEP_F, tool_name="synthesize", status="error", duration_ms=dur_syn, trace_id=trace_id))
                    break
                except ModelUnavailableError:
                    raise

                if result.function_call is None:
                    answer = result.text or build_data_only_summary(all_metrics, all_sources, language)
                    model_used = result.model_used
                    fallback_used = result.fallback_used
                    break

                tool_name = result.function_call.name
                tool_args = dict(result.function_call.arguments or {})

                yield _sse(ThinkingStepEvent(
                    step=step_num,
                    tool_name=tool_name,
                    label=tool_label(tool_name),
                    trace_id=trace_id,
                ))

                messages.append({
                    "role": "model",
                    "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
                })

                if tool_name == "answer_directly":
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": tool_name, "response": {"status": "ok"}}}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "skipped"})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="skipped", trace_id=trace_id,
                    ))
                    continue

                if tool_name == "web_lookup":
                    t0 = time.perf_counter()
                    web_response = self._execute_web_lookup(tool_args)
                    dur = int((time.perf_counter() - t0) * 1000)
                    for wr in web_response.get("results", []):
                        wr_url = wr.get("url", "")
                        if wr_url:
                            all_sources.append({
                                "layer": "Web", "dataset": wr.get("title", wr_url),
                                "data_source": "web_search", "url": wr_url,
                            })
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": tool_name, "response": web_response}}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "ok",
                                        "result_count": len(web_response.get("results", []))})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="ok", duration_ms=dur, trace_id=trace_id,
                    ))
                    continue

                # RBAC
                allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
                if tool_name not in allowed_tools:
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": tool_name,
                            "response": {"error": f"Access denied for role '{request.role.value}'."},
                        }}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "denied"})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="denied", trace_id=trace_id,
                    ))
                    continue

                # Execute tool
                t0 = time.perf_counter()
                try:
                    data, sources = self._tool_executor.execute(tool_name, tool_args, request.role)
                    dur = int((time.perf_counter() - t0) * 1000)
                    all_sources.extend(sources)
                    _merge_tool_result(all_metrics, tool_name, data)
                    if tool_name in TOOL_NAME_TO_TOPIC and locked_topic is None:
                        topic = ChatTopic(TOOL_NAME_TO_TOPIC[tool_name])
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": tool_name, "response": data}}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "ok",
                                        "metric_keys": sorted(data.keys()), "source_count": len(sources)})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="ok",
                        metric_keys=sorted(data.keys()), duration_ms=dur, trace_id=trace_id,
                    ))
                except DatabricksDataUnavailableError:
                    raise
                except ToolRoutingBlockedError as block_err:
                    # Soft-skip: feed the guidance back to the LLM so it can
                    # pick a better tool, but don't surface a red ❌ to the user
                    # — the model will recover on its next step.
                    dur = int((time.perf_counter() - t0) * 1000)
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": tool_name,
                            "response": {"error": str(block_err)},
                        }}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "skipped",
                                        "detail": "routed to better tool"})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="skipped", duration_ms=dur, trace_id=trace_id,
                    ))
                except Exception as tool_err:
                    dur = int((time.perf_counter() - t0) * 1000)
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": tool_name,
                            "response": {"error": str(tool_err)},
                        }}],
                    })
                    step_events.append({"step": step_num, "tool": tool_name, "status": "error",
                                        "detail": str(tool_err)})
                    yield _sse(ToolResultEvent(
                        step=step_num, tool_name=tool_name, status="error", duration_ms=dur, trace_id=trace_id,
                    ))

            # Force synthesis if loop exhausted
            if answer is None:
                _SYNTH_STEP_A = len(step_events)
                yield _sse(StatusUpdateEvent(text="Finalizing answer…"))
                yield _sse(ThinkingStepEvent(
                    step=_SYNTH_STEP_A, tool_name="synthesize",
                    label="Synthesizing answer", trace_id=trace_id,
                ))
                t_syn = time.perf_counter()
                try:
                    force_result = self._model_router.generate_with_tools(
                        messages, [],
                        max_output_tokens=self._synthesis_max_output_tokens,
                    )
                    answer = force_result.text or build_data_only_summary(all_metrics, all_sources, language)
                    model_used = force_result.model_used
                    fallback_used = force_result.fallback_used
                    dur_syn = int((time.perf_counter() - t_syn) * 1000)
                    step_events.append({"step": _SYNTH_STEP_A, "tool": "synthesize", "status": "ok", "duration_ms": dur_syn, "detail": f"via {model_used}"})
                    yield _sse(ToolResultEvent(step=_SYNTH_STEP_A, tool_name="synthesize", status="ok", duration_ms=dur_syn, trace_id=trace_id))
                except Exception:
                    dur_syn = int((time.perf_counter() - t_syn) * 1000)
                    answer = build_data_only_summary(all_metrics, all_sources, language)
                    fallback_used = True
                    step_events.append({"step": _SYNTH_STEP_A, "tool": "synthesize", "status": "error", "duration_ms": dur_syn})
                    yield _sse(ToolResultEvent(step=_SYNTH_STEP_A, tool_name="synthesize", status="error", duration_ms=dur_syn, trace_id=trace_id))


            def _fmt_step_detail(ev: dict) -> str:
                """Return a clean human-readable detail string for a step event."""
                keys = ev.get("metric_keys")
                if keys and isinstance(keys, list):
                    return f"{len(keys)} metric{'s' if len(keys) != 1 else ''} retrieved"
                rc = ev.get("result_count")
                if rc is not None:
                    return f"{rc} result{'s' if int(rc) != 1 else ''} found"
                detail = ev.get("detail")
                if detail:
                    return str(detail)[:120]
                return ev.get("status", "ok")

            def _fmt_step_label(ev: dict) -> str:
                return tool_label(ev.get("tool", ""))

            topic_label = topic.value.replace("_", " ").title()
            thinking_trace = ThinkingTrace(
                summary=f"{len(step_events)} tool call{'s' if len(step_events) != 1 else ''} · {topic_label} · {model_used}",
                steps=[
                    ThinkingStep(
                        step=_fmt_step_label(ev),
                        detail=_fmt_step_detail(ev),
                        status="success" if ev.get("status") in ("ok", "skipped") else "warning",
                    )
                    for ev in step_events
                ],
                trace_id=trace_id,
            )

            if locked_topic is not None:
                topic = locked_topic

            source_objects: list[SourceMetadata] = [
                SourceMetadata(**s) if isinstance(s, dict) else s
                for s in all_sources
            ]

            final_resp = self._finalize_response(
                request=request,
                answer=answer,
                topic=topic,
                metrics=all_metrics,
                sources=source_objects,
                model_used=model_used,
                fallback_used=fallback_used,
                started=started,
                trace_id=trace_id,
                intent_confidence=intent_confidence,
                thinking_trace=thinking_trace,
            )

            yield _sse(DoneEvent(
                answer=final_resp.answer,
                topic=final_resp.topic.value,
                role=final_resp.role.value,
                sources=[s.model_dump() for s in final_resp.sources],
                key_metrics=final_resp.key_metrics,
                model_used=final_resp.model_used,
                fallback_used=final_resp.fallback_used,
                latency_ms=final_resp.latency_ms,
                intent_confidence=final_resp.intent_confidence,
                warning_message=final_resp.warning_message,
                thinking_trace=(
                    final_resp.thinking_trace.model_dump()
                    if final_resp.thinking_trace else None
                ),
                ui_features=final_resp.ui_features,
                data_table=final_resp.data_table.model_dump() if final_resp.data_table else None,
                chart=final_resp.chart.model_dump() if final_resp.chart else None,
                kpi_cards=final_resp.kpi_cards.model_dump() if final_resp.kpi_cards else None,
                trace_id=trace_id,
            ))

        except DatabricksDataUnavailableError as db_err:
            yield _sse(ErrorEvent(
                message="Databricks is temporarily unavailable. Please retry shortly.",
                code="databricks_unavailable",
            ))
        except Exception as err:
            logger.exception("solar_chat_stream_unhandled trace_id=%s error=%s", trace_id, err)
            yield _sse(ErrorEvent(
                message="Solar AI Chat is temporarily unavailable. Please retry shortly.",
                code="stream_error",
            ))

    # ------------------------------------------------------------------
    # Agentic tool-calling loop
    # ------------------------------------------------------------------

    def _handle_with_agentic_loop(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
    ) -> SolarChatResponse:
        """Native LLM tool-calling loop (ReAct pattern).

        The LLM:
         * Receives the full Lakehouse architecture context
         * Decides which tools to call and in what sequence
         * Synthesises the final answer after collecting all tool results
        """
        # Bug #5: Expand facility ID codes (e.g. WRSF1 → "White Rock Solar Farm")
        # BEFORE language detection and intent classification so the LLM and
        # intent service receive meaningful station names, not opaque codes.
        expanded_message = expand_facility_codes_in_message(request.message)
        if expanded_message != request.message:
            logger.info(
                "solar_chat_facility_code_expand trace_id=%s original=%r expanded=%r",
                trace_id,
                request.message,
                expanded_message,
            )
            request = request.model_copy(update={"message": expanded_message})

        language = self._query_rewriter.rewrite(request.message).language

        # Tool-mode override: answer directly without calling tools.
        if getattr(request, "tool_mode", "auto") == "none" and self._model_router is not None:
            try:
                synthesis_prompt = build_synthesis_prompt(
                    user_message=request.message,
                    evidence_text="",
                    language=language,
                    history=history_messages or None,
                )
                gen = self._model_router.generate(
                    synthesis_prompt,
                    max_output_tokens=self._synthesis_max_output_tokens,
                    temperature=0.2,
                )
                return self._finalize_response(
                    request=request, answer=gen.text,
                    topic=ChatTopic.GENERAL, metrics={}, sources=[],
                    model_used=gen.model_used, fallback_used=gen.fallback_used,
                    started=started, trace_id=trace_id, intent_confidence=0.0,
                    thinking_trace=ThinkingTrace(
                        summary="Direct answer (no tools).",
                        steps=[ThinkingStep(step="answer_directly", detail=f"via {gen.model_used}",
                                             status="success")],
                        trace_id=trace_id,
                    ),
                )
            except Exception as direct_err:
                logger.warning("solar_chat_no_tools_sync_failed trace_id=%s err=%s", trace_id, direct_err)
                return self._finalize_response(
                    request=request, answer=build_insufficient_data_response(language),
                    topic=ChatTopic.GENERAL, metrics={}, sources=[],
                    model_used="none", fallback_used=True,
                    started=started, trace_id=trace_id, intent_confidence=0.0,
                )

        # No LLM -> cannot do agentic reasoning
        if self._model_router is None:
            return self._finalize_response(
                request=request,
                answer=build_insufficient_data_response(language),
                topic=ChatTopic.GENERAL,
                metrics={},
                sources=[],
                model_used="none",
                fallback_used=True,
                started=started,
                trace_id=trace_id,
                intent_confidence=0.0,
                warning_message="No LLM configured. Agentic tool calling requires a model.",
                thinking_trace=ThinkingTrace(
                    summary="No LLM configured.",
                    steps=[ThinkingStep(step="No LLM", detail="model_router is None", status="warning")],
                    trace_id=trace_id,
                ),
            )

        try:
            latest_mart_date = self._repo._resolve_latest_date("gold.mart_energy_daily")
            today_str = latest_mart_date.isoformat()
        except Exception:
            latest_mart_date = None
            today_str = None
            
        messages = build_agentic_messages(request.message, language, history_messages, today_str=today_str)
        all_sources: list[dict[str, str]] = []
        all_metrics: dict[str, Any] = {}
        topic = ChatTopic.GENERAL
        step_events: list[dict[str, Any]] = []
        answer: str | None = None
        model_used = "llm-agentic"
        fallback_used = False
        locked_topic: ChatTopic | None = None

        # ------------------------------------------------------------------
        # Intent-based pre-fetch: if the query topic is clear, execute the
        # primary tool immediately and inject the result into the message
        # thread BEFORE the first LLM call.  This makes the chatbot robust
        # against models that ignore tool_choice or system prompts.
        # ------------------------------------------------------------------
        intent_result = self._intent_service.detect_intent(request.message)
        try:
            intent_confidence = float(intent_result.confidence)
        except (TypeError, ValueError):
            intent_confidence = 0.0
        intent_topic = intent_result.topic

        # Context-aware disambiguation for known weak spots in deep validation.
        if _is_cross_topic_summary_query(request.message):
            intent_topic = ChatTopic.SYSTEM_OVERVIEW
            intent_confidence = max(intent_confidence, 0.8)
        elif _is_capacity_or_station_query(request.message):
            intent_topic = ChatTopic.FACILITY_INFO
            intent_confidence = max(intent_confidence, 0.85)
        elif _is_implicit_followup(request.message):
            previous_topic = _last_assistant_topic(history_messages)
            if previous_topic:
                intent_topic = previous_topic
                intent_confidence = max(intent_confidence, 0.8)
                locked_topic = previous_topic
            else:
                # Very short follow-ups like "chi so do..." in this app almost
                # always refer back to energy KPI context.
                intent_topic = ChatTopic.ENERGY_PERFORMANCE
                intent_confidence = max(intent_confidence, 0.8)
                locked_topic = ChatTopic.ENERGY_PERFORMANCE

        logger.info(
            "solar_chat_intent trace_id=%s topic=%s confidence=%.2f",
            trace_id,
            getattr(intent_topic, "value", str(intent_topic)),
            intent_confidence,
        )

        is_prompt_injection = _is_prompt_injection_request(request.message)

        if is_prompt_injection:
            refusal_answer = _build_scope_refusal(language)
            reason = "prompt_injection"
            logger.info(
                "solar_chat_scope_guard_refusal trace_id=%s reason=%s",
                trace_id,
                reason,
            )
            return self._finalize_response(
                request=request,
                answer=refusal_answer,
                topic=ChatTopic.GENERAL,
                metrics={},
                sources=[],
                model_used="scope-guard",
                fallback_used=True,
                started=started,
                trace_id=trace_id,
                intent_confidence=intent_confidence,
                warning_message="Prompt-injection request refused.",
                thinking_trace=ThinkingTrace(
                    summary="Scope guard refused an unsafe prompt-injection request.",
                    steps=[
                        ThinkingStep(
                            step="Scope guard",
                            detail=reason,
                            status="warning",
                        )
                    ],
                    trace_id=trace_id,
                ),
            )

        # For general intent, avoid agentic tool-calling entirely.
        # Let model follow scope-guard instructions from system prompt.
        if (
            intent_topic == ChatTopic.GENERAL
            and not _needs_web_search(request.message)
            and not _is_in_domain_query(request.message)
            and not _has_in_domain_history(history_messages)
        ):
            return self._finalize_response(
                request=request,
                answer=_build_scope_refusal(language),
                topic=ChatTopic.GENERAL,
                metrics={},
                sources=[],
                model_used="scope-guard",
                fallback_used=True,
                started=started,
                trace_id=trace_id,
                intent_confidence=intent_confidence,
                warning_message="General query handled with deterministic scope refusal.",
                thinking_trace=ThinkingTrace(
                    summary="General intent deterministic scope refusal.",
                    steps=[
                        ThinkingStep(
                            step="General scope guard",
                            detail="Tool-calling bypassed for general intent.",
                            status="success",
                        )
                    ],
                    trace_id=trace_id,
                ),
            )

        # ------------------------------------------------------------------
        # Deep planner (pre-agentic): a single structured LLM call that
        # enumerates ALL tools needed to answer the question.  Handles
        # compound/multi-intent prompts in one pass so synthesis sees the
        # full evidence set instead of a partial retrieval.  If the planner
        # returns any actions, we execute them and SKIP the legacy
        # single-tool intent pre-fetch below.
        # ------------------------------------------------------------------
        planner_used = False
        if (
            self._deep_planner.enabled
            and not _needs_web_search(request.message)
        ):
            try:
                plan = self._deep_planner.plan(
                    request.message, language, history_messages or None
                )
            except Exception as plan_err:
                logger.warning("solar_chat_deep_planner_error trace_id=%s error=%s", trace_id, plan_err)
                plan = None

            if plan and plan.actions:
                logger.info(
                    "solar_chat_plan trace_id=%s intent=%s actions=%s confidence=%.2f",
                    trace_id,
                    plan.intent_type,
                    [f"{a.tool}({sorted(a.arguments.keys())})" for a in plan.actions],
                    plan.confidence,
                )
                derived = self._execute_plan(
                    list(plan.actions), request.role, messages,
                    all_metrics, all_sources, step_events, trace_id,
                )
                if all_metrics:
                    planner_used = True
                    if derived is not None and locked_topic is None:
                        topic = derived

        # Skip intent pre-fetch when the user message references a specific
        # date (e.g. "ngày 10/4/2026").  Pre-fetch tools like
        # get_energy_performance don't accept date parameters, so their
        # results would be irrelevant.  Instead, directly pre-fetch
        # get_station_daily_report with the extracted anchor_date.
        user_query_date = extract_query_date(request.message, base_date=latest_mart_date)

        if planner_used:
            # Planner has already satisfied retrieval — fall through to the
            # fast-path synthesis block below, skipping the legacy single-tool
            # pre-fetch to avoid redundant calls.
            pass
        elif user_query_date is not None:
            # Date-specific query: pre-fetch station daily report with the
            # extracted date so the LLM has real data to synthesise from.
            prefetch_tool = "get_station_daily_report"
            allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
            if prefetch_tool in allowed_tools:
                try:
                    tool_args = {"anchor_date": user_query_date.isoformat()}
                    data, sources = self._tool_executor.execute(
                        prefetch_tool, tool_args, request.role
                    )
                    _merge_tool_result(all_metrics, prefetch_tool, data)
                    all_sources.extend(sources)
                    topic = ChatTopic.ENERGY_PERFORMANCE
                    messages.append({
                        "role": "model",
                        "parts": [{"functionCall": {"name": prefetch_tool, "args": tool_args}}],
                    })
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": prefetch_tool,
                            "response": data,
                        }}],
                    })
                    step_events.append({
                        "step": 0,
                        "tool": prefetch_tool,
                        "status": "ok",
                        "metric_keys": sorted(data.keys()),
                        "source_count": len(sources),
                    })
                    logger.info(
                        "solar_chat_date_prefetch trace_id=%s tool=%s date=%s metric_keys=%s",
                        trace_id,
                        prefetch_tool,
                        user_query_date.isoformat(),
                        sorted(data.keys()),
                    )
                except DatabricksDataUnavailableError as db_err:
                    logger.error(
                        "solar_chat_date_prefetch_databricks_unavailable trace_id=%s error=%s",
                        trace_id,
                        db_err,
                    )
                    raise
                except Exception as prefetch_err:
                    logger.warning(
                        "solar_chat_date_prefetch_failed trace_id=%s error=%s",
                        trace_id,
                        prefetch_err,
                    )
        elif (
            intent_topic != ChatTopic.GENERAL
            and intent_confidence >= 0.6
        ):
            primary_tool = _TOPIC_TO_PRIMARY_TOOL.get(intent_topic)
            allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
            if primary_tool and primary_tool in allowed_tools:
                try:
                    from app.services.solar_ai_chat.nlp_parser import parse_timeframe_days
                    tool_args = {}
                    
                    # Inject timeframe parameter if explicitly found in query
                    tf_days = parse_timeframe_days(request.message)
                    if tf_days is not None:
                        tool_args["timeframe_days"] = tf_days
                        
                    data, sources = self._tool_executor.execute(
                        primary_tool, tool_args, request.role
                    )
                    _merge_tool_result(all_metrics, primary_tool, data)
                    all_sources.extend(sources)
                    topic = intent_topic
                    # Inject as a completed tool call in the message thread
                    messages.append({
                        "role": "model",
                        "parts": [{"functionCall": {"name": primary_tool, "args": tool_args}}],
                    })
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": primary_tool,
                            "response": data,
                        }}],
                    })
                    step_events.append({
                        "step": 0,
                        "tool": primary_tool,
                        "status": "ok",
                        "metric_keys": sorted(data.keys()),
                        "source_count": len(sources),
                    })
                    logger.info(
                        "solar_chat_intent_prefetch trace_id=%s tool=%s metric_keys=%s",
                        trace_id,
                        primary_tool,
                        sorted(data.keys()),
                    )
                except DatabricksDataUnavailableError as db_err:
                    logger.error(
                        "solar_chat_intent_prefetch_databricks_unavailable trace_id=%s tool=%s error=%s",
                        trace_id,
                        primary_tool,
                        db_err,
                    )
                    raise
                except Exception as prefetch_err:
                    logger.warning(
                        "solar_chat_intent_prefetch_failed trace_id=%s tool=%s error=%s",
                        trace_id,
                        primary_tool,
                        prefetch_err,
                    )

        # ------------------------------------------------------------------
        # Fast-path synthesis: if the intent pre-fetch already loaded data,
        # pass it directly as evidence to the LLM so even models that ignore
        # tool_call messages still produce a data-grounded answer.
        # Skipped when the user explicitly requests a web search (handled
        # by the direct web-search path below).
        # ------------------------------------------------------------------
        if all_metrics and not _needs_web_search(request.message):
            evidence_text = json.dumps(all_metrics, ensure_ascii=False)
            synthesis_prompt = build_synthesis_prompt(
                user_message=request.message,
                evidence_text=evidence_text,
                language=language,
                history=history_messages or None,
            )
            logger.info(
                "solar_chat_prefetch_synthesis trace_id=%s topic=%s evidence_keys=%s",
                trace_id,
                topic.value,
                sorted(all_metrics.keys()),
            )
            try:
                gen = self._model_router.generate(
                    synthesis_prompt,
                    max_output_tokens=self._synthesis_max_output_tokens,
                    temperature=0.1,
                )
                answer = gen.text
                model_used = gen.model_used
                fallback_used = gen.fallback_used
            except Exception:
                answer = build_data_only_summary(all_metrics, all_sources, language)
                fallback_used = True

        # ------------------------------------------------------------------
        # Agentic loop — LLM may call additional tools or synthesise directly
        # (only entered if fast-path did not already set answer)
        # ------------------------------------------------------------------
        allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
        role_tool_declarations = [td for td in TOOL_DECLARATIONS if td.get("name") in allowed_tools]
        # Apply caller-side tool_mode filter on top of RBAC palette
        if getattr(request, "tool_mode", "auto") == "selected" and request.allowed_tools:
            _user_allowed = set(request.allowed_tools)
            filtered = [td for td in role_tool_declarations if td.get("name") in _user_allowed]
            if filtered:
                role_tool_declarations = filtered

        for step_num in range(1, self._max_tool_steps + 1):
            if answer is not None:
                break
            try:
                result = self._model_router.generate_with_tools(
                    messages,
                    role_tool_declarations,
                    max_output_tokens=self._synthesis_max_output_tokens,
                    require_function_call=(step_num == 1 and not all_metrics),
                )
            except ToolCallNotSupportedError:
                # Model does not support native tool calling -> evidence-in-prompt fallback
                evidence_text = json.dumps(all_metrics, ensure_ascii=False)
                synthesis_prompt = build_synthesis_prompt(
                    user_message=request.message,
                    evidence_text=evidence_text,
                    language=language,
                    history=history_messages or None,
                )
                try:
                    gen = self._model_router.generate(
                        synthesis_prompt,
                        max_output_tokens=self._synthesis_max_output_tokens,
                        temperature=0.1,
                    )
                    answer = gen.text
                    model_used = gen.model_used
                    fallback_used = gen.fallback_used
                except Exception:
                    answer = build_data_only_summary(all_metrics, all_sources, language)
                    fallback_used = True
                break
            except ModelUnavailableError:
                raise

            if result.function_call is None:
                # LLM produced a final text response
                answer = result.text or build_data_only_summary(all_metrics, all_sources, language)
                model_used = result.model_used
                fallback_used = result.fallback_used
                break

            # LLM requested a tool call
            tool_name = result.function_call.name
            tool_args = dict(result.function_call.arguments or {})
            logger.info(
                "solar_chat_agentic_tool_call trace_id=%s step=%d tool=%s",
                trace_id,
                step_num,
                tool_name,
            )

            # Append the model's tool call to the conversation
            messages.append({
                "role": "model",
                "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
            })

            # Handle answer_directly (LLM will produce text on the next turn)
            if tool_name == "answer_directly":
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": {"status": "ok"}}}],
                })
                step_events.append({"step": step_num, "tool": tool_name, "status": "skipped"})
                continue

            # Handle web_lookup
            if tool_name == "web_lookup":
                web_response = self._execute_web_lookup(tool_args)
                # Propagate web URLs into sources
                for wr in web_response.get("results", []):
                    wr_url = wr.get("url", "")
                    if wr_url:
                        all_sources.append({
                            "layer": "Web",
                            "dataset": wr.get("title", wr_url),
                            "data_source": "web_search",
                            "url": wr_url,
                        })
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": web_response}}],
                })
                step_events.append({
                    "step": step_num,
                    "tool": tool_name,
                    "status": "ok",
                    "result_count": len(web_response.get("results", [])),
                })
                continue

            # RBAC check
            allowed_tools = ROLE_TOOL_PERMISSIONS.get(request.role, set())
            if tool_name not in allowed_tools:
                logger.info(
                    "solar_chat_agentic_permission_denied trace_id=%s role=%s tool=%s",
                    trace_id,
                    request.role.value,
                    tool_name,
                )
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {
                        "name": tool_name,
                        "response": {"error": f"Access denied for role '{request.role.value}'."},
                    }}],
                })
                step_events.append({"step": step_num, "tool": tool_name, "status": "denied"})
                continue

            # Execute data tool
            try:
                data, sources = self._tool_executor.execute(tool_name, tool_args, request.role)
                all_sources.extend(sources)
                _merge_tool_result(all_metrics, tool_name, data)
                if tool_name in TOOL_NAME_TO_TOPIC and locked_topic is None:
                    topic = ChatTopic(TOOL_NAME_TO_TOPIC[tool_name])
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": data}}],
                })
                step_events.append({
                    "step": step_num,
                    "tool": tool_name,
                    "status": "ok",
                    "metric_keys": sorted(data.keys()),
                    "source_count": len(sources),
                })
                logger.info(
                    "solar_chat_agentic_tool_done trace_id=%s step=%d tool=%s metric_keys=%s sources=%d",
                    trace_id,
                    step_num,
                    tool_name,
                    sorted(data.keys()),
                    len(sources),
                )
            except DatabricksDataUnavailableError:
                raise
            except Exception as tool_err:
                logger.warning(
                    "solar_chat_agentic_tool_error trace_id=%s step=%d tool=%s error=%s",
                    trace_id,
                    step_num,
                    tool_name,
                    tool_err,
                )
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {
                        "name": tool_name,
                        "response": {"error": str(tool_err)},
                    }}],
                })
                step_events.append({
                    "step": step_num,
                    "tool": tool_name,
                    "status": "error",
                    "detail": str(tool_err),
                })

        # Force final synthesis if loop ran out of steps without a text answer
        if answer is None:
            logger.info("solar_chat_agentic_force_synthesis trace_id=%s", trace_id)
            try:
                force_result = self._model_router.generate_with_tools(
                    messages,
                    [],  # No tools -> LLM must produce text
                    max_output_tokens=self._synthesis_max_output_tokens,
                )
                answer = force_result.text or build_data_only_summary(all_metrics, all_sources, language)
                model_used = force_result.model_used
                fallback_used = force_result.fallback_used
            except Exception:
                answer = build_data_only_summary(all_metrics, all_sources, language)
                fallback_used = True

        def _fmt_detail(ev: dict) -> str:
            keys = ev.get("metric_keys")
            if keys and isinstance(keys, list):
                return f"{len(keys)} metric{'s' if len(keys) != 1 else ''} retrieved"
            rc = ev.get("result_count")
            if rc is not None:
                return f"{rc} result{'s' if int(rc) != 1 else ''} found"
            det = ev.get("detail")
            if det:
                return str(det)[:120]
            return ev.get("status", "ok")

        topic_label_str = topic.value.replace("_", " ").title()
        thinking_trace = ThinkingTrace(
            summary=f"{len(step_events)} tool call{'s' if len(step_events) != 1 else ''} · {topic_label_str} · {model_used}",
            steps=[
                ThinkingStep(
                    step=tool_label(ev.get("tool", "")),
                    detail=_fmt_detail(ev),
                    status="success" if ev.get("status") in ("ok", "skipped") else "warning",
                )
                for ev in step_events
            ],
            trace_id=trace_id,
        )


        source_objects: list[SourceMetadata] = [
            SourceMetadata(**s) if isinstance(s, dict) else s
            for s in all_sources
        ]

        if locked_topic is not None:
            topic = locked_topic

        return self._finalize_response(
            request=request,
            answer=answer,
            topic=topic,
            metrics=all_metrics,
            sources=source_objects,
            model_used=model_used,
            fallback_used=fallback_used,
            started=started,
            trace_id=trace_id,
            intent_confidence=TOOL_PATH_CONFIDENCE,
            thinking_trace=thinking_trace,
        )

    def _execute_plan(
        self,
        actions: list[Any],
        role: ChatRole,
        messages: list[dict[str, Any]],
        all_metrics: dict[str, Any],
        all_sources: list[dict[str, str]],
        step_events: list[dict[str, Any]],
        trace_id: str,
    ) -> ChatTopic | None:
        """Execute a planner's action list. Appends results to the provided
        containers so downstream synthesis sees all collected evidence.

        Returns the topic derived from the first data-producing tool (for
        response metadata).  web_lookup is handled here; answer_directly is
        skipped (no data fetch).  RBAC-denied actions are recorded but do not
        abort the pipeline.
        """
        allowed_tools = ROLE_TOOL_PERMISSIONS.get(role, set())
        derived_topic: ChatTopic | None = None
        step_index = len(step_events)

        for action in actions[:MAX_PLANNED_ACTIONS]:
            tool_name = str(getattr(action, "tool", "") or "")
            if not tool_name:
                continue
            tool_args = dict(getattr(action, "arguments", {}) or {})
            step_index += 1

            if tool_name == "answer_directly":
                step_events.append({
                    "step": step_index, "tool": tool_name, "status": "skipped",
                    "detail": getattr(action, "rationale", "") or "planner: no data needed",
                })
                continue

            if tool_name == "web_lookup":
                try:
                    web_response = self._execute_web_lookup(tool_args)
                    results = web_response.get("results", [])
                    for wr in results:
                        wr_url = wr.get("url", "")
                        if wr_url:
                            all_sources.append({
                                "layer": "Web", "dataset": wr.get("title", wr_url),
                                "data_source": "web_search", "url": wr_url,
                            })
                    messages.append({
                        "role": "model",
                        "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
                    })
                    messages.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": tool_name, "response": web_response}}],
                    })
                    step_events.append({
                        "step": step_index, "tool": tool_name, "status": "ok",
                        "result_count": len(results),
                    })
                except Exception as err:
                    step_events.append({
                        "step": step_index, "tool": tool_name, "status": "error", "detail": str(err)[:120],
                    })
                continue

            if tool_name not in allowed_tools:
                step_events.append({
                    "step": step_index, "tool": tool_name, "status": "denied",
                })
                continue

            try:
                data, sources = self._tool_executor.execute(tool_name, tool_args, role)
                _merge_tool_result(all_metrics, tool_name, data)
                all_sources.extend(sources)
                messages.append({
                    "role": "model",
                    "parts": [{"functionCall": {"name": tool_name, "args": tool_args}}],
                })
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": tool_name, "response": data}}],
                })
                step_events.append({
                    "step": step_index, "tool": tool_name, "status": "ok",
                    "metric_keys": sorted(data.keys()), "source_count": len(sources),
                })
                if derived_topic is None and tool_name in TOOL_NAME_TO_TOPIC:
                    derived_topic = ChatTopic(TOOL_NAME_TO_TOPIC[tool_name])
                logger.info(
                    "solar_chat_plan_exec trace_id=%s tool=%s metric_keys=%s sources=%d",
                    trace_id, tool_name, sorted(data.keys()), len(sources),
                )
            except DatabricksDataUnavailableError:
                raise
            except Exception as err:
                step_events.append({
                    "step": step_index, "tool": tool_name, "status": "error", "detail": str(err)[:120],
                })
                logger.warning(
                    "solar_chat_plan_exec_failed trace_id=%s tool=%s error=%s",
                    trace_id, tool_name, err,
                )

        return derived_topic

    def _execute_web_lookup(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Web search is disabled; always returns a disabled-error payload so
        any overlooked call site yields a safe response instead of raising."""
        _ = tool_args  # noqa: B007
        return {"error": "Web search is not configured or is disabled."}

    # ------------------------------------------------------------------
    # Response finalization
    # ------------------------------------------------------------------

    def _finalize_response(
        self,
        request: SolarChatRequest,
        answer: str,
        topic: ChatTopic,
        metrics: dict[str, Any],
        sources: list[SourceMetadata],
        model_used: str,
        fallback_used: bool,
        started: float,
        trace_id: str,
        intent_confidence: float,
        warning_message: str | None = None,
        thinking_trace: ThinkingTrace | None = None,
    ) -> SolarChatResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        viz_requested_flag = _should_visualize(
            request.message, getattr(request, "tool_hints", None),
        )

        # Build viz FIRST so we persist the exact same snapshot we render live.
        data_table = None
        chart = None
        kpi_cards = None
        try:
            data_table, chart, kpi_cards = self._chart_spec_builder.build(
                metrics,
                topic=topic.value if hasattr(topic, "value") else str(topic),
                user_query=request.message,
            )
            # Chart is opt-in: only keep it when the user explicitly asked
            # (via the Visualize pill or a viz-related keyword in the prompt).
            if not viz_requested_flag:
                chart = None
        except Exception as viz_err:
            logger.warning(
                "solar_chat_viz_build_failed trace_id=%s error_type=%s error=%s",
                trace_id,
                type(viz_err).__name__,
                viz_err,
            )

        viz_snapshot: dict | None = None
        if data_table or chart or kpi_cards:
            viz_snapshot = {
                "data_table": data_table.model_dump() if data_table else None,
                "chart": chart.model_dump() if chart else None,
                "kpi_cards": kpi_cards.model_dump() if kpi_cards else None,
            }

        try:
            self._persist_exchange(
                session_id=request.session_id,
                user_message=request.message,
                answer=answer,
                topic=topic,
                sources=sources,
                thinking_trace=thinking_trace,
                key_metrics=metrics if metrics else None,
                viz_requested=viz_requested_flag,
                viz_payload=viz_snapshot,
            )
        except Exception as persist_error:
            logger.warning(
                "solar_chat_history_persist_failed trace_id=%s session_id=%s error_type=%s error=%s",
                trace_id,
                request.session_id,
                type(persist_error).__name__,
                persist_error,
            )
        logger.info(
            "solar_chat_trace_end trace_id=%s topic=%s role=%s model=%s fallback=%s latency_ms=%d warning=%s metrics_keys=%s",
            trace_id,
            topic.value,
            request.role.value,
            model_used,
            fallback_used,
            latency_ms,
            warning_message or "",
            sorted(list(metrics.keys())),
        )
        return SolarChatResponse(
            answer=answer,
            topic=topic,
            role=request.role,
            sources=sources,
            key_metrics=metrics,
            model_used=model_used,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            intent_confidence=intent_confidence,
            warning_message=warning_message,
            thinking_trace=thinking_trace,
            ui_features=resolve_ui_features(request.role),
            data_table=data_table,
            chart=chart,
            kpi_cards=kpi_cards,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _load_history(self, session_id: str | None) -> list[ChatMessage]:
        if not session_id or not self._history_repository:
            return []
        return self._history_repository.get_recent_messages(session_id, limit=10)

    def _persist_exchange(
        self,
        session_id: str | None,
        user_message: str,
        answer: str,
        topic: ChatTopic,
        sources: list[SourceMetadata],
        thinking_trace: ThinkingTrace | None = None,
        key_metrics: dict | None = None,
        viz_requested: bool = False,
        viz_payload: dict | None = None,
    ) -> None:
        if not session_id or not self._history_repository:
            return
        self._history_repository.add_message(session_id=session_id, sender="user", content=user_message)
        # Some repository implementations (e.g. tests, Databricks) may not
        # accept the extra kwargs — fall back gracefully.
        try:
            self._history_repository.add_message(
                session_id=session_id,
                sender="assistant",
                content=answer,
                topic=topic,
                sources=sources,
                thinking_trace=thinking_trace,
                key_metrics=key_metrics,
                viz_requested=viz_requested,
                viz_payload=viz_payload,
            )
        except TypeError:
            self._history_repository.add_message(
                session_id=session_id,
                sender="assistant",
                content=answer,
                topic=topic,
                sources=sources,
                thinking_trace=thinking_trace,
            )

    # ------------------------------------------------------------------
    # Backward-compat static helpers (used by existing unit tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_extreme_metric_query(message: str) -> ExtremeMetricQuery | None:
        return extract_extreme_metric_query(message)

    @staticmethod
    def _extract_timeframe(message: str, specific_hour: int | None = None) -> str:
        normed = normalize_vietnamese_text(message)
        return extract_timeframe(normed, specific_hour=specific_hour)

    @staticmethod
    def _short(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _normalize_for_matching(value: str) -> str:
        lowered = str(value or "").strip().lower()
        without_marks = normalize("NFD", lowered)
        return "".join(c for c in without_marks if ord(c) < 128)

    def _validate_role(self, topic: ChatTopic, role: ChatRole) -> None:
        allowed_topics = ROLE_TOPIC_PERMISSIONS.get(role, set())
        if topic not in allowed_topics:
            raise PermissionError(
                f"Role '{role.value}' is not allowed to access topic '{topic.value}'."
            )
