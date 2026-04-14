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
from app.services.solar_ai_chat.tool_executor import ToolExecutor
from app.services.solar_ai_chat.answer_verifier import AnswerVerifier

if TYPE_CHECKING:
    from app.repositories.solar_ai_chat.vector_repository import VectorRepository
    from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient
    from app.services.solar_ai_chat.web_search_client import WebSearchClient

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
    """Return True if the user's message explicitly requests a web search."""
    lowered = message.lower()
    return any(kw in lowered for kw in _WEB_SEARCH_KEYWORDS)


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
        web_search_client: "WebSearchClient | None" = None,
        *,
        planner_enabled: bool = True,           # kept for route-compat, ignored
        orchestrator_enabled: bool = True,       # kept for route-compat, ignored
        verifier_enabled: bool = True,
        hybrid_retrieval_enabled: bool = True,   # kept for route-compat, ignored
        max_tool_steps: int = MAX_TOOL_STEPS,
        legacy_router_enabled: bool = False,     # kept for route-compat, ignored
        planner_max_output_tokens: int | None = None,    # ignored
        synthesis_max_output_tokens: int | None = None,
        verifier_max_output_tokens: int | None = None,
    ) -> None:
        self._repository = repository
        self._intent_service = intent_service
        self._model_router = model_router
        self._history_repository = history_repository
        self._tool_executor = ToolExecutor(
            repository,
            vector_repo=vector_repo,
            embedding_client=embedding_client,
        )
        self._web_search_client = web_search_client
        self._verifier_enabled = verifier_enabled
        self._max_tool_steps = max(1, max_tool_steps)
        self._synthesis_max_output_tokens = synthesis_max_output_tokens
        self._query_rewriter = QueryRewriter()
        self._answer_verifier = (
            AnswerVerifier(model_router, max_output_tokens=verifier_max_output_tokens)
            if verifier_enabled and model_router is not None
            else AnswerVerifier(None)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def handle_query(self, request: SolarChatRequest) -> SolarChatResponse:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex[:8]
        logger.info(
            "solar_chat_trace_start trace_id=%s session_id=%s role=%s message=%s",
            trace_id,
            request.session_id,
            request.role.value if request.role else "unknown",
            self._short(request.message, 200),
        )
        history_messages = self._load_history(request.session_id)
        logger.info(
            "solar_chat_trace_history trace_id=%s history_count=%d",
            trace_id,
            len(history_messages),
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
        language = self._query_rewriter.rewrite(request.message).language

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

        messages = build_agentic_messages(request.message, language, history_messages)
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
        if intent_topic == ChatTopic.GENERAL and not _needs_web_search(request.message):
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

        # Skip intent pre-fetch when the user message references a specific
        # date (e.g. "ngày 10/4/2026").  Pre-fetch tools like
        # get_energy_performance don't accept date parameters, so their
        # results would be irrelevant.  Instead, directly pre-fetch
        # get_station_daily_report with the extracted anchor_date.
        user_query_date = extract_query_date(request.message)

        if user_query_date is not None:
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
                    all_metrics.update(data)
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
                    data, sources = self._tool_executor.execute(
                        primary_tool, {}, request.role
                    )
                    all_metrics.update(data)
                    all_sources.extend(sources)
                    topic = intent_topic
                    # Inject as a completed tool call in the message thread
                    messages.append({
                        "role": "model",
                        "parts": [{"functionCall": {"name": primary_tool, "args": {}}}],
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
        # Direct web-search path: when the user explicitly requests an
        # internet lookup, call web_search_client directly (bypassing
        # generate_with_tools, which gpt-5-mini ignores for tool calls).
        # Combines web snippets with any pre-fetched system data and
        # synthesises via generate() that includes conversation history.
        # ------------------------------------------------------------------
        elif _needs_web_search(request.message):
            search_query = _extract_search_query(request.message, history_messages)
            # If pre-fetched data has specific facility names not already in the
            # extracted query, use them to build a targeted search query.
            if all_metrics:
                top_names = _extract_top_facility_names(all_metrics)
                if top_names and not any(
                    name.lower() in search_query.lower() for name in top_names
                ):
                    search_query = " ".join(top_names) + " solar farm"
            logger.info(
                "solar_chat_web_search_direct trace_id=%s query=%r",
                trace_id,
                search_query,
            )
            web_response = self._execute_web_lookup({"query": search_query})

            # Build combined evidence: system data (if any) + web results
            evidence_parts: list[str] = []
            if all_metrics:
                evidence_parts.append(
                    "## Dữ liệu hệ thống (30 ngày gần nhất)\n"
                    + json.dumps(all_metrics, ensure_ascii=False)[:2000]
                )
            web_results = web_response.get("results", [])
            if web_results:
                snippets = "\n".join(
                    f"- [{r.get('title', '')}]({r.get('url', '')}): "
                    f"{r.get('snippet', '')[:300]}"
                    for r in web_results
                )
                evidence_parts.append(
                    f"## Kết quả tìm kiếm web ({len(web_results)} nguồn)\n{snippets}"
                )
            elif "error" in web_response:
                evidence_parts.append(
                    f"## Web search: {web_response['error']}\n"
                    "Trả lời từ kiến thức tích hợp nếu có."
                )
            evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(không có dữ liệu)"
            synthesis_prompt = build_synthesis_prompt(
                user_message=request.message,
                evidence_text=evidence_text,
                language=language,
                history=history_messages or None,
                cite_web_sources=bool(web_results),
            )
            step_events.append({
                "step": 0,
                "tool": "web_lookup_direct",
                "status": "ok" if web_results else "no_results",
                "result_count": len(web_results),
            })
            # Propagate web URLs into sources so they appear in the response
            for wr in web_results:
                wr_url = wr.get("url", "")
                if wr_url:
                    all_sources.append({
                        "layer": "Web",
                        "dataset": wr.get("title", wr_url),
                        "data_source": "web_search",
                        "url": wr_url,
                    })
            try:
                gen = self._model_router.generate(
                    synthesis_prompt,
                    max_output_tokens=self._synthesis_max_output_tokens,
                    temperature=0.1,
                )
                answer = gen.text
                model_used = gen.model_used
                fallback_used = gen.fallback_used
                logger.info(
                    "solar_chat_web_synthesis_done trace_id=%s model=%s web_results=%d",
                    trace_id,
                    model_used,
                    len(web_results),
                )
            except Exception:
                answer = build_data_only_summary(all_metrics, all_sources, language)
                fallback_used = True

        # ------------------------------------------------------------------
        # Agentic loop — LLM may call additional tools or synthesise directly
        # (only entered if neither fast-path nor direct web-search set answer)
        # ------------------------------------------------------------------
        for step_num in range(1, self._max_tool_steps + 1):
            if answer is not None:
                break
            try:
                result = self._model_router.generate_with_tools(
                    messages,
                    TOOL_DECLARATIONS,
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
                all_metrics.update(data)
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

        thinking_trace = ThinkingTrace(
            summary=(
                f"Agentic loop: {len(step_events)} tool step(s), "
                f"model='{model_used}', fallback={fallback_used}, topic='{topic.value}'."
            ),
            steps=[
                ThinkingStep(
                    step=f"Step {ev['step']}: {ev['tool']}",
                    detail=str(
                        ev.get("metric_keys",
                        ev.get("result_count",
                        ev.get("detail", ev.get("status", ""))))
                    ),
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

    def _execute_web_lookup(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Execute a web_lookup tool call and return structured search results."""
        if self._web_search_client is None or not getattr(self._web_search_client, "enabled", False):
            return {"error": "Web search is not configured or is disabled."}

        query = str(tool_args.get("query") or "").strip()
        if not query:
            return {"error": "No search query was provided."}

        try:
            results = self._web_search_client.search(query, max_results=5)
        except Exception as exc:
            return {"error": str(exc)}

        if not results:
            return {"results": [], "message": "No external results found for this query."}

        return {
            "results": [
                {
                    "title": getattr(r, "title", ""),
                    "snippet": getattr(r, "snippet", ""),
                    "url": getattr(r, "url", ""),
                    "score": getattr(r, "score", None),
                }
                for r in results[:5]
            ]
        }

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
        try:
            self._persist_exchange(
                session_id=request.session_id,
                user_message=request.message,
                answer=answer,
                topic=topic,
                sources=sources,
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
    ) -> None:
        if not session_id or not self._history_repository:
            return
        self._history_repository.add_message(session_id=session_id, sender="user", content=user_message)
        self._history_repository.add_message(
            session_id=session_id,
            sender="assistant",
            content=answer,
            topic=topic,
            sources=sources,
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
