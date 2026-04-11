"""Solar AI Chat service — orchestration layer.

Coordinates intent detection, RBAC, data retrieval, LLM generation,
and response finalization. NLP parsing is in nlp_parser.py and
prompt/summary building is in prompt_builder.py.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any
from unicodedata import normalize

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
from app.services.solar_ai_chat.intent_service import IntentDetectionResult, VietnameseIntentService
from app.services.solar_ai_chat.nlp_parser import (
    ExtremeMetricQuery,
    extract_extreme_metric_query,
    extract_query_date,
    resolve_weather_metric,
    topic_for_extreme_metric,
)
from app.services.solar_ai_chat.permissions import ROLE_TOPIC_PERMISSIONS
from app.services.solar_ai_chat.prompt_builder import (
    build_fallback_summary,
    build_prompt,
    build_tool_messages,
    format_source_text,
)
from app.services.solar_ai_chat.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from app.repositories.solar_ai_chat.vector_repository import VectorRepository
    from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient
    from app.services.solar_ai_chat.web_search_client import WebSearchClient, WebSearchResult

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Intent confidence levels for response metadata
TOOL_PATH_CONFIDENCE = 0.95
TEXT_ONLY_CONFIDENCE = 0.85
REGEX_EXTREME_CONFIDENCE = 0.92
MAX_TOOL_STEPS = 2
LLM_PLANNER_MIN_CONFIDENCE = 0.55
LLM_PLANNER_HISTORY_LIMIT = 4
WEBSEARCH_DEFINITION_MARKERS = (
    "la gi",
    "nghia la gi",
    "dinh nghia",
    "giai thich",
    "what is",
    "define",
    "definition",
    "meaning",
)
WEBSEARCH_EXTERNAL_REQUEST_MARKERS = (
    "tren internet",
    "internet",
    "web",
    "online",
    "google",
    "search",
    "tra cuu",
    "tham khao ben ngoai",
    "external source",
)
WEBSEARCH_DATA_REQUEST_MARKERS = (
    "top",
    "xep hang",
    "rank",
    "cao nhat",
    "thap nhat",
    "bao nhieu",
    "how much",
    "bao cao",
    "report",
)
WEBSEARCH_SYSTEM_KEYWORDS = (
    "solar",
    "pv",
    "photovoltaic",
    "facility",
    "station",
    "tram",
    "nha may",
    "capacity",
    "cong suat",
    "timezone",
    "location",
    "nang luong",
    "energy",
    "performance ratio",
    "capacity factor",
    "forecast",
    "inverter",
    "irradiance",
    "aqi",
    "weather",
    "lakehouse",
)
WEBSEARCH_TRUSTED_DOMAIN_HINTS = (
    "nrel.gov",
    "energy.gov",
    "iea.org",
    "wikipedia.org",
    "sandia.gov",
    "pvpmc.sandia.gov",
    "irena.org",
)


class SolarAIChatService:
    """Orchestrates intent routing, RBAC, data retrieval, and LLM response."""

    def __init__(
        self,
        repository: SolarChatRepository,
        intent_service: VietnameseIntentService,
        model_router: LLMModelRouter | None,
        history_repository: Any | None = None,
        vector_repo: "VectorRepository | None" = None,
        embedding_client: "GeminiEmbeddingClient | None" = None,
        web_search_client: "WebSearchClient | None" = None,
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
        # Learns at runtime whether the current model stack can reliably call tools.
        # Once disabled, requests go straight to deterministic data path for lower latency.
        self._tool_path_supported: bool | None = None

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

        if self._model_router is not None and self._tool_path_supported is not False:
            try:
                return self._handle_with_tools(request, started, history_messages, trace_id)
            except ToolCallNotSupportedError as tool_error:
                self._tool_path_supported = False
                logger.warning(
                    "solar_chat_tool_path_tooling_unsupported trace_id=%s error=%s, falling back to regex intent",
                    trace_id,
                    tool_error,
                )
                return self._handle_with_regex_fallback(
                    request,
                    started,
                    history_messages,
                    trace_id=trace_id,
                    skip_model_generation=True,
                    allow_llm_planner=True,
                    warning_message=(
                        "Current model does not support reliable tool-calling. "
                        "Returned a deterministic data-backed summary."
                    ),
                )
            except ModelUnavailableError as model_error:
                logger.warning(
                    "solar_chat_tool_path_unavailable trace_id=%s error=%s, falling back to regex intent",
                    trace_id,
                    model_error,
                )
                return self._handle_with_regex_fallback(
                    request,
                    started,
                    history_messages,
                    trace_id=trace_id,
                    skip_model_generation=True,
                    warning_message=(
                        "LLM service is temporarily unavailable (429/503). "
                        "Returned a data-backed summary."
                    ),
                )
            except Exception as tool_err:
                logger.exception(
                    "solar_chat_tool_path_failed trace_id=%s error_type=%s error=%s, falling back to regex intent",
                    trace_id,
                    type(tool_err).__name__,
                    tool_err,
                )

        if self._model_router is not None and self._tool_path_supported is False:
            logger.info("solar_chat_tool_path_bypassed trace_id=%s reason=unsupported_cached", trace_id)
            try:
                cached_extreme = extract_extreme_metric_query(request.message)
                if cached_extreme is not None:
                    cached_topic = topic_for_extreme_metric(cached_extreme.metric_name)
                else:
                    cached_topic = self._intent_service.detect_intent(request.message).topic
            except Exception:
                cached_topic = ChatTopic.GENERAL

            skip_model_generation = cached_topic is not ChatTopic.GENERAL
            logger.info(
                "solar_chat_trace_cached_route trace_id=%s topic=%s skip_model_generation=%s",
                trace_id,
                cached_topic.value,
                skip_model_generation,
            )
            return self._handle_with_regex_fallback(
                request=request,
                started=started,
                history_messages=history_messages,
                trace_id=trace_id,
                skip_model_generation=skip_model_generation,
                allow_llm_planner=skip_model_generation,
                warning_message=(
                    "Tool-calling is unavailable for current model stack. "
                    "Returned deterministic data-backed summary."
                ) if skip_model_generation else None,
            )

        return self._handle_with_regex_fallback(
            request,
            started,
            history_messages,
            trace_id=trace_id,
        )

    def _handle_with_tools(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
    ) -> SolarChatResponse:
        messages = build_tool_messages(
            request_message=request.message,
            role_value=request.role.value,
            history=history_messages,
        )
        
        all_metrics: dict[str, Any] = {}
        all_sources: list[SourceMetadata] = []
        topic = ChatTopic.GENERAL
        last_model_used = ""
        last_fallback_used = False
        tool_failures = 0
        trace_steps: list[ThinkingStep] = [
            ThinkingStep(
                step="Routing mode",
                detail="Started tool-assisted routing with structured function-calling.",
                status="info",
            )
        ]

        logger.info("solar_chat_trace_tool_loop_start trace_id=%s max_steps=%d", trace_id, MAX_TOOL_STEPS)
        for step_idx in range(1, MAX_TOOL_STEPS + 1):
            tool_result = self._model_router.generate_with_tools(messages, TOOL_DECLARATIONS)
            last_model_used = tool_result.model_used
            last_fallback_used = tool_result.fallback_used
            logger.info(
                "solar_chat_trace_tool_step trace_id=%s step=%d model=%s fallback=%s has_text=%s has_function_call=%s",
                trace_id,
                step_idx,
                last_model_used,
                last_fallback_used,
                tool_result.text is not None,
                tool_result.function_call is not None,
            )

            if tool_result.text is not None:
                if not all_metrics:
                    inferred_topic = self._detect_topic_from_text(request.message)
                    if inferred_topic is not ChatTopic.GENERAL:
                        self._tool_path_supported = False
                        logger.warning(
                            "solar_chat_tool_path_text_without_data trace_id=%s topic=%s model=%s; forcing deterministic regex fallback",
                            trace_id,
                            inferred_topic.value,
                            last_model_used,
                        )
                        return self._handle_with_regex_fallback(
                            request=request,
                            started=started,
                            history_messages=history_messages,
                            trace_id=trace_id,
                            skip_model_generation=True,
                            warning_message=(
                                "Primary model response did not include tool calls. "
                                "Switched to deterministic data-backed route."
                            ),
                            trace_steps_seed=trace_steps
                            + [
                                ThinkingStep(
                                    step="Tool-calling unavailable",
                                    detail=(
                                        "Primary model returned plain text without function calls, "
                                        "so the service switched to deterministic data route."
                                    ),
                                    status="warning",
                                )
                            ],
                        )
                    topic = inferred_topic
                    logger.info(
                        "solar_chat_trace_text_only_response trace_id=%s inferred_topic=%s",
                        trace_id,
                        topic.value,
                    )
                trace_steps.append(
                    ThinkingStep(
                        step="Model response",
                        detail="Model returned final text response.",
                        status="success",
                    )
                )
                self._tool_path_supported = True
                thinking_trace = self._build_thinking_trace(
                    trace_id=trace_id,
                    topic=topic,
                    model_used=last_model_used,
                    fallback_used=last_fallback_used,
                    intent_confidence=(
                        TOOL_PATH_CONFIDENCE if all_metrics else TEXT_ONLY_CONFIDENCE
                    ),
                    sources=all_sources,
                    warning_message=None,
                    steps=trace_steps,
                )
                return self._finalize_response(
                    request=request,
                    answer=tool_result.text,
                    topic=topic,
                    metrics=all_metrics,
                    sources=all_sources,
                    model_used=last_model_used,
                    fallback_used=last_fallback_used,
                    started=started,
                    trace_id=trace_id,
                    intent_confidence=TOOL_PATH_CONFIDENCE if all_metrics else TEXT_ONLY_CONFIDENCE,
                    thinking_trace=thinking_trace,
                )

            fc = tool_result.function_call
            if fc is None:
                logger.info("solar_chat_trace_tool_loop_break trace_id=%s reason=no_function_call", trace_id)
                break

            logger.info(
                "solar_chat_trace_tool_call trace_id=%s step=%d tool=%s args=%s",
                trace_id,
                step_idx,
                fc.name,
                self._short(fc.arguments, 400),
            )
            trace_steps.append(
                ThinkingStep(
                    step=f"Tool call {step_idx}",
                    detail=f"Requested tool '{fc.name}'.",
                    status="info",
                )
            )
                
            try:
                metrics, source_rows = self._tool_executor.execute(fc.name, fc.arguments, request.role)
                self._tool_path_supported = True
                all_metrics.update(metrics)
                for row in source_rows:
                    sm = SourceMetadata(**row)
                    if sm not in all_sources:
                        all_sources.append(sm)

                if topic == ChatTopic.GENERAL:
                    topic = ChatTopic(TOOL_NAME_TO_TOPIC.get(fc.name, "general"))

                logger.info(
                    "solar_chat_trace_tool_result trace_id=%s step=%d tool=%s topic=%s metric_keys=%s source_count=%d",
                    trace_id,
                    step_idx,
                    fc.name,
                    topic.value,
                    sorted(list(metrics.keys())),
                    len(source_rows),
                )
                trace_steps.append(
                    ThinkingStep(
                        step=f"Tool result {step_idx}",
                        detail=(
                            f"Tool '{fc.name}' returned {len(metrics)} metric group(s) "
                            f"and {len(source_rows)} source row(s)."
                        ),
                        status="success",
                    )
                )

                if self._is_station_daily_report_no_data(fc.name, metrics):
                    answer = self._build_station_report_no_data_answer(metrics)
                    logger.info("station_daily_report_no_data_guard trace_id=%s triggered", trace_id)
                    trace_steps.append(
                        ThinkingStep(
                            step="No-data guard",
                            detail="Applied deterministic no-data guard for station daily report.",
                            status="warning",
                        )
                    )
                    thinking_trace = self._build_thinking_trace(
                        trace_id=trace_id,
                        topic=topic,
                        model_used="deterministic-summary",
                        fallback_used=True,
                        intent_confidence=TOOL_PATH_CONFIDENCE,
                        sources=all_sources,
                        warning_message=None,
                        steps=trace_steps,
                    )
                    return self._finalize_response(
                        request=request,
                        answer=answer,
                        topic=topic,
                        metrics=all_metrics,
                        sources=all_sources,
                        model_used="deterministic-summary",
                        fallback_used=True,
                        started=started,
                        trace_id=trace_id,
                        intent_confidence=TOOL_PATH_CONFIDENCE,
                        thinking_trace=thinking_trace,
                    )

                messages.append({"role": "model", "parts": [{"functionCall": {"name": fc.name, "args": fc.arguments}}]})
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": fc.name, "response": {"result": metrics}}}],
                })
            except Exception as e:
                tool_failures += 1
                logger.exception(
                    "solar_chat_trace_tool_error trace_id=%s step=%d tool=%s error=%s",
                    trace_id,
                    step_idx,
                    fc.name,
                    e,
                )
                trace_steps.append(
                    ThinkingStep(
                        step=f"Tool error {step_idx}",
                        detail=f"Tool '{fc.name}' failed, switching strategy.",
                        status="warning",
                    )
                )
                messages.append({"role": "model", "parts": [{"functionCall": {"name": fc.name, "args": fc.arguments}}]})
                messages.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": fc.name, "response": {"error": str(e)}}}],
                })
                if tool_failures >= 1:
                    logger.info("solar_chat_trace_tool_loop_break trace_id=%s reason=tool_failure", trace_id)
                    break

        if not all_metrics:
            self._tool_path_supported = False
            logger.warning(
                "solar_chat_tool_path_no_metrics trace_id=%s model=%s; forcing deterministic regex fallback",
                trace_id,
                last_model_used,
            )
            return self._handle_with_regex_fallback(
                request=request,
                started=started,
                history_messages=history_messages,
                trace_id=trace_id,
                skip_model_generation=True,
                warning_message=(
                    "Tool-calling produced no tool-backed metrics. "
                    "Switched to deterministic data-backed route."
                ),
                trace_steps_seed=trace_steps
                + [
                    ThinkingStep(
                        step="Tool-calling unavailable",
                        detail=(
                            "No tool-backed metrics were collected, so the service "
                            "switched to deterministic data route."
                        ),
                        status="warning",
                    )
                ],
            )

        answer = build_fallback_summary(
            topic,
            all_metrics,
            all_sources,
            user_message=request.message,
        )
        logger.info(
            "solar_chat_trace_tool_loop_complete trace_id=%s final_topic=%s collected_metric_keys=%s",
            trace_id,
            topic.value,
            sorted(list(all_metrics.keys())),
        )
        trace_steps.append(
            ThinkingStep(
                step="Summary generation",
                detail="Generated deterministic summary from collected tool outputs.",
                status="success",
            )
        )
        thinking_trace = self._build_thinking_trace(
            trace_id=trace_id,
            topic=topic,
            model_used=last_model_used,
            fallback_used=last_fallback_used,
            intent_confidence=TOOL_PATH_CONFIDENCE,
            sources=all_sources,
            warning_message=None,
            steps=trace_steps,
        )
        return self._finalize_response(
            request=request,
            answer=answer,
            topic=topic,
            metrics=all_metrics,
            sources=all_sources,
            model_used=last_model_used,
            fallback_used=last_fallback_used,
            started=started,
            trace_id=trace_id,
            intent_confidence=TOOL_PATH_CONFIDENCE,
            thinking_trace=thinking_trace,
        )

    def _handle_with_regex_fallback(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
        skip_model_generation: bool = False,
        allow_llm_planner: bool = False,
        warning_message: str | None = None,
        trace_steps_seed: list[ThinkingStep] | None = None,
    ) -> SolarChatResponse:
        trace_steps = list(trace_steps_seed or [])
        extreme_query = extract_extreme_metric_query(request.message)
        if extreme_query is not None:
            response_topic = topic_for_extreme_metric(extreme_query.metric_name)
            self._validate_role(topic=response_topic, role=request.role)
            query_date = extract_query_date(request.message)
            metrics, source_rows = self._fetch_extreme_metrics(
                extreme_query=extreme_query,
                query_date=query_date,
                message=request.message,
            )
            intent_confidence = REGEX_EXTREME_CONFIDENCE
            logger.info(
                "solar_chat_trace_intent trace_id=%s mode=extreme metric=%s query_type=%s timeframe=%s",
                trace_id,
                extreme_query.metric_name,
                extreme_query.query_type,
                extreme_query.timeframe,
            )
            trace_steps.append(
                ThinkingStep(
                    step="Intent parsing",
                    detail=(
                        f"Detected extreme-metric query ({extreme_query.metric_name}, "
                        f"{extreme_query.query_type}, timeframe={extreme_query.timeframe})."
                    ),
                    status="info",
                )
            )
        else:
            detection: IntentDetectionResult | None = None
            if allow_llm_planner and self._model_router is not None:
                detection = self._plan_topic_with_llm(
                    message=request.message,
                    history_messages=history_messages,
                )

            if detection is None:
                detection = self._intent_service.detect_intent(request.message)
                routing_mode = "intent"
            else:
                routing_mode = "llm_planner"

            response_topic = detection.topic
            intent_confidence = detection.confidence
            normalized_message = self._normalize_for_matching(request.message)
            if self._is_top_facility_comparison_query(normalized_message):
                response_topic = ChatTopic.ENERGY_PERFORMANCE
                intent_confidence = max(intent_confidence, 0.9)
            self._validate_role(topic=response_topic, role=request.role)
            metrics, source_rows = self._repository.fetch_topic_metrics(response_topic)
            logger.info(
                "solar_chat_trace_intent trace_id=%s mode=%s topic=%s confidence=%.2f",
                trace_id,
                routing_mode,
                response_topic.value,
                intent_confidence,
            )
            trace_steps.append(
                ThinkingStep(
                    step="Intent routing" if routing_mode == "intent" else "LLM planner routing",
                    detail=(
                        f"Routed request to topic '{response_topic.value}' via {routing_mode} "
                        f"with confidence {intent_confidence:.2f}."
                    ),
                    status="info",
                )
            )

        sources = [SourceMetadata(**row) for row in source_rows]
        logger.info(
            "solar_chat_trace_data_fetch trace_id=%s topic=%s metric_keys=%s source_count=%d",
            trace_id,
            response_topic.value,
            sorted(list(metrics.keys())),
            len(sources),
        )
        trace_steps.append(
            ThinkingStep(
                step="Data retrieval",
                detail=(
                    f"Fetched {len(metrics)} metric group(s) from {len(sources)} source(s) "
                    f"for topic '{response_topic.value}'."
                ),
                status="success",
            )
        )

        web_search_result = self._maybe_build_web_search_answer(
            message=request.message,
            topic=response_topic,
            skip_model_generation=skip_model_generation,
            metrics=metrics,
        )
        if web_search_result is not None:
            answer, citation_count, external_requested = web_search_result
            response_metrics = dict(metrics)
            response_metrics["web_search_used"] = True
            response_metrics["web_search_source_count"] = citation_count
            if external_requested:
                web_warning = (
                    "Used vetted external web references because you explicitly requested internet lookup."
                )
            else:
                web_warning = (
                    "Used vetted external web references because internal data did not directly cover this concept."
                )
            if warning_message:
                web_warning = f"{warning_message} {web_warning}"
            warning_message = web_warning
            trace_steps.append(
                ThinkingStep(
                    step="Web grounding",
                    detail=(
                        f"Retrieved and filtered {citation_count} external source(s) "
                        f"for {'an internet-requested' if external_requested else 'a definition-style'} query."
                    ),
                    status="warning",
                )
            )
            thinking_trace = self._build_thinking_trace(
                trace_id=trace_id,
                topic=response_topic,
                model_used="web-search-fallback",
                fallback_used=True,
                intent_confidence=intent_confidence,
                sources=sources,
                warning_message=warning_message,
                steps=trace_steps,
            )
            return self._finalize_response(
                request=request,
                answer=answer,
                topic=response_topic,
                metrics=response_metrics,
                sources=sources,
                model_used="web-search-fallback",
                fallback_used=True,
                started=started,
                trace_id=trace_id,
                intent_confidence=intent_confidence,
                warning_message=warning_message,
                thinking_trace=thinking_trace,
            )

        model_used = "deterministic-summary"
        fallback_used = skip_model_generation

        prompt = build_prompt(
            user_message=request.message,
            role=request.role,
            topic=response_topic,
            metrics=metrics,
            sources=sources,
            history=history_messages,
        )

        if skip_model_generation:
            answer = build_fallback_summary(
                response_topic,
                metrics,
                sources,
                user_message=request.message,
            )
            logger.info("solar_chat_trace_generation trace_id=%s mode=deterministic", trace_id)
            trace_steps.append(
                ThinkingStep(
                    step="Answer generation",
                    detail="Skipped model generation and returned deterministic data-backed summary.",
                    status="warning",
                )
            )
        elif self._model_router is not None:
            try:
                model_result = self._model_router.generate(prompt)
                answer = model_result.text
                fallback_used = model_result.fallback_used
                model_used = model_result.model_used
                logger.info(
                    "solar_chat_trace_generation trace_id=%s mode=llm model=%s fallback=%s",
                    trace_id,
                    model_used,
                    fallback_used,
                )
                trace_steps.append(
                    ThinkingStep(
                        step="Answer generation",
                        detail=f"Generated answer with model '{model_used}' (fallback={fallback_used}).",
                        status="success",
                    )
                )
            except ModelUnavailableError:
                answer = build_fallback_summary(
                    response_topic,
                    metrics,
                    sources,
                    user_message=request.message,
                )
                warning_message = "The AI model is temporarily unavailable. Returned a data-backed summary instead."
                fallback_used = True
                logger.warning("solar_chat_trace_generation trace_id=%s mode=llm_unavailable", trace_id)
                trace_steps.append(
                    ThinkingStep(
                        step="Answer generation",
                        detail="Model unavailable, switched to deterministic data-backed summary.",
                        status="warning",
                    )
                )
            except Exception as model_error:
                logger.exception(
                    "solar_chat_generation_failed trace_id=%s error_type=%s error=%s",
                    trace_id,
                    type(model_error).__name__,
                    model_error,
                )
                answer = build_fallback_summary(
                    response_topic,
                    metrics,
                    sources,
                    user_message=request.message,
                )
                warning_message = "The AI model request failed. Returned a data-backed summary instead."
                fallback_used = True
                trace_steps.append(
                    ThinkingStep(
                        step="Answer generation",
                        detail="Model request failed, switched to deterministic data-backed summary.",
                        status="warning",
                    )
                )
        else:
            answer = build_fallback_summary(
                response_topic,
                metrics,
                sources,
                user_message=request.message,
            )
            warning_message = "Gemini API key is not configured. Returned a data-backed summary."
            logger.warning("solar_chat_trace_generation trace_id=%s mode=no_model_router", trace_id)
            trace_steps.append(
                ThinkingStep(
                    step="Answer generation",
                    detail="Model router unavailable, returned deterministic data-backed summary.",
                    status="warning",
                )
            )

        thinking_trace = self._build_thinking_trace(
            trace_id=trace_id,
            topic=response_topic,
            model_used=model_used,
            fallback_used=fallback_used,
            intent_confidence=intent_confidence,
            sources=sources,
            warning_message=warning_message,
            steps=trace_steps,
        )

        return self._finalize_response(
            request=request,
            answer=answer,
            topic=response_topic,
            metrics=metrics,
            sources=sources,
            model_used=model_used,
            fallback_used=fallback_used,
            started=started,
            trace_id=trace_id,
            intent_confidence=intent_confidence,
            warning_message=warning_message,
            thinking_trace=thinking_trace,
        )

    @staticmethod
    def _is_station_daily_report_no_data(function_name: str, metrics: dict[str, Any]) -> bool:
        if function_name != "get_station_daily_report":
            return False
        station_count = metrics.get("station_count")
        try:
            return int(station_count or 0) == 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _build_station_report_no_data_answer(metrics: dict[str, Any]) -> str:
        report_date = str(metrics.get("report_date", "N/A"))
        available_min = metrics.get("available_date_min")
        available_max = metrics.get("available_date_max")

        if available_min and available_max:
            return (
                f"Không có dữ liệu cho ngày {report_date}. "
                f"Khoảng ngày dữ liệu hiện có là từ {available_min} đến {available_max}."
            )

        return f"Không có dữ liệu cho ngày {report_date}."

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

    def _build_thinking_trace(
        self,
        *,
        trace_id: str,
        topic: ChatTopic,
        model_used: str,
        fallback_used: bool,
        intent_confidence: float,
        sources: list[SourceMetadata],
        warning_message: str | None,
        steps: list[ThinkingStep],
    ) -> ThinkingTrace:
        route_mode = (
            "deterministic data-backed"
            if model_used == "deterministic-summary" or fallback_used
            else "tool-assisted model"
        )
        summary = (
            f"Routed to topic '{topic.value}' (confidence {intent_confidence:.2f}), "
            f"used {len(sources)} data source(s), and generated answer via {route_mode} path "
            f"with model '{model_used}'."
        )
        if warning_message:
            summary = f"{summary} Warning: {warning_message}"
        return ThinkingTrace(
            summary=summary,
            steps=steps,
            trace_id=trace_id,
        )

    @staticmethod
    def _short(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _validate_role(self, topic: ChatTopic, role: ChatRole) -> None:
        allowed_topics = ROLE_TOPIC_PERMISSIONS.get(role, set())
        if topic not in allowed_topics:
            raise PermissionError(
                f"Role '{role.value}' is not allowed to access topic '{topic.value}'."
            )

    def _fetch_extreme_metrics(
        self,
        extreme_query: ExtremeMetricQuery,
        query_date: Any,
        message: str,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if extreme_query.metric_name == "aqi":
            return self._repository.fetch_extreme_aqi(
                query_type=extreme_query.query_type,
                timeframe=extreme_query.timeframe,
                anchor_date=query_date,
                specific_hour=extreme_query.specific_hour,
            )
        if extreme_query.metric_name == "energy":
            return self._repository.fetch_extreme_energy(
                query_type=extreme_query.query_type,
                timeframe=extreme_query.timeframe,
                anchor_date=query_date,
                specific_hour=extreme_query.specific_hour,
            )
        weather_metric = resolve_weather_metric(message)
        return self._repository.fetch_extreme_weather(
            query_type=extreme_query.query_type,
            timeframe=extreme_query.timeframe,
            anchor_date=query_date,
            specific_hour=extreme_query.specific_hour,
            weather_metric=weather_metric["key"],
            weather_metric_label=weather_metric["label"],
            weather_unit=weather_metric["unit"],
        )

    def _maybe_build_web_search_answer(
        self,
        message: str,
        topic: ChatTopic,
        skip_model_generation: bool,
        metrics: dict[str, Any],
    ) -> tuple[str, int, bool] | None:
        if self._web_search_client is None or not self._web_search_client.enabled:
            return None

        normalized = self._normalize_for_matching(message)
        if not normalized:
            return None

        external_requested = self._is_external_search_request(normalized)
        definition_style = self._is_definition_style_query(normalized)

        if not definition_style and not external_requested:
            return None

        if self._is_data_request_query(normalized) and not external_requested:
            return None

        if not self._is_system_related_query(normalized) and not external_requested:
            return None

        if topic not in {
            ChatTopic.GENERAL,
            ChatTopic.ENERGY_PERFORMANCE,
            ChatTopic.ML_MODEL,
            ChatTopic.SYSTEM_OVERVIEW,
            ChatTopic.FACILITY_INFO,
        }:
            return None

        if (
            not external_requested
            and not skip_model_generation
            and topic is not ChatTopic.GENERAL
        ):
            return None

        station_name: str | None = None
        station_capacity_mw: float | None = None
        search_query = message
        if topic is ChatTopic.FACILITY_INFO:
            station_name, station_capacity_mw = self._resolve_facility_station_context(
                normalized_message=normalized,
                metrics=metrics,
            )
            if external_requested and station_name:
                search_query = f"{station_name} solar farm capacity location timezone"
            elif external_requested and not station_name:
                # Avoid broad/keyword-only web answers when the target station is unknown.
                return None

        search_results = self._web_search_client.search(search_query)
        if not search_results:
            return None

        focus_terms = self._extract_web_focus_terms(
            normalized,
            topic,
            station_name=station_name,
        )
        filtered_results = [
            row for row in search_results
            if self._is_web_result_relevant(row=row, focus_terms=focus_terms)
        ]
        if topic is ChatTopic.FACILITY_INFO and station_name:
            filtered_results = [
                row
                for row in filtered_results
                if self._is_facility_result_station_aligned(
                    row=row,
                    station_name=station_name,
                )
            ]
        if not filtered_results:
            return None

        filtered_results.sort(
            key=lambda row: (
                self._trusted_domain_score(row.url),
                float(row.score or 0.0),
            ),
            reverse=True,
        )
        selected = filtered_results[:3]
        answer = self._build_web_search_answer(
            message=message,
            selected_results=selected,
            topic=topic,
            external_requested=external_requested,
            station_name=station_name,
            station_capacity_mw=station_capacity_mw,
        )
        return answer, len(selected), external_requested

    @staticmethod
    def _normalize_for_matching(value: str) -> str:
        lowered = str(value or "").strip().lower()
        without_marks = normalize("NFD", lowered)
        return "".join(character for character in without_marks if ord(character) < 128)

    @staticmethod
    def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _is_definition_style_query(self, normalized_message: str) -> bool:
        return self._contains_any(normalized_message, WEBSEARCH_DEFINITION_MARKERS)

    def _is_external_search_request(self, normalized_message: str) -> bool:
        return self._contains_any(normalized_message, WEBSEARCH_EXTERNAL_REQUEST_MARKERS)

    def _is_data_request_query(self, normalized_message: str) -> bool:
        return self._contains_any(normalized_message, WEBSEARCH_DATA_REQUEST_MARKERS)

    def _is_system_related_query(self, normalized_message: str) -> bool:
        return self._contains_any(normalized_message, WEBSEARCH_SYSTEM_KEYWORDS)

    def _extract_web_focus_terms(
        self,
        normalized_message: str,
        topic: ChatTopic,
        station_name: str | None = None,
    ) -> tuple[str, ...]:
        terms: set[str] = {"solar", "pv", "photovoltaic", "energy"}

        if (
            "performance ratio" in normalized_message
            or "capacity factor" in normalized_message
            or "he so cong suat" in normalized_message
            or "ti le hieu suat" in normalized_message
        ):
            terms.update({"performance ratio", "capacity factor"})

        if topic is ChatTopic.ML_MODEL:
            terms.update({"forecast", "model", "solar forecasting"})
        if topic is ChatTopic.SYSTEM_OVERVIEW:
            terms.update({"solar power", "generation", "plant"})
        if topic is ChatTopic.FACILITY_INFO:
            terms.update({"solar farm", "capacity", "photovoltaic"})
            if not station_name:
                terms.update({"facility", "station", "timezone", "location"})

        if station_name:
            normalized_station = self._normalize_for_matching(station_name)
            if normalized_station:
                terms.add(normalized_station)
                for token in normalized_station.split():
                    if len(token) >= 4:
                        terms.add(token)

        return tuple(sorted(terms))

    def _is_web_result_relevant(
        self,
        row: "WebSearchResult",
        focus_terms: tuple[str, ...],
    ) -> bool:
        combined = self._normalize_for_matching(
            f"{row.title} {row.snippet} {row.url}"
        )
        if not combined:
            return False

        if not self._contains_any(combined, WEBSEARCH_SYSTEM_KEYWORDS):
            return False

        if focus_terms and not any(term in combined for term in focus_terms):
            return False

        return True

    def _is_facility_result_station_aligned(
        self,
        row: "WebSearchResult",
        station_name: str,
    ) -> bool:
        normalized_station = self._normalize_for_matching(station_name)
        if not normalized_station:
            return False

        combined = self._normalize_for_matching(
            f"{row.title} {row.snippet} {row.url}"
        )
        if not combined:
            return False

        station_tokens = [
            token for token in normalized_station.split()
            if len(token) >= 4
        ]
        if normalized_station in combined:
            station_match = True
        else:
            station_match = bool(station_tokens) and any(
                token in combined for token in station_tokens
            )
        if not station_match:
            return False

        context_markers = (
            "solar",
            "photovoltaic",
            "pv",
            "renewable",
            "energy",
            "power",
            "capacity",
            "commissioning",
            "planning",
            "project",
            "farm",
        )
        return any(marker in combined for marker in context_markers)

    @staticmethod
    def _trusted_domain_score(url: str) -> int:
        normalized_url = str(url or "").strip().lower()
        if any(domain in normalized_url for domain in WEBSEARCH_TRUSTED_DOMAIN_HINTS):
            return 1
        return 0

    def _build_web_search_answer(
        self,
        message: str,
        selected_results: list["WebSearchResult"],
        topic: ChatTopic,
        external_requested: bool,
        station_name: str | None,
        station_capacity_mw: float | None,
    ) -> str:
        normalized_message = self._normalize_for_matching(message)
        lines: list[str] = []

        if external_requested and topic is ChatTopic.FACILITY_INFO:
            lines.append("**Kết luận nội bộ**")
            if station_name:
                capacity_text = "N/A"
                if station_capacity_mw is not None:
                    capacity_text = f"{station_capacity_mw:.2f}".rstrip("0").rstrip(".")
                if self._is_bottom_capacity_request(normalized_message):
                    lines.append(
                        f"- Trạm có capacity nhỏ nhất hiện tại: **{station_name} ({capacity_text} MW)**."
                    )
                elif self._is_top_capacity_request(normalized_message):
                    lines.append(
                        f"- Trạm có capacity lớn nhất hiện tại: **{station_name} ({capacity_text} MW)**."
                    )
                else:
                    lines.append(
                        f"- Trạm mục tiêu: **{station_name} ({capacity_text} MW)**."
                    )
            else:
                lines.append(
                    "- Chưa xác định được chính xác 1 trạm từ dữ liệu nội bộ, nên mình tổng hợp nguồn internet liên quan."
                )

            lines.append("")
            lines.append("**Thông tin tổng hợp từ internet (đã lọc nhiễu)**")
            lines.extend(self._format_web_reference_rows(selected_results[:2]))
            lines.append("")
            lines.append("**Nguồn tham khảo**")
            lines.extend(self._format_web_reference_rows(selected_results, include_snippet=False))
            return "\n".join(lines)

        if (
            "performance ratio" in normalized_message
            or "capacity factor" in normalized_message
            or "he so cong suat" in normalized_message
            or "ti le hieu suat" in normalized_message
        ):
            lines.extend(
                [
                    "Theo nguồn kỹ thuật bên ngoài, Performance Ratio (PR) là tỷ lệ giữa sản lượng AC thực tế và sản lượng kỳ vọng sau khi quy đổi theo bức xạ mặt trời.",
                    "PR phản ánh mức tổn thất vận hành của hệ PV (nhiệt độ, inverter, cáp, bụi bẩn, suy hao thiết bị).",
                    "Trong hệ thống của bạn, hiện đang dùng capacity_factor_pct như chỉ số proxy để so sánh tương đối giữa các trạm vì chưa có cột PR chuẩn trực tiếp cho mọi bản ghi.",
                ]
            )
        else:
            lines.append(
                "Mình chưa có dữ liệu nội bộ trực tiếp cho định nghĩa này, nên đã đối chiếu thêm nguồn kỹ thuật bên ngoài."
            )
            lines.extend(self._format_web_reference_rows(selected_results[:2]))

        lines.append("")
        lines.append("**Nguồn tham khảo**")
        lines.extend(self._format_web_reference_rows(selected_results, include_snippet=False))

        return "\n".join(lines)

    def _format_web_reference_rows(
        self,
        selected_results: list["WebSearchResult"],
        *,
        include_snippet: bool = True,
    ) -> list[str]:
        rows: list[str] = []
        for index, row in enumerate(selected_results, start=1):
            title = row.title.strip() or f"Source {index}"
            url = row.url.strip()
            rows.append(f"{index}. [{title}]({url})")
            if include_snippet:
                snippet = self._clean_web_snippet(row.snippet)
                if not snippet:
                    snippet = self._clean_web_snippet(row.title)
                if snippet:
                    if self._is_noisy_web_snippet(snippet):
                        rows.append(
                            "- Tóm tắt: Nguồn này cung cấp hồ sơ dự án, vị trí vận hành và quy mô công suất của trạm."
                        )
                    else:
                        rows.append(f"- Tóm tắt: {self._short(snippet, 220)}")
        return rows

    @staticmethod
    def _clean_web_snippet(value: str | None) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\|+", " ", text)
        text = re.sub(r"#{1,6}\s*", " ", text)
        text = re.sub(r"-{2,}", " ", text)
        text = re.sub(r"\bwikipedia\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthe free encyclopedia\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcontents\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" .:-")

    @staticmethod
    def _is_noisy_web_snippet(text: str) -> bool:
        lowered = str(text or "").lower()
        noise_terms = ("country", "location", "coordinates", "map", "contents")
        noise_hits = sum(1 for term in noise_terms if term in lowered)
        if noise_hits >= 2:
            return True

        token_count = len(lowered.split())
        has_sentence_break = any(mark in lowered for mark in (".", "!", "?", ";", ":"))
        if token_count >= 28 and not has_sentence_break:
            return True

        return False

    @staticmethod
    def _is_top_capacity_request(normalized_message: str) -> bool:
        markers = (
            "lon nhat",
            "largest",
            "biggest",
            "max capacity",
            "capacity lon nhat",
            "cong suat lon nhat",
            "highest capacity",
        )
        return any(marker in normalized_message for marker in markers)

    @staticmethod
    def _is_bottom_capacity_request(normalized_message: str) -> bool:
        markers = (
            "nho nhat",
            "thap nhat",
            "smallest",
            "lowest",
            "minimum",
            "min capacity",
            "capacity nho nhat",
            "capacty nho nhat",
            "cong suat nho nhat",
            "lowest capacity",
        )
        return any(marker in normalized_message for marker in markers)

    @staticmethod
    def _extract_facility_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
        rows = metrics.get("facilities", []) if isinstance(metrics, dict) else []
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _extract_capacity_mw(row: dict[str, Any]) -> float:
        raw = row.get("total_capacity_mw")
        if raw is None:
            raw = row.get("capacity_mw")
        if raw is None:
            raw = row.get("capacity")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _resolve_facility_station_context(
        self,
        normalized_message: str,
        metrics: dict[str, Any],
    ) -> tuple[str | None, float | None]:
        facilities = self._extract_facility_rows(metrics)
        if not facilities:
            return None, None

        for row in facilities:
            name = str(row.get("facility_name") or row.get("facility") or "").strip()
            if not name:
                continue
            normalized_name = self._normalize_for_matching(name)
            if normalized_name and normalized_name in normalized_message:
                return name, self._extract_capacity_mw(row)

        if self._is_bottom_capacity_request(normalized_message):
            bottom_row = min(
                facilities,
                key=self._extract_capacity_mw,
            )
            bottom_name = str(bottom_row.get("facility_name") or bottom_row.get("facility") or "").strip()
            if bottom_name:
                return bottom_name, self._extract_capacity_mw(bottom_row)

        if self._is_top_capacity_request(normalized_message):
            top_row = max(
                facilities,
                key=self._extract_capacity_mw,
            )
            top_name = str(top_row.get("facility_name") or top_row.get("facility") or "").strip()
            if top_name:
                return top_name, self._extract_capacity_mw(top_row)

        return None, None

    def _detect_topic_from_text(self, message: str) -> ChatTopic:
        normalized = self._normalize_for_matching(message)
        if self._is_top_facility_comparison_query(normalized):
            return ChatTopic.ENERGY_PERFORMANCE
        try:
            return self._intent_service.detect_intent(message).topic
        except Exception:
            logger.debug("Intent detection failed for topic text mapping, defaulting to GENERAL.")
            return ChatTopic.GENERAL

    @staticmethod
    def _is_top_facility_comparison_query(normalized_message: str) -> bool:
        compare_markers = ("so sanh", "compare", "comparison", "versus", " vs ")
        facility_markers = ("facility", "facilities", "tram", "co so", "nha may")
        ranking_markers = (
            "top",
            "top 2",
            "largest",
            "highest",
            "lon nhat",
            "2 facilities",
            "2 facility",
            "hai tram",
            "hai co so",
        )
        has_compare = any(marker in normalized_message for marker in compare_markers)
        has_facility = any(marker in normalized_message for marker in facility_markers)
        has_ranking = any(marker in normalized_message for marker in ranking_markers)
        return has_compare and has_facility and has_ranking

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        stripped = str(text or "").strip()
        if stripped:
            candidates.append(stripped)

        if "```" in stripped:
            non_fence_lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
            joined = "\n".join(non_fence_lines).strip()
            if joined:
                candidates.append(joined)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _plan_topic_with_llm(
        self,
        message: str,
        history_messages: list[ChatMessage],
    ) -> IntentDetectionResult | None:
        if self._model_router is None:
            return None

        topic_values = [topic.value for topic in ChatTopic]
        recent = history_messages[-LLM_PLANNER_HISTORY_LIMIT:]
        history_lines = [
            f"- {msg.sender}: {self._short(msg.content, 180)}"
            for msg in recent
        ]
        history_block = "\n".join(history_lines) if history_lines else "- (no history)"

        planner_prompt = (
            "You are a routing planner for Solar AI Chat.\n"
            "Task: choose exactly one topic label for the current user query.\n"
            f"Allowed topics: {', '.join(topic_values)}\n"
            "Return ONLY valid JSON in one line with this schema:\n"
            '{"topic":"<allowed_topic>","confidence":<number_0_to_1>}\n'
            "Do not include markdown or extra text.\n"
            f"Recent conversation:\n{history_block}\n"
            f"Current user message: {message}"
        )

        try:
            planner_result = self._model_router.generate(planner_prompt)
        except Exception as planner_error:
            logger.warning("solar_chat_llm_planner_failed error=%s", planner_error)
            return None

        payload = self._parse_json_object(planner_result.text)
        if not payload:
            logger.info("solar_chat_llm_planner_invalid_json text=%s", self._short(planner_result.text, 240))
            return None

        topic_raw = str(payload.get("topic") or "").strip().lower()
        if topic_raw not in topic_values:
            logger.info("solar_chat_llm_planner_unknown_topic topic=%s", topic_raw)
            return None

        raw_confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if confidence < LLM_PLANNER_MIN_CONFIDENCE:
            logger.info(
                "solar_chat_llm_planner_low_confidence topic=%s confidence=%.2f threshold=%.2f",
                topic_raw,
                confidence,
                LLM_PLANNER_MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "solar_chat_llm_planner_selected topic=%s confidence=%.2f model=%s fallback=%s",
            topic_raw,
            confidence,
            planner_result.model_used,
            planner_result.fallback_used,
        )
        return IntentDetectionResult(
            topic=ChatTopic(topic_raw),
            confidence=round(confidence, 2),
        )

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
            session_id=session_id, sender="assistant", content=answer,
            topic=topic, sources=sources,
        )
