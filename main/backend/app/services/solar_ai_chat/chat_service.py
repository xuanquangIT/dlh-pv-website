"""Solar AI Chat service — orchestration layer.

Coordinates intent detection, RBAC, data retrieval, LLM generation,
and response finalization. NLP parsing is in nlp_parser.py and
prompt/summary building is in prompt_builder.py.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.repositories.solar_ai_chat.history_repository import ChatHistoryRepository
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat import (
    ChatMessage,
    ChatRole,
    ChatTopic,
    SolarChatRequest,
    SolarChatResponse,
    SourceMetadata,
)
from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS, TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.gemini_client import GeminiModelRouter, ModelUnavailableError
from app.services.solar_ai_chat.intent_service import VietnameseIntentService
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

logger = logging.getLogger(__name__)

# Intent confidence levels for response metadata
TOOL_PATH_CONFIDENCE = 0.95
TEXT_ONLY_CONFIDENCE = 0.85
REGEX_EXTREME_CONFIDENCE = 0.92


class SolarAIChatService:
    """Orchestrates intent routing, RBAC, data retrieval, and LLM response."""

    def __init__(
        self,
        repository: SolarChatRepository,
        intent_service: VietnameseIntentService,
        model_router: GeminiModelRouter | None,
        history_repository: ChatHistoryRepository | None = None,
        vector_repo: "VectorRepository | None" = None,
        embedding_client: "GeminiEmbeddingClient | None" = None,
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

    def handle_query(self, request: SolarChatRequest) -> SolarChatResponse:
        started = time.perf_counter()
        history_messages = self._load_history(request.session_id)

        if self._model_router is not None:
            try:
                return self._handle_with_tools(request, started, history_messages)
            except ModelUnavailableError:
                logger.warning("solar_chat_tool_path_unavailable, falling back to regex intent")
            except Exception as tool_err:
                logger.warning("solar_chat_tool_path_failed (%s), falling back to regex intent", tool_err)

        return self._handle_with_regex_fallback(request, started, history_messages)

    def _handle_with_tools(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
    ) -> SolarChatResponse:
        messages = build_tool_messages(
            request_message=request.message,
            role_value=request.role.value,
            history=history_messages,
        )
        tool_result = self._model_router.generate_with_tools(messages, TOOL_DECLARATIONS)

        if tool_result.text is not None:
            topic = self._detect_topic_from_text(request.message)
            return self._finalize_response(
                request=request,
                answer=tool_result.text,
                topic=topic,
                metrics={},
                sources=[],
                model_used=tool_result.model_used,
                fallback_used=tool_result.fallback_used,
                started=started,
                intent_confidence=TEXT_ONLY_CONFIDENCE,
            )

        fc = tool_result.function_call
        metrics, source_rows = self._tool_executor.execute(fc.name, fc.arguments, request.role)
        sources = [SourceMetadata(**row) for row in source_rows]
        topic = ChatTopic(TOOL_NAME_TO_TOPIC.get(fc.name, "general"))

        messages.append({"role": "model", "parts": [{"functionCall": {"name": fc.name, "args": fc.arguments}}]})
        messages.append({
            "role": "user",
            "parts": [{"functionResponse": {"name": fc.name, "response": {"result": metrics}}}],
        })

        final_result = self._model_router.send_tool_result(messages, tool_result.model_used)
        answer = final_result.text or build_fallback_summary(topic, metrics, sources)

        return self._finalize_response(
            request=request,
            answer=answer,
            topic=topic,
            metrics=metrics,
            sources=sources,
            model_used=tool_result.model_used,
            fallback_used=tool_result.fallback_used,
            started=started,
            intent_confidence=TOOL_PATH_CONFIDENCE,
        )

    def _handle_with_regex_fallback(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
    ) -> SolarChatResponse:
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
        else:
            detection = self._intent_service.detect_intent(request.message)
            response_topic = detection.topic
            intent_confidence = detection.confidence
            self._validate_role(topic=response_topic, role=request.role)
            metrics, source_rows = self._repository.fetch_topic_metrics(response_topic)

        sources = [SourceMetadata(**row) for row in source_rows]

        warning_message: str | None = None
        model_used = "deterministic-summary"
        fallback_used = False

        prompt = build_prompt(
            user_message=request.message,
            role=request.role,
            topic=response_topic,
            metrics=metrics,
            sources=sources,
            history=history_messages,
        )

        if self._model_router is not None:
            try:
                model_result = self._model_router.generate(prompt)
                answer = model_result.text
                fallback_used = model_result.fallback_used
                model_used = model_result.model_used
            except ModelUnavailableError:
                answer = build_fallback_summary(response_topic, metrics, sources)
                warning_message = "The AI model is temporarily unavailable. Returned a data-backed summary instead."
                fallback_used = True
        else:
            answer = build_fallback_summary(response_topic, metrics, sources)
            warning_message = "Gemini API key is not configured. Returned a data-backed summary."

        return self._finalize_response(
            request=request,
            answer=answer,
            topic=response_topic,
            metrics=metrics,
            sources=sources,
            model_used=model_used,
            fallback_used=fallback_used,
            started=started,
            intent_confidence=intent_confidence,
            warning_message=warning_message,
        )

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
        intent_confidence: float,
        warning_message: str | None = None,
    ) -> SolarChatResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._persist_exchange(
            session_id=request.session_id,
            user_message=request.message,
            answer=answer,
            topic=topic,
            sources=sources,
        )
        logger.info(
            "solar_chat_request_completed",
            extra={
                "topic": topic.value,
                "role": request.role.value,
                "model_used": model_used,
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
            },
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
        )

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

    def _detect_topic_from_text(self, message: str) -> ChatTopic:
        try:
            return self._intent_service.detect_intent(message).topic
        except Exception:
            logger.debug("Intent detection failed for topic text mapping, defaulting to GENERAL.")
            return ChatTopic.GENERAL

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
