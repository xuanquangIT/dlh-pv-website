"""Solar AI Chat service — orchestration layer.

Phase 4 cleanup: the v1 14-tool agentic loop, intent overrides, prompt-rule
context, deep planner, and answer verifier have been deleted. The only
runtime path is the engine (`services/solar_ai_chat/`) which composes
6 generic primitives over a YAML semantic layer.

This file owns:
  - `SolarAIChatService` — public class consumed by `api/solar_ai_chat/routes.py`
  - persistence + history plumbing (Postgres chat_messages)
  - prompt-injection refusal short-circuit
  - SSE streaming bridge (handle_query_stream)

It does NOT own (any more):
  - Legacy ToolExecutor / DeepPlanner / AnswerVerifier — deleted
  - intent_service / nlp_parser / query_rewriter / chart_service /
    prompt_builder — deleted (engine has its own off-topic + chart-intent
    + language detection)
"""
from __future__ import annotations

import json as _json
import logging
import time
import uuid
from typing import Any
from unicodedata import normalize

from app.repositories.solar_ai_chat.base_repository import (
    DatabricksDataUnavailableError,
)
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
)
from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS
from app.services.solar_ai_chat.cancellation import (
    EngineCancelled,
    register as _register_cancel,
    unregister as _unregister_cancel,
)
from app.services.solar_ai_chat.llm_client import LLMModelRouter

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


_PROMPT_INJECTION_TOKEN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ignore", "previous"),
    ("ignore", "instructions"),
    ("bo qua", "huong dan"),
    ("bo qua", "chi dan"),
    ("reveal", "system"),
    ("system", "prompt"),
    ("developer", "prompt"),
    ("authentication", "token"),
)


def _strip_diacritics_lower(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics for tolerant matching."""
    lowered = (text or "").lower()
    decomposed = normalize("NFD", lowered)
    return "".join(c for c in decomposed if ord(c) < 128)


def _is_prompt_injection_request(message: str) -> bool:
    normalized = _strip_diacritics_lower(message)
    for group in _PROMPT_INJECTION_TOKEN_GROUPS:
        if all(token in normalized for token in group):
            return True
    return False


def _scope_refusal(language: str) -> str:
    if language == "vi":
        return (
            "Mình chỉ trả lời các câu hỏi liên quan đến hệ thống PV "
            "Lakehouse — sản lượng điện mặt trời, hiệu suất trạm, thời "
            "tiết, AQI, dự báo, mô hình ML, và pipeline. Bạn thử hỏi "
            "lại trong phạm vi đó nhé."
        )
    return (
        "I only answer questions about the PV Lakehouse — solar energy "
        "production, facility performance, weather, AQI, forecasts, ML "
        "models, and pipeline status."
    )


class SolarAIChatService:
    """Solar AI Chat orchestrator.

    Constructor accepts the full pre-Phase-4 kwarg surface so existing
    factory code in `api/solar_ai_chat/routes.py` and tests keep working;
    v1-only kwargs are ignored.
    """

    def __init__(
        self,
        repository: SolarChatRepository,
        model_router: LLMModelRouter | None = None,
        history_repository: Any | None = None,
        tool_usage_logger: Any | None = None,
        # Accept-and-ignore for any external caller still passing the
        # pre-Phase-4 kwarg surface (intent_service, vector_repo,
        # embedding_client, web_search_client, planner_enabled, etc).
        **legacy_kwargs: Any,
    ) -> None:
        self._repository = repository
        self._model_router = model_router
        self._history_repository = history_repository
        self._tool_usage_logger = tool_usage_logger
        _ = legacy_kwargs  # silently swallow ABI-compat args

    # ------------------------------------------------------------------
    # Public — non-streaming
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

        if _is_prompt_injection_request(request.message):
            return self._build_refusal_response(
                request, started, trace_id,
                warning="Prompt-injection request refused.",
                model_used="scope-guard",
            )

        history_messages = self._load_history(request.session_id)
        logger.info(
            "solar_chat_trace_history trace_id=%s history_count=%d",
            trace_id, len(history_messages),
        )

        cancel_event = _register_cancel(trace_id)
        try:
            return self._handle_with_engine(
                request, started, history_messages, trace_id,
                cancel_event=cancel_event,
            )
        except EngineCancelled:
            from app.services.solar_ai_chat.engine import _detect_language
            language = _detect_language(request.message)
            return self._build_refusal_response(
                request, started, trace_id,
                warning="Stopped by user.",
                model_used="engine-cancelled",
                answer=(
                    "Đã dừng theo yêu cầu."
                    if language == "vi"
                    else "Stopped by user."
                ),
                fallback_used=False,
            )
        except DatabricksDataUnavailableError:
            raise
        except Exception as err:  # noqa: BLE001
            logger.exception(
                "solar_chat_engine_failed trace_id=%s error=%s",
                trace_id, err,
            )
            from app.services.solar_ai_chat.engine import _detect_language
            language = _detect_language(request.message)
            return self._build_refusal_response(
                request, started, trace_id,
                warning=f"Engine failed: {err}",
                model_used="engine-error",
                answer=(
                    "Xin lỗi, hiện chưa thể trả lời. Vui lòng thử lại sau."
                    if language == "vi"
                    else "Sorry, I couldn't answer that. Please try again."
                ),
                fallback_used=True,
                steps=[ThinkingStep(step="engine failure", detail=str(err), status="warning")],
            )
        finally:
            _unregister_cancel(trace_id)

    # ------------------------------------------------------------------
    # Public — SSE streaming
    # ------------------------------------------------------------------

    def handle_query_stream(self, request: SolarChatRequest):
        """Yield SSE-ready ``data: <json>\\n\\n`` strings."""
        from app.schemas.solar_ai_chat.stream import (
            CancelledEvent,
            DoneEvent,
            ErrorEvent,
            StartEvent,
            StatusUpdateEvent,
        )

        def _sse(event_obj) -> str:
            return "data: " + _json.dumps(event_obj.model_dump()) + "\n\n"

        started = time.perf_counter()
        trace_id = uuid.uuid4().hex[:8]
        cancel_event = _register_cancel(trace_id)

        try:
            yield _sse(StartEvent(trace_id=trace_id))
            yield _sse(StatusUpdateEvent(text="Analyzing your request…"))

            if self._model_router is None:
                yield _sse(ErrorEvent(message="No LLM configured.", code="no_llm"))
                return

            from app.services.solar_ai_chat.engine import _detect_language
            language = _detect_language(request.message)

            if _is_prompt_injection_request(request.message):
                latency_ms = int((time.perf_counter() - started) * 1000)
                yield _sse(DoneEvent(
                    answer=_scope_refusal(language),
                    topic=ChatTopic.GENERAL.value,
                    role=request.role.value if request.role else "admin",
                    sources=[],
                    key_metrics={},
                    model_used="scope-guard",
                    fallback_used=True,
                    latency_ms=latency_ms,
                    intent_confidence=0.0,
                    warning_message="Prompt-injection request refused.",
                    ui_features=resolve_ui_features(request.role),
                    trace_id=trace_id
                ))
                return

            history_messages = self._load_history(request.session_id)

            try:
                yield from self._stream_with_engine(
                    request, started, history_messages, trace_id, language,
                    cancel_event=cancel_event,
                )
                return
            except EngineCancelled:
                logger.info("solar_chat_stream_cancelled trace_id=%s", trace_id)
                yield _sse(CancelledEvent(trace_id=trace_id))
                return
            except DatabricksDataUnavailableError:
                raise
            except Exception as engine_err:  # noqa: BLE001
                logger.exception(
                    "solar_chat_stream_failed trace_id=%s error=%s",
                    trace_id, engine_err,
                )
                yield _sse(ErrorEvent(
                    message=f"Engine failed: {engine_err}",
                    code="engine_error",
                ))
                return
        except DatabricksDataUnavailableError:
            raise
        except Exception as err:  # noqa: BLE001
            logger.exception(
                "solar_chat_stream_failed trace_id=%s error=%s",
                trace_id, err,
            )
            yield _sse(ErrorEvent(
                message=f"Streaming failed: {err}",
                code="stream_error",
            ))
        finally:
            _unregister_cancel(trace_id)

    # ------------------------------------------------------------------
    # Engine bridge — non-streaming
    # ------------------------------------------------------------------

    def _handle_with_engine(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
        cancel_event: Any | None = None,
    ) -> SolarChatResponse:
        from app.core.settings import SolarChatSettings
        from app.services.solar_ai_chat.dispatcher import Dispatcher
        from app.services.solar_ai_chat.engine import ChatEngine, _detect_language

        if self._model_router is None:
            raise RuntimeError("ChatEngine requires an LLMModelRouter")

        settings = SolarChatSettings()
        role_id = (request.role.value if request.role else "admin").lower()
        dispatcher = Dispatcher(settings, role_id=role_id)
        engine = ChatEngine(self._model_router, dispatcher)
        language = _detect_language(request.message)
        force_chart = bool(
            getattr(request, "tool_hints", None)
            and "visualize" in (request.tool_hints or [])
        )

        result = engine.run(
            user_message=request.message,
            history=history_messages,
            language=language,
            force_chart=force_chart,
            cancel_event=cancel_event,
        )

        # Fast-paths inside the engine (e.g. `_answer_conceptual`) make a
        # single blocking LLM call without checking cancel_event. If the user
        # hit Stop while that call was in flight, drop the result on the
        # floor so it isn't persisted or returned to the now-cancelled UI.
        if cancel_event is not None and cancel_event.is_set():
            raise EngineCancelled("user requested stop after engine.run returned")

        chart_payload, data_table_payload, kpi_payload, viz_snapshot = (
            self._build_viz_payloads(result)
        )
        thinking_trace = self._build_thinking_trace(result, trace_id)

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
                "engine_persist_failed trace_id=%s err=%s",
                trace_id, persist_err,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "solar_chat_done trace_id=%s steps=%d model=%s latency_ms=%d chart=%s",
            trace_id, len(result.trace_steps), result.model_used, latency_ms,
            chart_payload.format if chart_payload else "none",
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
            kpi_cards=kpi_payload
        )

    # ------------------------------------------------------------------
    # Engine bridge — streaming
    # ------------------------------------------------------------------

    def _stream_with_engine(
        self,
        request: SolarChatRequest,
        started: float,
        history_messages: list[ChatMessage],
        trace_id: str,
        language: str,
        cancel_event: Any | None = None,
    ):
        """Run the engine in a worker thread and stream progress events."""
        import queue as _queue
        import threading as _threading

        from app.core.settings import SolarChatSettings
        from app.schemas.solar_ai_chat.stream import (
            DoneEvent,
            StatusUpdateEvent,
            ThinkingStepEvent,
            ToolResultEvent,
            tool_label,
        )
        from app.services.solar_ai_chat.dispatcher import Dispatcher
        from app.services.solar_ai_chat.engine import ChatEngine, _detect_language

        def _sse(event_obj) -> str:
            return "data: " + _json.dumps(event_obj.model_dump()) + "\n\n"

        if self._model_router is None:
            raise RuntimeError("ChatEngine requires an LLMModelRouter")

        yield _sse(StatusUpdateEvent(text="Running engine…"))

        settings = SolarChatSettings()
        role_id = (request.role.value if request.role else "admin").lower()
        dispatcher = Dispatcher(settings, role_id=role_id)
        engine = ChatEngine(self._model_router, dispatcher)
        language = _detect_language(request.message)
        force_chart = bool(
            getattr(request, "tool_hints", None)
            and "visualize" in (request.tool_hints or [])
        )

        events_q: _queue.Queue = _queue.Queue()
        SENTINEL = object()
        result_holder: dict[str, Any] = {}

        def _on_progress(payload: dict[str, Any]) -> None:
            events_q.put(payload)

        def _runner() -> None:
            try:
                r = engine.run(
                    user_message=request.message,
                    history=history_messages,
                    language=language or "en",
                    force_chart=force_chart,
                    progress_callback=_on_progress,
                    cancel_event=cancel_event,
                )
                result_holder["result"] = r
            except Exception as run_err:  # noqa: BLE001
                result_holder["error"] = run_err
            finally:
                events_q.put(SENTINEL)

        thread = _threading.Thread(target=_runner, daemon=True)
        thread.start()

        global_step = -1
        step_index_for_call: dict[tuple[int, str], int] = {}
        while True:
            try:
                ev = events_q.get(timeout=0.5)
            except _queue.Empty:
                if cancel_event is not None and cancel_event.is_set():
                    raise EngineCancelled("user requested stop mid-stream")
                continue
            if ev is SENTINEL:
                break
            if not isinstance(ev, dict):
                continue
            kind = ev.get("event")
            primitive = str(ev.get("primitive") or "")
            step_num = int(ev.get("step", 0))
            key = (step_num, primitive)
            if kind == "tool_start":
                global_step += 1
                step_index_for_call[key] = global_step
                yield _sse(ThinkingStepEvent(
                    step=global_step, tool_name=primitive,
                    label=tool_label(primitive), trace_id=trace_id,
                ))
            elif kind == "tool_end":
                idx = step_index_for_call.get(key, global_step)
                yield _sse(ToolResultEvent(
                    step=idx, tool_name=primitive,
                    status="ok" if ev.get("ok") else "error",
                    duration_ms=int(ev.get("duration_ms", 0)),
                    trace_id=trace_id,
                ))

        thread.join(timeout=1.0)
        if "error" in result_holder:
            raise result_holder["error"]
        # If the user hit Stop while a blocking fast-path LLM call was in
        # flight, the runner thread completes normally with a `result`. Drop
        # it before persistence so the cancelled exchange is not stored.
        if cancel_event is not None and cancel_event.is_set():
            raise EngineCancelled("user requested stop after engine returned")
        result = result_holder.get("result")
        if result is None:
            raise RuntimeError("ChatEngine returned no result")

        chart_payload, data_table_payload, kpi_payload, viz_snapshot = (
            self._build_viz_payloads(result)
        )
        thinking_trace = self._build_thinking_trace(result, trace_id)

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
                "stream_persist_failed trace_id=%s err=%s",
                trace_id, persist_err,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "solar_chat_stream_done trace_id=%s steps=%d model=%s latency_ms=%d chart=%s",
            trace_id, len(result.trace_steps), result.model_used, latency_ms,
            chart_payload.format if chart_payload else "none",
        )

        yield _sse(DoneEvent(
            answer=result.answer,
            topic=ChatTopic.GENERAL.value,
            role=request.role.value if request.role else "admin",
            sources=[s if isinstance(s, dict) else s.model_dump() for s in result.sources],
            key_metrics=result.key_metrics or {},
            model_used=result.model_used,
            fallback_used=result.fallback_used,
            latency_ms=latency_ms,
            intent_confidence=TOOL_PATH_CONFIDENCE,
            warning_message=result.error,
            thinking_trace=thinking_trace.model_dump(),
            ui_features=resolve_ui_features(request.role),
            data_table=data_table_payload.model_dump() if data_table_payload else None,
            chart=chart_payload.model_dump() if chart_payload else None,
            kpi_cards=kpi_payload.model_dump() if kpi_payload else None,
            trace_id=trace_id
        ))

    # ------------------------------------------------------------------
    # Shared viz / trace builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_viz_payloads(result):
        """Convert engine result.chart / data_table / key_metrics to API payloads."""
        from app.schemas.solar_ai_chat.visualization import (
            ChartPayload,
            DataTableColumn,
            DataTablePayload,
            KpiCard,
            KpiCardsPayload,
        )

        chart_payload: ChartPayload | None = None
        if result.chart and result.chart.get("format") == "leaflet-map":
            chart_payload = ChartPayload(
                chart_type="map",
                title=str(result.chart.get("title") or ""),
                format="leaflet-map",
                points=result.chart.get("points") or [],
                size_field=result.chart.get("size_field"),
                label_field=result.chart.get("label_field"),
                row_count=result.chart.get("row_count"),
            )
        elif result.chart and result.chart.get("format") == "vega-lite":
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
        return chart_payload, data_table_payload, kpi_payload, viz_snapshot

    @staticmethod
    def _build_thinking_trace(result, trace_id: str) -> ThinkingTrace:
        return ThinkingTrace(
            summary=f"Engine ran {len(result.trace_steps)} primitive call(s).",
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

    # ------------------------------------------------------------------
    # Refusal builder for prompt-injection / engine-failure cases
    # ------------------------------------------------------------------

    def _build_refusal_response(
        self,
        request: SolarChatRequest,
        started: float,
        trace_id: str,
        *,
        warning: str,
        model_used: str,
        answer: str | None = None,
        fallback_used: bool = True,
        steps: list[ThinkingStep] | None = None,
    ) -> SolarChatResponse:
        from app.services.solar_ai_chat.engine import _detect_language
        language = _detect_language(request.message)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return SolarChatResponse(
            answer=answer or _scope_refusal(language),
            topic=ChatTopic.GENERAL,
            role=request.role,
            key_metrics={},
            sources=[],
            model_used=model_used,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            intent_confidence=0.0,
            warning_message=warning,
            thinking_trace=ThinkingTrace(
                summary=warning,
                steps=steps or [],
                trace_id=trace_id,
            ),
            ui_features=resolve_ui_features(request.role)
        )

    # ------------------------------------------------------------------
    # Tool-mode declaration filter (used by routes when caller pins a
    # tool subset; kept since /sessions endpoints reference TOOL_DECLARATIONS).
    # ------------------------------------------------------------------

    @staticmethod
    def _select_tool_declarations(request: SolarChatRequest) -> list:
        mode = getattr(request, "tool_mode", "auto") or "auto"
        if mode != "selected":
            return list(TOOL_DECLARATIONS)
        allowed = set(request.allowed_tools or [])
        if not allowed:
            return list(TOOL_DECLARATIONS)

        def _name(decl: Any) -> str:
            if isinstance(decl, dict):
                return str(decl.get("name") or decl.get("function", {}).get("name", ""))
            return str(getattr(decl, "name", ""))

        filtered = [d for d in TOOL_DECLARATIONS if _name(d) in allowed]
        return filtered or list(TOOL_DECLARATIONS)

    # ------------------------------------------------------------------
    # Persistence + history
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
        self._history_repository.add_message(
            session_id=session_id, sender="user", content=user_message,
        )
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
    # Misc helpers (referenced by routes / tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _short(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _validate_role(self, topic: ChatTopic, role: ChatRole) -> None:
        from app.services.solar_ai_chat.permissions import ROLE_TOPIC_PERMISSIONS
        allowed_topics = ROLE_TOPIC_PERMISSIONS.get(role, set())
        if topic not in allowed_topics:
            raise PermissionError(
                f"Role '{role.value}' is not allowed to access topic '{topic.value}'."
            )
