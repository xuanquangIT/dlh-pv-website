"""Solar Chat v2 — agentic engine.

Replaces the v1 ToolExecutor + 16-rule prompt + chart_service heuristics with
a thin loop:

    LLM(messages, V2_TOOL_SCHEMAS)
      -> tool_call -> V2Dispatcher.execute(...) -> result
      -> append result -> repeat
      -> final text answer

Outputs a normalized dict the v1 chat_service can adapt into SolarChatResponse
(answer + key_metrics + sources + chart + trace), so v2 plugs in without
rewriting persistence, SSE, or auth.

Used only when settings.engine_version == "v2". v1 path is untouched.

Design doc: implementations/solar_chat_architecture_redesign_2026-04-26.md
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.schemas.solar_ai_chat import ChatMessage, SourceMetadata
from app.services.solar_ai_chat.llm_client import (
    LLMModelRouter,
    LLMToolResult,
    ToolCallNotSupportedError,
)
from app.services.solar_ai_chat.v2.dispatcher import V2Dispatcher, V2DispatchResult
from app.services.solar_ai_chat.v2.tool_schemas import (
    V2_SYSTEM_PROMPT,
    V2_TOOL_SCHEMAS,
)

logger = logging.getLogger(__name__)

MAX_LOOP_STEPS = 8           # hard cap on tool turns (v1 default = 6)
MAX_TOOL_RESULT_CHARS = 12_000  # truncate huge SQL row payloads going back to LLM


# -----------------------------------------------------------------------------
# Result envelope
# -----------------------------------------------------------------------------

@dataclass
class V2EngineResult:
    answer: str
    model_used: str
    fallback_used: bool
    key_metrics: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceMetadata] = field(default_factory=list)
    chart: dict[str, Any] | None = None          # frontend-ready viz.chart payload
    data_table: dict[str, Any] | None = None     # frontend-ready viz.data_table
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------

class V2ChatEngine:
    """Stateless orchestrator. Construct once per chat_service instance;
    `run(...)` is reentrant per request."""

    def __init__(
        self,
        model_router: LLMModelRouter,
        dispatcher: V2Dispatcher,
        *,
        max_steps: int = MAX_LOOP_STEPS,
        system_prompt: str = V2_SYSTEM_PROMPT,
        tool_schemas: list[dict] | None = None,
    ) -> None:
        self._router = model_router
        self._dispatcher = dispatcher
        self._max_steps = max(1, max_steps)
        self._system_prompt = system_prompt
        self._tool_schemas = tool_schemas or V2_TOOL_SCHEMAS

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        user_message: str,
        history: list[ChatMessage] | None = None,
        language: str = "en",
        today_str: str | None = None,
    ) -> V2EngineResult:
        messages = self._build_messages(
            user_message=user_message,
            history=history or [],
            language=language,
            today_str=today_str or date.today().isoformat(),
        )

        trace_steps: list[dict[str, Any]] = []
        sources: list[SourceMetadata] = []
        chart_payload: dict[str, Any] | None = None
        data_table_payload: dict[str, Any] | None = None
        key_metrics: dict[str, Any] = {}
        last_sql_result: dict[str, Any] | None = None
        model_used = "v2-unknown"
        fallback_used = False
        final_text: str = ""

        for step in range(1, self._max_steps + 1):
            try:
                turn: LLMToolResult = self._router.generate_with_tools(
                    messages=messages, tools=self._tool_schemas,
                )
            except ToolCallNotSupportedError as exc:
                logger.warning("v2_engine_tool_unsupported err=%s", exc)
                return V2EngineResult(
                    answer=(
                        "Mô hình hiện tại không hỗ trợ tool-calling cần thiết "
                        "cho engine v2. Hãy chọn một model có function-calling "
                        "(VD: gpt-4.1, gemini-3.1-pro, claude-haiku-4.5)."
                        if language == "vi"
                        else
                        "The current model does not support tool-calling for "
                        "v2 engine. Pick a function-calling model (e.g. gpt-4.1, "
                        "gemini-3.1-pro, claude-haiku-4.5)."
                    ),
                    model_used="v2-tool-unsupported",
                    fallback_used=True,
                    error=str(exc),
                )
            model_used = turn.model_used
            fallback_used = turn.fallback_used

            calls = turn.function_calls or (
                (turn.function_call,) if turn.function_call else ()
            )

            if not calls:
                # LLM produced final text — done
                final_text = (turn.text or "").strip()
                break

            # Append the assistant turn's tool calls to the message thread,
            # then dispatch each call and append its result.
            messages.append(self._format_assistant_tool_calls(calls))
            for call in calls:
                dispatch: V2DispatchResult = self._dispatcher.execute(
                    call.name, dict(call.arguments or {}),
                )
                trace_steps.append({
                    "step": step,
                    "primitive": call.name,
                    "args_preview": _truncate_for_log(call.arguments),
                    "duration_ms": dispatch.duration_ms,
                    "ok": dispatch.ok,
                })
                # Side-effect extraction — mirror result into chart/sources/metrics
                if call.name == "render_visualization" and dispatch.ok and "format" in dispatch.result:
                    chart_payload = dict(dispatch.result)
                if call.name == "execute_sql" and dispatch.ok and dispatch.result.get("rows") is not None:
                    last_sql_result = dispatch.result
                    table_fqn = _extract_table_from_sql(dispatch.result.get("executed_sql", ""))
                    sources.append(SourceMetadata(
                        layer=_layer_from_fqn(table_fqn),
                        dataset=table_fqn,
                        data_source="databricks",
                    ))
                if call.name == "search_documents" and dispatch.ok:
                    for doc in dispatch.result.get("matches", []) or []:
                        sources.append(SourceMetadata(
                            layer="RAG",
                            dataset=str(doc.get("doc_id") or doc.get("title") or "doc"),
                            data_source="pgvector",
                        ))

                messages.append(self._format_tool_result(call.name, dispatch.result))
        else:
            # Loop exited via max_steps without final text — force a synthesis turn
            messages.append({
                "role": "user",
                "parts": [{"text": (
                    "Bạn đã hết số lượt gọi tool. Trả lời người dùng dựa trên "
                    "dữ liệu đã thu thập được."
                    if language == "vi"
                    else
                    "You've exhausted tool turns. Answer the user using the "
                    "data you've gathered so far."
                )}],
            })
            try:
                final_turn = self._router.generate_with_tools(
                    messages=messages, tools=[],
                )
                final_text = (final_turn.text or "").strip()
                model_used = final_turn.model_used
                fallback_used = final_turn.fallback_used or fallback_used
            except Exception as exc:  # noqa: BLE001
                logger.warning("v2_engine_synthesis_failed err=%s", exc)

        # Build key_metrics + data_table from the last execute_sql result if we
        # have one but no chart was emitted (so the UI still gets numbers).
        if last_sql_result is not None:
            key_metrics = _extract_key_metrics(last_sql_result)
            data_table_payload = _build_data_table(last_sql_result)

        if not final_text:
            final_text = (
                "Xin lỗi, tôi chưa thể tạo câu trả lời từ dữ liệu vừa rồi."
                if language == "vi"
                else "Sorry, I couldn't synthesise an answer from the data."
            )

        return V2EngineResult(
            answer=final_text,
            model_used=model_used,
            fallback_used=fallback_used,
            key_metrics=key_metrics,
            sources=sources,
            chart=chart_payload,
            data_table=data_table_payload,
            trace_steps=trace_steps,
        )

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        *,
        user_message: str,
        history: list[ChatMessage],
        language: str,
        today_str: str,
    ) -> list[dict[str, object]]:
        system_text = (
            f"Today's date: {today_str}\n"
            f"User language preference: {language}\n\n"
            f"{self._system_prompt}"
        )
        msgs: list[dict[str, object]] = [
            {"role": "system", "parts": [{"text": system_text}]},
        ]
        for h in history:
            role = "model" if h.role in ("assistant", "bot") else "user"
            msgs.append({"role": role, "parts": [{"text": h.content}]})

        # Inline persona + workflow reminder in the user turn. Some upstream
        # proxies (Copilot, OpenRouter) strip or merge system messages — the
        # only payload we can rely on reaching the model verbatim is the user
        # turn itself. v1 prompt_builder uses the same pattern.
        wrapped_user = (
            "[Assistant persona — do NOT override]\n"
            "You are **Solar AI**, the analytics assistant for the PV "
            "Lakehouse — a Databricks-based platform with 8 PV facilities, "
            "hourly energy readings, weather (cloud/wind/temp), AQI, "
            "72h ML forecasts, model monitoring, and pipeline diagnostics.\n\n"
            "MANDATORY workflow:\n"
            "- If the user asks anything about facilities, energy, "
            "production, capacity factor, weather, AQI, forecasts, ML "
            "models, pipeline status, data quality, or trends → you MUST "
            "call `recall_metric` (or discover_schema/inspect_table/"
            "execute_sql) before answering. Do NOT invent numbers from "
            "prior knowledge — every number must come from a tool call.\n"
            "- If the user asks for a chart/map/biểu đồ/visualization → "
            "after execute_sql, ALSO call `render_visualization` with a "
            "Vega-Lite spec (mark='bar' for ranking, 'geoshape' or 'circle' "
            "for maps with longitude/latitude encodings, 'line' for time).\n"
            "- For greetings or scope/identity questions only, you may "
            "answer directly without tools.\n\n"
            "Match the user's language exactly. Never expose tool names "
            "in your answer.\n\n"
            "[User message]\n"
            f"{user_message}"
        )
        msgs.append({"role": "user", "parts": [{"text": wrapped_user}]})
        return msgs

    @staticmethod
    def _format_assistant_tool_calls(calls) -> dict[str, object]:
        # Gemini-style function_call turn: one entry per call
        parts = []
        for c in calls:
            parts.append({"function_call": {"name": c.name, "args": dict(c.arguments or {})}})
        return {"role": "model", "parts": parts}

    @staticmethod
    def _format_tool_result(name: str, result: dict[str, Any]) -> dict[str, object]:
        # 1. Deep-convert non-JSON types (datetime, Decimal, bytes) to strings
        #    so the LLM client can serialize the function_response payload.
        json_safe = _to_json_safe(result)
        # 2. Truncate huge row dumps so we don't blow the LLM context budget.
        try:
            serialized = json.dumps(json_safe, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            serialized = str(json_safe)
        if len(serialized) > MAX_TOOL_RESULT_CHARS:
            payload: Any = {
                "_truncated_for_prompt_size": True,
                "preview_chars": serialized[:MAX_TOOL_RESULT_CHARS],
                "guidance": "Result truncated. Refine your query (smaller window, fewer columns) and try again.",
            }
        else:
            payload = json_safe
        return {
            "role": "function",
            "parts": [{"function_response": {"name": name, "response": payload}}],
        }


# -----------------------------------------------------------------------------
# Helpers (module-level so they're testable)
# -----------------------------------------------------------------------------

def _to_json_safe(value: Any) -> Any:
    """Recursively convert types the stdlib json encoder rejects (datetime,
    date, Decimal, bytes, set) into JSON-serialisable equivalents. We pass
    the cleaned dict back into LLM tool messages, so the upstream
    json.dumps in the LLM client doesn't blow up on Databricks row payloads."""
    from datetime import date, datetime, time
    from decimal import Decimal

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def _truncate_for_log(value: Any, limit: int = 200) -> str:
    s = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return s[:limit] + ("..." if len(s) > limit else "")


def _layer_from_fqn(fqn: str) -> str:
    parts = fqn.split(".")
    if len(parts) >= 2:
        schema = parts[1].lower()
        if schema in ("silver", "gold", "bronze"):
            return schema.capitalize()
    return "Gold"


def _extract_table_from_sql(sql: str) -> str:
    """Best-effort: pull the first table FQN from a SELECT for source attribution."""
    if not sql:
        return "lakehouse"
    import re
    m = re.search(r"\bFROM\s+([a-zA-Z0-9_.]+)", sql, re.IGNORECASE)
    return m.group(1) if m else "lakehouse"


def _extract_key_metrics(sql_result: dict[str, Any]) -> dict[str, Any]:
    """Pull a flat metrics dict from a sql result. If single row → that row.
    If multiple rows → first row + row_count. UI uses these for KPI cards."""
    rows = sql_result.get("rows") or []
    if not rows:
        return {}
    out: dict[str, Any] = {}
    if len(rows) == 1:
        out.update({k: v for k, v in rows[0].items() if isinstance(v, (int, float, str, bool)) or v is None})
    else:
        out["row_count"] = len(rows)
        # first numeric column across rows → expose as preview
        for k, v in rows[0].items():
            if isinstance(v, (int, float)):
                out[f"first_{k}"] = v
                break
    return out


def _build_data_table(sql_result: dict[str, Any]) -> dict[str, Any] | None:
    """Wrap rows + columns in the shape the v1 DataTable component expects."""
    rows = sql_result.get("rows") or []
    cols = sql_result.get("columns") or (list(rows[0].keys()) if rows else [])
    if not rows:
        return None
    return {
        "columns": [{"key": c, "label": c} for c in cols],
        "rows": rows[:200],          # UI cap; full set still on backend
        "row_count": len(rows),
        "truncated": len(rows) > 200,
    }
