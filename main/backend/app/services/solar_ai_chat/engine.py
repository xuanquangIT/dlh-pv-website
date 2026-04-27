"""Solar AI Chat — agentic engine.

The engine runs a thin tool-calling loop:

    LLM(messages, TOOL_SCHEMAS)
      -> tool_call -> Dispatcher.execute(...) -> result
      -> append result -> repeat
      -> final text answer

Outputs a normalized dict ``chat_service`` adapts into a ``SolarChatResponse``
(answer + key_metrics + sources + chart + trace).

Design doc: implementations/solar_chat_architecture_redesign_2026-04-26.md
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from app.schemas.solar_ai_chat import ChatMessage, SourceMetadata
from app.services.solar_ai_chat.llm_client import (
    LLMModelRouter,
    LLMToolResult,
    ToolCallNotSupportedError,
)
from app.services.solar_ai_chat.dispatcher import Dispatcher, DispatchResult
from app.services.solar_ai_chat.tool_schemas import (
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
)

logger = logging.getLogger(__name__)

MAX_LOOP_STEPS = 8           # hard cap on tool turns
MAX_TOOL_RESULT_CHARS = 12_000  # truncate huge SQL row payloads going back to LLM
MAX_PARALLEL_CALLS_PER_TURN = 3  # cap fan-out (Grok loves to emit 6 inspects)


# -----------------------------------------------------------------------------
# Result envelope
# -----------------------------------------------------------------------------

@dataclass
class ChatEngineResult:
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

class ChatEngine:
    """Stateless orchestrator. Construct once per chat_service instance;
    `run(...)` is reentrant per request."""

    def __init__(
        self,
        model_router: LLMModelRouter,
        dispatcher: Dispatcher,
        *,
        max_steps: int = MAX_LOOP_STEPS,
        system_prompt: str = SYSTEM_PROMPT,
        tool_schemas: list[dict] | None = None,
    ) -> None:
        self._router = model_router
        self._dispatcher = dispatcher
        self._max_steps = max(1, max_steps)
        self._system_prompt = system_prompt
        self._tool_schemas = tool_schemas or TOOL_SCHEMAS

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
        force_chart: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ChatEngineResult:
        """Run the agentic loop.

        progress_callback (optional): invoked at every primitive call boundary
        with a dict like {"event": "tool_start"|"tool_end", "step": int,
        "primitive": str, "duration_ms": int, "ok": bool}. Used by the SSE
        streaming bridge to surface live progress to the UI; non-stream
        callers can ignore it.
        """
        def _emit(payload: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception as cb_err:  # noqa: BLE001 — callback failures must not poison the engine
                logger.warning("engine_progress_callback_failed err=%s", cb_err)

        parsed_hints = _parse_user_hints(user_message)
        wants_chart = force_chart or _detect_chart_intent(user_message)

        # Off-topic scope guard. The system prompt asks the LLM to redirect
        # off-topic questions, but weak models routinely ignore that and
        # call execute_sql anyway, returning irrelevant numbers. Catch
        # obvious off-topic patterns up-front and return a polite redirect
        # without burning tool calls.
        if _is_off_topic(user_message):
            logger.info("engine_off_topic_redirect msg=%r", user_message[:120])
            return ChatEngineResult(
                answer=_build_scope_redirect(language),
                model_used="engine-scope-guard",
                fallback_used=False,
                trace_steps=[],
            )

        # Conceptual / definitional question guard.
        # Q: "Performance Ratio là gì", "What is capacity factor",
        # "Công thức tính nRMSE" — purely educational. Don't run SQL,
        # don't pull KPIs. The LLM has the knowledge; just answer with text.
        # Without this, recall_metric falls through to system_overview as a
        # weak match, auto-execute runs, and the user sees irrelevant KPI
        # cards alongside the definition.
        if _is_conceptual_question(user_message):
            logger.info("engine_conceptual_question msg=%r", user_message[:120])
            return self._answer_conceptual(
                user_message, language, history or [],
                today_str or date.today().isoformat(),
            )

        messages = self._build_messages(
            user_message=user_message,
            history=history or [],
            language=language,
            today_str=today_str or date.today().isoformat(),
            parsed_hints=parsed_hints,
            wants_chart=wants_chart,
        )

        trace_steps: list[dict[str, Any]] = []
        sources: list[SourceMetadata] = []
        chart_payload: dict[str, Any] | None = None
        data_table_payload: dict[str, Any] | None = None
        key_metrics: dict[str, Any] = {}
        last_sql_result: dict[str, Any] | None = None
        model_used = "engine-unknown"
        fallback_used = False
        final_text: str = ""
        # Duplicate-call detector: smaller models (Nemotron, gpt-oss) get stuck
        # calling the same tool repeatedly. We use per-tool thresholds because
        # `recall_metric` is a SEARCH tool — fan-out across 3-4 phrasings is
        # legitimate behaviour for compound queries (e.g. "compare X and Y"
        # may reasonably search "facility metadata", "energy ranking", and
        # "weather impact"). Banning it after 3 consecutive turns burns the
        # model's only path to canonical SQL templates.
        recent_call_signatures: list[str] = []
        recent_tool_sets: list[frozenset[str]] = []
        # Default: ban after appearing in 3 consecutive turns (threshold=2).
        # `recall_metric` gets +1 — search fan-out is expected for compound
        # questions; we only ban after 4 consecutive turns of pure searching.
        DEFAULT_DUPLICATE_THRESHOLD = 2
        PER_TOOL_DUPLICATE_THRESHOLD = {
            "recall_metric": 3,
            "discover_schema": 3,
        }
        banned_tools: set[str] = set()
        # Exact-call signatures we've already executed. If the model issues
        # the SAME (name, args) twice we ban the tool immediately — gemini-flash
        # routinely re-calls execute_sql with byte-identical SQL after a
        # successful run instead of synthesising.
        executed_call_keys: set[str] = set()
        # Track the highest-confidence recall_metric top match so we can
        # auto-execute its SQL if the model gets stuck recall-looping. This
        # is the single biggest reliability win for weak models — when the
        # YAML semantic layer obviously already knows the answer, we don't
        # need the LLM to make the trivial "use the top match" decision.
        last_recall_top: dict[str, Any] | None = None
        # Track which metric the latest successful execute_sql came from so
        # we can detect stale data when the model's intent shifts mid-loop.
        # Without this, a query like "tỉ lệ đóng góp" can have its first
        # successful SQL be facility_locations_map (lat/lng), then later
        # JOIN attempts fail, and forced synthesis serves the lat/lng
        # numbers as if they answered the question — a hallucination.
        last_sql_metric: str | None = None

        for step in range(1, self._max_steps + 1):
            # If loop-breaker has flagged a tool, remove it from the schema so
            # the model literally cannot call it again.
            active_tools = [
                t for t in self._tool_schemas if t.get("name") not in banned_tools
            ]
            try:
                turn: LLMToolResult = self._router.generate_with_tools(
                    messages=messages, tools=active_tools,
                )
            except RuntimeError as exc:
                # Some providers return empty responses when all tools are
                # banned. Force a synthesis turn with all tools restored so
                # the model produces text from gathered data.
                if "neither tool call nor text" in str(exc):
                    logger.warning(
                        "engine_empty_response step=%d banned=%s — forcing synthesis",
                        step, sorted(banned_tools),
                    )
                    auto = self._maybe_auto_execute(
                        last_recall_top, last_sql_result, parsed_hints,
                        step, trace_steps, messages, _emit, sources, last_sql_metric=last_sql_metric,
                    )
                    if auto is not None:
                        last_sql_result = auto
                        last_sql_metric = (last_recall_top or {}).get("name")
                    final_text, model_used, fallback_used = self._fresh_synthesis(
                        user_message=user_message,
                        language=language,
                        last_sql_result=last_sql_result,
                        today_str=today_str or date.today().isoformat(),
                        fallback_model=model_used,
                        fallback_used=fallback_used,
                    )
                    break
                raise
            except ToolCallNotSupportedError as exc:
                logger.warning("engine_tool_unsupported err=%s", exc)
                return ChatEngineResult(
                    answer=(
                        "Mô hình hiện tại không hỗ trợ tool-calling cần thiết "
                        "cho engine. Hãy chọn một model có function-calling "
                        "(VD: gpt-4.1, gemini-3.1-pro, claude-haiku-4.5)."
                        if language == "vi"
                        else
                        "The current model does not support tool-calling for "
                        "engine. Pick a function-calling model (e.g. gpt-4.1, "
                        "gemini-3.1-pro, claude-haiku-4.5)."
                    ),
                    model_used="engine-tool-unsupported",
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
                final_text = _strip_inline_cot(
                    _strip_reasoning(turn.text or ""), language,
                )
                break

            # Cap parallel-call fan-out. Some models (grok-4-fast) emit 5-6
            # tool_calls per turn — usually inspect_tables on hallucinated
            # table names. Without a cap, an 8-turn budget can burn 30+
            # primitive calls in one go. Take only the first N.
            if len(calls) > MAX_PARALLEL_CALLS_PER_TURN:
                logger.warning(
                    "engine_parallel_calls_capped step=%d emitted=%d kept=%d",
                    step, len(calls), MAX_PARALLEL_CALLS_PER_TURN,
                )
                calls = calls[:MAX_PARALLEL_CALLS_PER_TURN]

            # HARD ENFORCEMENT of bans. Schema filtering alone doesn't stop
            # weak models (gemini-flash-lite, glm) — they ignore the schema
            # and re-call banned tools anyway. Intercept here: any call to a
            # banned tool gets an error tool_result; if EVERY call in the
            # turn is banned, jump straight to forced synthesis.
            if banned_tools and all(c.name in banned_tools for c in calls):
                logger.warning(
                    "engine_all_calls_banned step=%d called=%s banned=%s — forcing synthesis",
                    step, [c.name for c in calls], sorted(banned_tools),
                )
                # Last-chance auto-execute before bailing to synthesis: if
                # we have a recall top match but no SQL result yet, run it.
                auto = self._maybe_auto_execute(
                    last_recall_top, last_sql_result, parsed_hints,
                    step, trace_steps, messages, _emit, sources, last_sql_metric=last_sql_metric,
                )
                if auto is not None:
                    last_sql_result = auto
                    last_sql_metric = (last_recall_top or {}).get("name")
                final_text, model_used, fallback_used = self._fresh_synthesis(
                    user_message=user_message,
                    language=language,
                    last_sql_result=last_sql_result,
                    today_str=today_str or date.today().isoformat(),
                    fallback_model=model_used,
                    fallback_used=fallback_used,
                )
                break
            # Otherwise rewrite the calls list to drop banned ones, telling
            # the model via tool_result that those got rejected.
            if banned_tools and any(c.name in banned_tools for c in calls):
                allowed_calls = []
                for c in calls:
                    if c.name in banned_tools:
                        # Emit a tool_result error so model sees the rejection
                        messages.append(self._format_assistant_tool_calls([c]))
                        messages.append(self._format_tool_result(c.name, {
                            "error": f"Tool '{c.name}' is disabled. Use the data already returned.",
                            "_banned": True,
                        }))
                    else:
                        allowed_calls.append(c)
                calls = allowed_calls
                if not calls:
                    continue

            # Exact-duplicate detection. If the model re-issues an identical
            # (name, args) call we've already dispatched, ban that tool now —
            # weak models like gemini-flash will otherwise loop on the same
            # successful execute_sql 5+ times before exhausting max_steps.
            #
            # EXCEPTION: `recall_metric` is a search tool. The model may
            # re-issue an identical search if the matches looked weak (top
            # score low) — banning recall_metric on the 2nd call leaves
            # the model with no path to canonical SQL templates and forces
            # forced-synthesis with empty data. For recall_metric we just
            # nudge the model with a correction message and let it try the
            # top match's SQL via execute_sql instead.
            exact_dups: set[str] = set()
            for c in calls:
                args_repr = json.dumps(dict(c.arguments or {}), sort_keys=True, default=str)
                key = f"{c.name}::{args_repr}"
                if (
                    key in executed_call_keys
                    and c.name not in banned_tools
                    and c.name != "recall_metric"
                ):
                    exact_dups.add(c.name)
                executed_call_keys.add(key)
            if exact_dups:
                banned_tools.update(exact_dups)
                logger.warning(
                    "engine_exact_dup_detected dups=%s banned_now=%s",
                    sorted(exact_dups), sorted(banned_tools),
                )
                # Re-dispatch this turn's calls so the trace stays consistent,
                # then inject a correction message and skip to next turn (where
                # the now-banned tool is gone from the schema).
                messages.append(self._format_assistant_tool_calls(calls))
                for c in calls:
                    _emit({"event": "tool_start", "step": step, "primitive": c.name})
                    d = self._dispatcher.execute(c.name, dict(c.arguments or {}))
                    if c.name == "execute_sql" and d.ok and d.result.get("rows") is not None:
                        last_sql_result = d.result
                    if c.name == "recall_metric" and d.ok:
                        matches = (d.result or {}).get("matches") or []
                        if matches:
                            last_recall_top = matches[0]
                    trace_steps.append({
                        "step": step, "primitive": c.name,
                        "args_preview": _truncate_for_log(c.arguments),
                        "duration_ms": d.duration_ms, "ok": d.ok,
                    })
                    _emit({"event": "tool_end", "step": step, "primitive": c.name,
                           "duration_ms": d.duration_ms, "ok": d.ok})
                    messages.append(self._format_tool_result(c.name, d.result))
                messages.append({
                    "role": "user",
                    "parts": [{"text": (
                        f"You re-issued an identical call to {sorted(exact_dups)}. "
                        "That tool is now disabled. Use the data already returned "
                        "to answer the user — do not try to fetch it again."
                    )}],
                })
                # AUTO-EXECUTE: if recall_metric was banned and we have a top
                # match but no successful SQL yet, run the metric's rendered
                # SQL ourselves so the model has data to summarise.
                auto = self._maybe_auto_execute(
                    last_recall_top, last_sql_result, parsed_hints,
                    step, trace_steps, messages, _emit, sources, last_sql_metric=last_sql_metric,
                )
                if auto is not None:
                    last_sql_result = auto
                    last_sql_metric = (last_recall_top or {}).get("name")
                continue

            # Detect call-name loops. Look for any tool that appears in EVERY
            # one of the last (DUPLICATE_THRESHOLD + 1) turns — robust against
            # both single-call loops AND parallel-call loops where the model
            # varies args while reusing the same tool (e.g. discover_schema
            # with domain=energy / "" / pipeline / forecast across turns).
            turn_tools = frozenset(c.name for c in calls)
            recent_tool_sets.append(turn_tools)
            recent_call_signatures.append("|".join(sorted(turn_tools)))
            logger.info(
                "engine_step step=%d turn_tools=%s history=%s",
                step, sorted(turn_tools), [sorted(s) for s in recent_tool_sets[-4:]],
            )
            # Per-tool persistence check. A tool is "persistent" when it
            # appears in every turn of its OWN threshold window (default 3,
            # search tools get 4). Stricter than the previous global window
            # because the per-tool window adapts to tool semantics.
            persistent_tools: set[str] = set()
            for candidate in (set().union(*recent_tool_sets) if recent_tool_sets else set()):
                if candidate in banned_tools:
                    continue
                threshold = PER_TOOL_DUPLICATE_THRESHOLD.get(
                    candidate, DEFAULT_DUPLICATE_THRESHOLD,
                )
                if len(recent_tool_sets) <= threshold:
                    continue
                window = recent_tool_sets[-(threshold + 1):]
                if all(candidate in turn_set for turn_set in window):
                    persistent_tools.add(candidate)
            if persistent_tools:
                # Dispatch this set so traces stay consistent, then BAN the
                # tools from future turns. Removing them from the schema is a
                # forcing function the model can't ignore (vs. textual nudges
                # which weak models routinely override).
                offending = sorted(persistent_tools)
                banned_tools.update(persistent_tools)
                logger.warning(
                    "engine_duplicate_loop_detected persistent=%s banned_now=%s",
                    offending, sorted(banned_tools),
                )
                messages.append(self._format_assistant_tool_calls(calls))
                last_dispatch_results: list[tuple[str, dict[str, Any]]] = []
                for c in calls:
                    _emit({"event": "tool_start", "step": step, "primitive": c.name})
                    d = self._dispatcher.execute(c.name, dict(c.arguments or {}))
                    if c.name == "execute_sql" and d.ok and d.result.get("rows") is not None:
                        last_sql_result = d.result
                    if c.name == "recall_metric" and d.ok:
                        m = (d.result or {}).get("matches") or []
                        if m:
                            last_recall_top = m[0]
                    trace_steps.append({
                        "step": step, "primitive": c.name,
                        "args_preview": _truncate_for_log(c.arguments),
                        "duration_ms": d.duration_ms, "ok": d.ok,
                    })
                    _emit({"event": "tool_end", "step": step, "primitive": c.name,
                           "duration_ms": d.duration_ms, "ok": d.ok})
                    messages.append(self._format_tool_result(c.name, d.result))
                    last_dispatch_results.append((c.name, d.result))

                # If recall_metric was banned, inline the top match's
                # sql_template + a rendered example into the correction so
                # weak models get a copy-pasteable SQL to run with execute_sql.
                inline_sql_hint = ""
                if "recall_metric" in offending:
                    for fn_name, result in last_dispatch_results:
                        if fn_name != "recall_metric":
                            continue
                        matches = (result or {}).get("matches") or []
                        if not matches:
                            continue
                        top = matches[0]
                        rendered = top.get("sql_template", "")
                        for p in top.get("parameters", []) or []:
                            pname = str(p.get("name", ""))
                            placeholder = "{" + pname + "}"
                            # User-message hints override YAML defaults so a
                            # query like "7 ngày qua" doesn't fall back to 30.
                            override = parsed_hints.get(pname)
                            default = override if override is not None else p.get("default")
                            if default is None and p.get("values"):
                                default = p["values"][0]
                            if default is not None:
                                rendered = rendered.replace(placeholder, str(default))
                        inline_sql_hint = (
                            f"\n\nThe top metric match was {top.get('name')!r}. "
                            f"Its rendered SQL (parameters substituted with defaults) is:\n"
                            f"```sql\n{rendered.strip()}\n```\n"
                            "CALL execute_sql with this exact SQL string now."
                        )
                        break

                messages.append({
                    "role": "user",
                    "parts": [{"text": (
                        f"You called {offending} multiple times in a row "
                        "without making progress. Those tools are now "
                        "disabled for the rest of this turn."
                        + inline_sql_hint
                        + (
                            "" if inline_sql_hint else
                            " Use the data already returned: pick a metric's "
                            "sql_template, substitute parameters with defaults, "
                            "and call execute_sql. If no match, call inspect_table "
                            "on a likely table from discover_schema and write a SELECT."
                        )
                    )}],
                })
                auto = self._maybe_auto_execute(
                    last_recall_top, last_sql_result, parsed_hints,
                    step, trace_steps, messages, _emit, sources, last_sql_metric=last_sql_metric,
                )
                if auto is not None:
                    last_sql_result = auto
                    last_sql_metric = (last_recall_top or {}).get("name")
                continue

            # Append the assistant turn's tool calls to the message thread,
            # then dispatch each call and append its result.
            messages.append(self._format_assistant_tool_calls(calls))
            for call in calls:
                _emit({"event": "tool_start", "step": step, "primitive": call.name})
                dispatch: DispatchResult = self._dispatcher.execute(
                    call.name, dict(call.arguments or {}),
                )
                trace_steps.append({
                    "step": step,
                    "primitive": call.name,
                    "args_preview": _truncate_for_log(call.arguments),
                    "duration_ms": dispatch.duration_ms,
                    "ok": dispatch.ok,
                })
                _emit({"event": "tool_end", "step": step, "primitive": call.name,
                       "duration_ms": dispatch.duration_ms, "ok": dispatch.ok})
                # Side-effect extraction — mirror result into chart/sources/metrics
                if call.name == "render_visualization" and dispatch.ok and "format" in dispatch.result:
                    chart_payload = _to_json_safe(dict(dispatch.result))
                if call.name == "execute_sql" and dispatch.ok and dispatch.result.get("rows") is not None:
                    last_sql_result = dispatch.result
                    # Tag this result with the recall_top metric whose
                    # SQL template likely generated it. We use a fuzzy match:
                    # if the current SQL contains the FQN from recall_top's
                    # template, it's that metric. Otherwise unknown.
                    last_sql_metric = _infer_metric_from_sql(
                        dict(call.arguments or {}).get("sql", "")
                        or dispatch.result.get("executed_sql", ""),
                        last_recall_top,
                    )
                    table_fqn = _extract_table_from_sql(
                        dispatch.result.get("executed_sql", "")
                        or dict(call.arguments or {}).get("sql", "")
                    )
                    sources.append(SourceMetadata(
                        layer=_layer_from_fqn(table_fqn),
                        dataset=table_fqn,
                        data_source="databricks",
                    ))
                if call.name == "recall_metric" and dispatch.ok:
                    matches = (dispatch.result or {}).get("matches") or []
                    if matches:
                        last_recall_top = matches[0]
                messages.append(self._format_tool_result(call.name, dispatch.result))
        else:
            # Loop exited via max_steps without final text — fresh synthesis
            final_text, model_used, fallback_used = self._fresh_synthesis(
                user_message=user_message,
                language=language,
                last_sql_result=last_sql_result,
                today_str=today_str or date.today().isoformat(),
                fallback_model=model_used,
                fallback_used=fallback_used,
            )

        # Build key_metrics + data_table from the last execute_sql result if we
        # have one but no chart was emitted (so the UI still gets numbers).
        if last_sql_result is not None:
            key_metrics = _extract_key_metrics(last_sql_result)
            data_table_payload = _build_data_table(last_sql_result)

        # Deterministic chart fallback. If the user explicitly asked for a
        # chart (toggle / keyword) but the model never called
        # render_visualization, fabricate a sensible Vega-Lite spec from
        # the SQL rows so the UI still shows something. Better than a
        # silent no-chart response when the user clicked "Visualize".
        if (
            wants_chart
            and chart_payload is None
            and last_sql_result is not None
            and len(last_sql_result.get("rows") or []) >= 2
        ):
            # Skip single-row aggregates — those belong in KPI cards, not a
            # chart. A 1-bar Vega-Lite chart for system_overview is noise.
            intent_hint: str | None = None
            low = (user_message or "").lower()
            if any(k in low for k in (
                "tỉ lệ", "ti le", "tỷ lệ", "ty le",
                "đóng góp", "dong gop", "phần trăm", "phan tram",
                "tỉ trọng", "ti trong", "tỷ trọng",
                "share", "contribution", "percentage", "percent of",
                "split by", "breakdown",
            )):
                intent_hint = "share"
            elif any(k in low for k in (
                "correlation", "tương quan", "tuong quan", "vs ",
                "tương quan giữa", "ảnh hưởng", "anh huong", "impact",
            )):
                intent_hint = "correlation"
            chart_payload = _auto_chart_from_rows(
                last_sql_result.get("rows") or [],
                title=("Visualization" if language != "vi" else "Trực quan hoá"),
                intent_hint=intent_hint,
            )

        if not final_text:
            final_text = (
                "Xin lỗi, tôi chưa thể tạo câu trả lời từ dữ liệu vừa rồi."
                if language == "vi"
                else "Sorry, I couldn't synthesise an answer from the data."
            )

        # POST-HOC HEDGE GUARD. If we have real SQL rows but the model
        # produced a refusal/menu/clarifying-question response (e.g. "I can
        # run that for you — which option do you prefer?"), replace with the
        # deterministic draft. This applies to BOTH the normal-text-exit
        # path AND the forced-synthesis path; the former wasn't checked
        # before, so gpt-5-mini's polite-deflect responses leaked through.
        rows_for_check = (last_sql_result or {}).get("rows") or []
        if rows_for_check and _is_hedging_response(final_text, rows_for_check):
            draft = _render_answer_draft(rows_for_check, language)
            if draft:
                logger.warning(
                    "engine_post_hoc_hedge_replace model_text=%r → draft (%d chars)",
                    final_text[:160], len(draft),
                )
                final_text = _strip_draft_instruction_suffix(draft)
                fallback_used = True

        # MISSING-DATA SUPPRESSION. When the answer explicitly says the
        # asked-about column isn't in the dataset, suppress the chart +
        # data_table + key_metrics so we don't ship irrelevant numbers
        # underneath the refusal. Without this, asking "humidity impact"
        # would render a wind_speed scatter and table beside the
        # "no humidity data" answer.
        if _answer_signals_missing_column(final_text):
            logger.info(
                "engine_missing_column_refusal — suppressing chart + data_table",
            )
            chart_payload = None
            data_table_payload = None
            key_metrics = {}

        return ChatEngineResult(
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
        parsed_hints: dict[str, Any] | None = None,
        wants_chart: bool = False,
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
            sender = getattr(h, "sender", None) or getattr(h, "role", "user")
            role = "model" if str(sender).lower() in ("assistant", "bot", "model") else "user"
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
            "7-day ML forecasts (D+1, D+3, D+5, D+7), model monitoring, and pipeline diagnostics.\n\n"
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
            "- ONLY for explicit greetings (\"hi\", \"hello\", \"chào\", "
            "\"hey\") or self-identity questions (\"who are you\", \"what "
            "can you do\", \"bạn là ai\", \"bạn làm được gì\") may you answer "
            "directly without tools. Phrases like \"tổng quan\" / "
            "\"overview\" / \"summary\" / \"system status\" / \"tình hình "
            "hệ thống\" are DATA questions — you MUST call recall_metric "
            "with query='system overview' first, then execute_sql.\n\n"
            "Match the user's language exactly. Never expose tool names "
            "in your answer.\n\n"
        )
        if parsed_hints:
            hints_str = ", ".join(f"{k}={v}" for k, v in parsed_hints.items())
            wrapped_user += (
                f"[Extracted parameters from the user message — use these "
                f"when calling recall_metric / execute_sql instead of "
                f"defaults]\n{hints_str}\n\n"
            )
        if wants_chart:
            wrapped_user += (
                "[Visualization requirement — MANDATORY]\n"
                "The user explicitly asked for a chart / map / visualization. "
                "After execute_sql succeeds, you MUST call render_visualization "
                "with a Vega-Lite spec. Pick the mark from the data shape:\n"
                "  • Ranking / top-N → mark='bar'\n"
                "  • Time series → mark='line' (encoding x: temporal)\n"
                "  • Geographic / map → mark='circle' or 'geoshape' with "
                "longitude+latitude encodings\n"
                "  • Correlation / scatter → mark='point'\n"
                "Pass the rows from execute_sql unchanged as the `data` field.\n\n"
            )
        wrapped_user += "[User message]\n" + user_message
        # (concat above replaces the single-string literal so we can splice
        # in the parsed-parameter block when present.)
        msgs.append({"role": "user", "parts": [{"text": wrapped_user}]})
        return msgs

    def _answer_conceptual(
        self,
        user_message: str,
        language: str,
        history: list[ChatMessage],
        today_str: str,
    ) -> ChatEngineResult:
        """Answer a definition / explainer question with LLM knowledge —
        no tools, no KPIs, no data table. Keeps the response focused on
        the concept the user actually asked about."""
        if language == "vi":
            sys_text = (
                f"Hôm nay: {today_str}. Bạn là Solar AI cho PV Lakehouse. "
                "Người dùng hỏi một câu khái niệm / định nghĩa — KHÔNG cần "
                "truy vấn dữ liệu. Trả lời ngắn gọn, chính xác bằng tiếng Việt: "
                "định nghĩa, công thức (nếu có), phạm vi giá trị thông thường, "
                "và 1-2 lưu ý thực tế. Không bịa số liệu."
            )
        else:
            sys_text = (
                f"Today: {today_str}. You are Solar AI for the PV Lakehouse. "
                "The user asked a conceptual / definition question — NO data "
                "lookup needed. Answer concisely and accurately: definition, "
                "formula (if any), typical value range, and 1-2 practical "
                "notes. Do not fabricate numbers."
            )
        msgs: list[dict[str, object]] = [
            {"role": "system", "parts": [{"text": sys_text}]},
        ]
        for h in history[-4:]:
            sender = getattr(h, "sender", None) or getattr(h, "role", "user")
            role = "model" if str(sender).lower() in ("assistant", "bot", "model") else "user"
            msgs.append({"role": role, "parts": [{"text": h.content}]})
        msgs.append({"role": "user", "parts": [{"text": user_message}]})

        try:
            resp = self._router.generate_with_tools(messages=msgs, tools=[])
            answer = _strip_inline_cot(
                _strip_reasoning(resp.text or ""), language,
            )
            return ChatEngineResult(
                answer=answer or (
                    "Xin lỗi, tôi chưa thể trả lời câu hỏi đó."
                    if language == "vi"
                    else "Sorry, I couldn't answer that question."
                ),
                model_used=resp.model_used,
                fallback_used=resp.fallback_used,
                trace_steps=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("engine_conceptual_failed err=%s", exc)
            return ChatEngineResult(
                answer=(
                    "Xin lỗi, hiện tôi chưa trả lời được câu hỏi này."
                    if language == "vi"
                    else "Sorry, I can't answer that right now."
                ),
                model_used="engine-conceptual-error",
                fallback_used=True,
                error=str(exc),
            )

    def _maybe_auto_execute(
        self,
        recall_top: dict[str, Any] | None,
        already_have_sql: dict[str, Any] | None,
        parsed_hints: dict[str, Any],
        step: int,
        trace_steps: list[dict[str, Any]],
        messages: list[dict[str, object]],
        emit: Callable[[dict[str, Any]], None],
        sources: list[SourceMetadata] | None = None,
        last_sql_metric: str | None = None,
    ) -> dict[str, Any] | None:
        """Auto-execute the top recall_metric match's SQL if the LLM gave up.

        Triggers when:
          • recall_top is set (we know which metric to use)
          • last_sql_result is None (no successful SQL yet)
          • the metric's SQL template can be fully rendered with parsed_hints
            + YAML defaults (no missing parameters)

        On success returns the new sql_result dict and appends the dispatch
        to trace_steps + messages so the model thread looks normal.

        Stale-data detection: if `already_have_sql` exists but
        `last_sql_metric` differs from the current recall_top metric, the
        existing data is from a wrong-intent SQL (e.g. lat/lng when user
        asked for energy share). In that case we refresh by running the
        right metric's SQL — caller is responsible for replacing
        `last_sql_result` with our return value.
        """
        if recall_top is None:
            return None
        recall_name = str(recall_top.get("name") or "")
        if already_have_sql is not None:
            # Refresh only when intent has clearly shifted to a different metric.
            if last_sql_metric and recall_name and last_sql_metric == recall_name:
                return None
            if not recall_name:
                return None
            logger.info(
                "engine_auto_execute_refresh stale_metric=%s → recall_top=%s",
                last_sql_metric, recall_name,
            )
        rendered = recall_top.get("sql_template", "") or ""
        if not rendered.strip():
            return None
        # Render parameters
        for p in recall_top.get("parameters", []) or []:
            pname = str(p.get("name", ""))
            placeholder = "{" + pname + "}"
            override = parsed_hints.get(pname) if parsed_hints else None
            default = override if override is not None else p.get("default")
            if default is None and p.get("values"):
                default = p["values"][0]
            if default is None:
                # Missing required parameter — give up rather than guess.
                logger.info(
                    "engine_auto_execute_skipped reason=missing_param param=%s metric=%s",
                    pname, recall_top.get("name"),
                )
                return None
            rendered = rendered.replace(placeholder, str(default))

        logger.info(
            "engine_auto_execute_triggered metric=%s sql=%s",
            recall_top.get("name"), rendered[:120].replace("\n", " "),
        )
        emit({"event": "tool_start", "step": step, "primitive": "execute_sql"})
        d = self._dispatcher.execute("execute_sql", {"sql": rendered})
        emit({"event": "tool_end", "step": step, "primitive": "execute_sql",
              "duration_ms": d.duration_ms, "ok": d.ok})
        trace_steps.append({
            "step": step,
            "primitive": "execute_sql",
            "args_preview": _truncate_for_log({"sql": rendered, "_auto_executed": True}),
            "duration_ms": d.duration_ms,
            "ok": d.ok,
        })
        # Inject as if the model had called execute_sql itself, so any
        # subsequent synthesis turn sees a coherent thread.
        synthetic_call = type("AutoCall", (), {
            "name": "execute_sql",
            "arguments": {"sql": rendered},
        })()
        messages.append(self._format_assistant_tool_calls([synthetic_call]))
        messages.append(self._format_tool_result("execute_sql", d.result))
        if d.ok and d.result.get("rows") is not None:
            if sources is not None:
                fqn = _extract_table_from_sql(
                    d.result.get("executed_sql", "") or rendered
                )
                sources.append(SourceMetadata(
                    layer=_layer_from_fqn(fqn),
                    dataset=fqn,
                    data_source="databricks",
                ))
            return d.result
        return None

    def _fresh_synthesis(
        self,
        *,
        user_message: str,
        language: str,
        last_sql_result: dict[str, Any] | None,
        today_str: str,
        fallback_model: str,
        fallback_used: bool,
    ) -> tuple[str, str, bool]:
        """Run synthesis with a CLEAN, minimal context (no polluted history).

        Sending the synthesis directive as the final user turn of a long
        tool-call thread doesn't work for hedging models like gpt-5-mini —
        they keep echoing the "menu" pattern from earlier in the thread.
        Building a fresh 2-message context (system + user-with-data) makes
        the model focus on the numbers in front of it.

        If the model still hedges (refusal phrases, no numbers from data,
        question marks asking for clarification) we fall back to the
        deterministic draft.
        """
        rows = (last_sql_result or {}).get("rows") or []
        draft = _render_answer_draft(rows, language) if rows else ""

        if not rows:
            return (
                draft or (
                    "Xin lỗi, tôi chưa lấy được dữ liệu cho yêu cầu này."
                    if language == "vi"
                    else "Sorry, I couldn't retrieve data for that request."
                ),
                fallback_model,
                fallback_used or True,
            )

        # Build a clean fresh context: just system + one user message with
        # the data block + the draft + a "rephrase concisely" instruction.
        # Crucially NO tool-call history — that's what poisons the synthesis
        # for gpt-5-mini (it picks up the "menu/option" pattern from the
        # thread and keeps echoing it).
        try:
            sample = rows[: min(10, len(rows))]
            rendered = json.dumps(sample, ensure_ascii=False, default=str, indent=2)
        except Exception:  # noqa: BLE001
            rendered = str(rows[:5])

        if language == "vi":
            sys_text = (
                f"Hôm nay: {today_str}. Bạn là Solar AI cho PV Lakehouse. "
                "Trả lời ngắn gọn, chính xác, KHÔNG bịa số."
            )
            user_text = (
                f"Câu hỏi của người dùng:\n{user_message}\n\n"
                f"Tôi đã chạy truy vấn và có dữ liệu sau ({len(rows)} dòng, "
                f"hiển thị {min(10, len(rows))} dòng đầu):\n```json\n{rendered}\n```\n\n"
                f"Bản nháp trả lời:\n---\n{draft}\n---\n\n"
                "→ Hãy trả lời người dùng bằng 2-3 câu, dùng các con số từ dữ liệu "
                "trên (KHÔNG được làm tròn). KHÔNG được nói tool offline. "
                "KHÔNG được hỏi lại người dùng câu nào nữa — trả lời thẳng."
            )
        else:
            sys_text = (
                f"Today: {today_str}. You are Solar AI for the PV Lakehouse. "
                "Answer concisely, accurately, NEVER make up numbers."
            )
            user_text = (
                f"User question:\n{user_message}\n\n"
                f"I ran the query and got this data ({len(rows)} rows, "
                f"first {min(10, len(rows))} shown):\n```json\n{rendered}\n```\n\n"
                f"Draft answer:\n---\n{draft}\n---\n\n"
                "→ Reply to the user in 2-3 sentences, using the numbers above "
                "verbatim (DO NOT round). DO NOT say tools are offline. "
                "DO NOT ask the user any clarifying questions — answer directly."
            )

        clean_messages = [
            {"role": "system", "parts": [{"text": sys_text}]},
            {"role": "user", "parts": [{"text": user_text}]},
        ]
        try:
            synth = self._router.generate_with_tools(
                messages=clean_messages, tools=[],
            )
            text = _strip_inline_cot(
                _strip_reasoning(synth.text or ""), language,
            )
            model_used = synth.model_used
            fb_used = synth.fallback_used or fallback_used
        except Exception as exc:  # noqa: BLE001
            logger.warning("engine_fresh_synthesis_failed err=%s", exc)
            return _strip_draft_instruction_suffix(draft), fallback_model, fallback_used or True

        # Hedge / refusal / clarifying-question detection. If the model still
        # ducks despite the clean context, fall back to the deterministic
        # draft so the user gets a usable answer.
        if _is_hedging_response(text, rows):
            logger.warning(
                "engine_synthesis_hedging — falling back to deterministic draft. "
                "model_text=%r", text[:200],
            )
            return _strip_draft_instruction_suffix(draft), model_used, True
        return text, model_used, fb_used

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

def _build_synthesis_directive(
    language: str, last_sql_result: dict[str, Any] | None,
) -> str:
    """Build the forced-synthesis prompt. Crucially includes the data we
    already pulled from execute_sql so weak models can't claim the tools
    are offline — they have the numbers right in front of them."""
    base_vi = (
        "STOP. Không gọi thêm tool nào nữa. Trả lời người dùng NGAY bằng "
        "văn bản tiếng Việt.\n\n"
        "QUY TẮC TUYỆT ĐỐI:\n"
        "1. CHỈ dùng các con số có trong khối 'DỮ LIỆU' bên dưới. KHÔNG "
        "được bịa ra số khác.\n"
        "2. KHÔNG được làm tròn hoặc thay đổi số liệu.\n"
        "3. KHÔNG được nói tool bị lỗi / offline / không dùng được — "
        "dữ liệu đã có sẵn ngay đây.\n"
        "4. Nếu một con số không có trong khối DỮ LIỆU (ví dụ CO2 avoided), "
        "KHÔNG được nhắc tới nó."
    )
    base_en = (
        "STOP. Do NOT call any more tools. Answer the user NOW in plain text.\n\n"
        "ABSOLUTE RULES:\n"
        "1. ONLY use numbers from the DATA block below. DO NOT make up "
        "different numbers.\n"
        "2. DO NOT round or alter the values.\n"
        "3. DO NOT say tools are offline / failed / unavailable — the "
        "data is right here.\n"
        "4. If a metric isn't in the DATA block (e.g. CO2 avoided), "
        "DO NOT mention it."
    )
    base = base_vi if language == "vi" else base_en
    if not last_sql_result:
        no_data_vi = (
            "\n\nDỮ LIỆU: (chưa có truy vấn nào trả về dữ liệu)\n"
            "→ Trả lời người dùng rằng bạn chưa lấy được dữ liệu, gợi ý "
            "câu hỏi cụ thể hơn (vd: chỉ định nhà máy, khoảng thời gian)."
        )
        no_data_en = (
            "\n\nDATA: (no successful query yet)\n"
            "→ Tell the user you couldn't pull the data; suggest a more "
            "specific question (e.g. name a facility, a time window)."
        )
        return base + (no_data_vi if language == "vi" else no_data_en)
    rows = last_sql_result.get("rows") or []
    if not rows:
        return base
    sample = rows[: min(10, len(rows))]
    try:
        rendered = json.dumps(sample, ensure_ascii=False, default=str, indent=2)
    except Exception:  # noqa: BLE001
        rendered = str(sample)

    # Deterministic draft that the model is told to use verbatim. This kills
    # hallucination — even gpt-5-mini routinely refuses ("I can't run the
    # query") despite explicit data in context. Pre-rendering an answer
    # template removes the refusal degree-of-freedom entirely.
    draft = _render_answer_draft(rows, language)

    suffix_vi = (
        "\n\nDỮ LIỆU "
        f"({len(rows)} dòng từ truy vấn execute_sql gần nhất, "
        f"hiển thị {min(10, len(rows))} dòng đầu):\n```json\n{rendered}\n```\n\n"
        "BẢN NHÁP TRẢ LỜI (dùng nguyên văn hoặc viết lại sát với nó — "
        "KHÔNG được thay đổi con số):\n"
        f"---\n{draft}\n---\n\n"
        "Trả lời người dùng NGAY bây giờ bằng nội dung trên (có thể paraphrase nhẹ)."
    )
    suffix_en = (
        "\n\nDATA "
        f"({len(rows)} rows from the latest execute_sql, "
        f"first {min(10, len(rows))} shown):\n```json\n{rendered}\n```\n\n"
        "DRAFT ANSWER (use verbatim or lightly paraphrase — DO NOT change "
        "any numbers):\n"
        f"---\n{draft}\n---\n\n"
        "Reply to the user NOW with the content above (light paraphrase OK)."
    )
    return base + (suffix_vi if language == "vi" else suffix_en)


_PERCENT_HINTS = ("_pct", "_pcent", "percent", "share", "ratio", "rate", "completeness")
_COUNT_HINTS = ("count", "_n", "rows", "events", "rank")


def _is_percent_or_avg_metric(col: str) -> bool:
    """True if SUMming this column would be meaningless (percentages,
    capacity factors, ratios, ranks, averaged metrics)."""
    low = col.lower()
    if low.startswith("avg_") or low.startswith("mean_") or low.startswith("median_"):
        return True
    if any(h in low for h in _PERCENT_HINTS):
        return True
    return False


def _humanize_col(col: str) -> str:
    """Turn 'avg_performance_ratio_pct' → 'Performance Ratio' — strip
    aggregator prefixes + unit suffixes; the unit is added separately by
    `_unit_for`."""
    s = col
    # Strip leading aggregator prefixes
    for prefix in ("avg_", "mean_", "median_", "total_", "sum_", "max_", "min_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Strip trailing unit suffixes (unit added back by _unit_for)
    for suffix in ("_pct", "_mwh", "_kwh", "_kw", "_mw", "_ms", "_m_s",
                   "_c", "_pcent", "_pp"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.replace("_", " ").strip()
    if not s:
        return col
    # Title-case each word
    return " ".join(w.capitalize() for w in s.split())


def _unit_for(col: str) -> str:
    low = col.lower()
    if low.endswith("_pct") or "percent" in low or "ratio" in low or "share" in low:
        return "%"
    if "mwh" in low:
        return " MWh"
    if "kwh" in low:
        return " kWh"
    if low.endswith("_ms") or "wind_speed" in low:
        return " m/s"
    if "temperature" in low or low.endswith("_c"):
        return "°C"
    return ""


def _render_answer_draft(rows: list[dict[str, Any]], language: str) -> str:
    """Generate a deterministic markdown answer template from query rows.

    Single-row case → bold KPI bullet list.
    Multi-row case  → markdown leaderboard (top 5) + headline narrative.

    Output is markdown so the frontend's `marked.parse()` renders bold,
    bullets, and section headers properly. Skips meaningless aggregations
    (e.g. SUM of percentage columns).
    """
    if not rows:
        return ""

    is_vi = language == "vi"

    if len(rows) == 1:
        row = rows[0]
        head = "**Tóm tắt kết quả**" if is_vi else "**Summary of results**"
        bullets = []
        for k, v in row.items():
            label = _humanize_col(k)
            unit = _unit_for(k)
            bullets.append(f"- **{label}:** {_format_value(v)}{unit}")
        return head + "\n" + "\n".join(bullets)

    # Multi-row case
    cols = list(rows[0].keys())
    numeric_cols = [
        c for c in cols
        if any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
               for r in rows)
    ]
    label_col = next(
        (c for c in cols if isinstance(rows[0].get(c), str) and c not in numeric_cols),
        None,
    )
    date_col = next(
        (c for c in cols
         if "date" in c.lower() or "_at" in c.lower() or c.lower().startswith("ts")),
        None,
    )
    primary = numeric_cols[0] if numeric_cols else None

    parts: list[str] = []

    # Headline (H3)
    if is_vi:
        parts.append(f"### Tổng quan ({len(rows):,} bản ghi)")
    else:
        parts.append(f"### Overview ({len(rows):,} records)")

    # Headline stats (skip SUM if percentage / ratio / avg)
    if primary:
        vals = [r[primary] for r in rows if isinstance(r.get(primary), (int, float))]
        if vals:
            avg = sum(vals) / len(vals)
            metric_name = _humanize_col(primary)
            unit = _unit_for(primary)
            stat_lines: list[str] = []
            if not _is_percent_or_avg_metric(primary):
                total = sum(vals)
                if is_vi:
                    stat_lines.append(f"- **Tổng {metric_name}:** {_format_value(total)}{unit}")
                    stat_lines.append(f"- **Trung bình:** {_format_value(avg)}{unit}")
                else:
                    stat_lines.append(f"- **Total {metric_name}:** {_format_value(total)}{unit}")
                    stat_lines.append(f"- **Average:** {_format_value(avg)}{unit}")
            else:
                if is_vi:
                    stat_lines.append(f"- **Trung bình {metric_name}:** {_format_value(avg)}{unit}")
                else:
                    stat_lines.append(f"- **Average {metric_name}:** {_format_value(avg)}{unit}")

            top_row = max(rows, key=lambda r: r.get(primary) if isinstance(r.get(primary), (int, float)) else float("-inf"))
            bot_row = min(rows, key=lambda r: r.get(primary) if isinstance(r.get(primary), (int, float)) else float("inf"))
            top_label = top_row.get(label_col) if label_col else None
            bot_label = bot_row.get(label_col) if label_col else None
            top_v = top_row.get(primary)
            bot_v = bot_row.get(primary)
            if (top_label and bot_label and top_label != bot_label
                    and isinstance(top_v, (int, float)) and isinstance(bot_v, (int, float))):
                if is_vi:
                    stat_lines.append(f"- **Cao nhất:** {top_label} ({_format_value(top_v)}{unit})")
                    stat_lines.append(f"- **Thấp nhất:** {bot_label} ({_format_value(bot_v)}{unit})")
                else:
                    stat_lines.append(f"- **Highest:** {top_label} ({_format_value(top_v)}{unit})")
                    stat_lines.append(f"- **Lowest:** {bot_label} ({_format_value(bot_v)}{unit})")
            parts.append("\n".join(stat_lines))

    # Date range (if any)
    if date_col:
        date_vals = [str(r.get(date_col)) for r in rows if r.get(date_col) is not None]
        if date_vals:
            uniq_sorted = sorted(set(date_vals))
            tag = "Khoảng thời gian" if is_vi else "Time range"
            parts.append(f"- **{tag}:** {uniq_sorted[0]} → {uniq_sorted[-1]}")

    # Leaderboard (top 5 by primary numeric, when there's a label column)
    if primary and label_col and len(rows) > 1:
        try:
            sorted_rows = sorted(
                rows,
                key=lambda r: r.get(primary) if isinstance(r.get(primary), (int, float)) else float("-inf"),
                reverse=True,
            )
        except Exception:  # noqa: BLE001
            sorted_rows = list(rows)
        top_n = sorted_rows[: min(5, len(sorted_rows))]
        unit = _unit_for(primary)
        head = f"### Top {len(top_n)} {'theo' if is_vi else 'by'} {_humanize_col(primary)}"
        rank_lines = [head]
        for i, r in enumerate(top_n, 1):
            name = r.get(label_col) or "?"
            v = r.get(primary)
            rank_lines.append(
                f"{i}. **{name}** — {_format_value(v)}{unit}"
            )
        parts.append("\n".join(rank_lines))

    # Footer pointer
    if is_vi:
        parts.append("_Bảng chi tiết và biểu đồ hiển thị bên dưới._")
    else:
        parts.append("_Full table and chart shown below._")
    return "\n\n".join(parts)


_DRAFT_INSTRUCTION_SUFFIXES = (
    "(Full table is shown below — DO NOT enumerate each row in the reply.)",
    "(Bảng chi tiết hiển thị bên dưới — KHÔNG cần liệt kê từng dòng trong câu trả lời.)",
)


# Reasoning-model chain-of-thought wrappers. Minimax M2.7, DeepSeek-R1,
# Qwen-QwQ, gpt-oss-thinking and friends emit a `<think>...</think>` block
# (or sometimes `<thinking>...`) with the model's internal monologue ahead
# of the actual answer. We strip them everywhere LLM text becomes
# user-facing — leaving them in leaks 5+ paragraphs of "The user wants me
# to..." into the chat UI and also breaks the hedge detector (which
# scans for cited row numbers and won't find them inside <think>).
_REASONING_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning|analysis|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Unclosed <think> at the START of the response — common when the model
# truncates or forgets the closing tag. Drop everything from the open
# tag through the next blank-line boundary or end of string.
_UNCLOSED_THINK_PREFIX_RE = re.compile(
    r"^\s*<\s*(think|thinking|reasoning|analysis|scratchpad)\s*>.*?(?:\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` (and friends) plus any leftover
    standalone open/close tags. Safe on text that has no CoT — just returns
    the input trimmed."""
    if not text:
        return text
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_PREFIX_RE.sub("", cleaned)
    # Stragglers: lone open/close tags after a malformed wrapper.
    cleaned = re.sub(
        r"</?\s*(think|thinking|reasoning|analysis|scratchpad)\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


# Inline-CoT prose detector. Minimax M2.7 (and other reasoning models)
# sometimes emit chain-of-thought as plain English paragraphs ahead of
# the actual answer — no <think> wrapper at all. Telltale openers:
_INLINE_COT_OPENERS = (
    "the user is",       "the user wants",     "the user asked",
    "the user provided", "the user requests",  "the user says",
    "let me",            "let's ",
    "i need to",         "i should",           "i'll ",          "i will ",
    "i'm noticing",      "i notice",           "i can see",
    "looking at",        "looking again",
    "analyzing the",     "to answer",          "first, ",
    "i can analyze",     "we have ",           "based on the data",
    "the data ",         "from the data",      "given the data",
)
# Vietnamese diacritics range — used to detect when a paragraph belongs to
# the user's language vs. inline English meta-prose.
_VIETNAMESE_CHARS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵĂÂĐÊÔƠƯ]")


def _strip_inline_cot(text: str, language: str) -> str:
    """Drop leading meta-prose paragraphs that look like English chain-of-
    thought when the user asked in Vietnamese (Minimax pattern). Only
    triggers when text leads with English meta phrases AND a later
    paragraph contains Vietnamese diacritics — guarantees we keep at least
    one user-facing paragraph."""
    if not text or language != "vi":
        return text
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < 2:
        return text
    has_vi_paragraph = any(_VIETNAMESE_CHARS.search(p) for p in paragraphs)
    if not has_vi_paragraph:
        return text  # answer is genuinely English — leave alone
    keep_from = 0
    for i, p in enumerate(paragraphs):
        stripped_low = p.strip().lower()
        if not stripped_low:
            continue
        # If this paragraph looks like English CoT meta-prose AND has no
        # Vietnamese chars, mark for removal.
        looks_meta = any(stripped_low.startswith(op) for op in _INLINE_COT_OPENERS)
        no_vi = not _VIETNAMESE_CHARS.search(p)
        if looks_meta and no_vi:
            keep_from = i + 1
        else:
            break
    if keep_from == 0:
        return text
    cleaned = "\n\n".join(paragraphs[keep_from:]).strip()
    return cleaned or text  # never strip everything — fall back to original


# Pattern for "no data for X" refusals. When the answer matches this AND
# the dataset doesn't actually contain the asked-about column, we must
# suppress the (irrelevant) chart + data_table so the user doesn't see
# wind_speed numbers when they asked about humidity.
_NO_DATA_PHRASES = (
    "không có dữ liệu", "khong co du lieu",
    "không chứa", "khong chua",
    "không có chỉ số", "khong co chi so",
    "không có cột", "khong co cot",
    "không có trường", "khong co truong",
    "không thể phân tích", "khong the phan tich",
    "doesn't contain", "does not contain",
    "no data for", "no humidity data", "no humidity field",
    "data doesn't include", "data does not include",
    "missing humidity", "humidity is null",
    "no data is available",
)


def _answer_signals_missing_column(answer: str) -> bool:
    """True iff the model answer explicitly says 'no data / not in dataset'."""
    if not answer:
        return False
    low = answer.lower()
    return any(p in low for p in _NO_DATA_PHRASES)


def _strip_draft_instruction_suffix(text: str) -> str:
    """When the deterministic draft is used as the FINAL answer, strip
    the trailing instruction-to-LLM line so the user sees a clean
    markdown summary instead of the meta-instruction. The new draft
    format uses an italic footer line (`_Full table and chart shown below._`)
    which is user-facing and should stay."""
    cleaned = text
    for suf in _DRAFT_INSTRUCTION_SUFFIXES:
        if suf in cleaned:
            cleaned = cleaned.replace(suf, "").rstrip()
    return cleaned


def _format_value(v: Any) -> str:
    """Format a single cell with sensible precision (2 dp for floats)."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        # Avoid 33.68988439204546 — round to 2 dp; large numbers stay readable
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


_HEDGE_PHRASES = (
    "i don't have", "i dont have", "i can't fetch", "i cant fetch",
    "i can't run", "i cant run", "couldn't run", "tools are offline",
    "tool is offline", "i'll fetch", "i will fetch", "would you like",
    "tell me which option", "pick one", "choose an option",
    "i can produce", "i've prepared", "ive prepared", "i have prepared",
    "let me know which", "i don't yet have",
    # Vietnamese
    "tôi chưa có", "toi chua co", "tôi sẽ", "toi se", "bạn muốn",
    "ban muon", "chọn một", "chon mot", "tôi đã chuẩn bị",
)


def _is_hedging_response(text: str, rows: list[dict[str, Any]]) -> bool:
    """Detect refusal/menu/clarifying-question responses that ignore the
    data we already retrieved. Heuristic: presence of a hedge phrase OR
    none of the row's numeric values appear in the answer text."""
    if not text:
        return True
    low = text.lower()
    if any(p in low for p in _HEDGE_PHRASES):
        return True
    if not rows:
        return False
    # Check at least one numeric value from the data appears in the text.
    # Look for ints, floats with sensible precision, and integer parts.
    found_any = False
    for r in rows[:10]:
        for v in r.values():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                int_part = str(int(v))
                if len(int_part) >= 2 and int_part in text:
                    found_any = True
                    break
                # Also check formatted version with comma separator
                if abs(v) >= 1000 and f"{int(v):,}" in text:
                    found_any = True
                    break
        if found_any:
            break
    return not found_any


def _auto_chart_from_rows(
    rows: list[dict[str, Any]], *,
    title: str = "Visualization",
    intent_hint: str | None = None,
) -> dict[str, Any] | None:
    """Build a sensible Vega-Lite chart from rows when the LLM didn't call
    render_visualization. Heuristic mark selection (in priority order):
      • intent_hint='share'        → pie/arc chart (mark='arc')
      • intent_hint='correlation'  → scatter (mark='point')
      • lat+lng columns            → geographic circle map (mark='circle')
      • date/timestamp x-axis      → line chart (mark='line')
      • share_pct / percent column → arc chart (auto-detected)
      • categorical x-axis         → bar chart (mark='bar')
    Returns the same {format, spec, title, row_count} shape that
    primitives.render_visualization produces, so the chat_service bridge
    treats it identically.
    """
    if not rows:
        return None
    safe_rows = [_to_json_safe(r) for r in rows[:500]]
    cols = list(safe_rows[0].keys())

    def _is_numeric(col: str) -> bool:
        return any(
            isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool)
            for r in safe_rows
        )

    numeric_cols_all = [
        c for c in cols
        if any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
               for r in safe_rows)
    ]

    # Share / contribution intent → pie chart. Picks share_pct column if
    # present, else the first numeric column. Categorical color = first
    # string column (facility_name typically).
    share_col = next(
        (c for c in cols if c.lower() in ("share_pct", "share", "percent",
                                          "percentage", "contribution_pct")),
        None,
    )
    if intent_hint == "share" or (share_col and len(safe_rows) <= 12):
        category_col = next((c for c in cols if isinstance(safe_rows[0].get(c), str)), None)
        theta_col = share_col or (numeric_cols_all[0] if numeric_cols_all else None)
        if category_col and theta_col:
            # Donut chart with hover-highlight + percentage labels overlay.
            spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": safe_rows},
                "height": 320,
                "width": "container",
                "params": [{
                    "name": "highlight",
                    "select": {"type": "point", "on": "mouseover", "fields": [category_col]},
                    "value": [],
                }],
                "layer": [
                    {
                        "mark": {"type": "arc", "innerRadius": 60, "outerRadius": 130,
                                 "stroke": "white", "strokeWidth": 2,
                                 "tooltip": True, "cursor": "pointer"},
                        "encoding": {
                            "theta": {"field": theta_col, "type": "quantitative", "stack": True},
                            "color": {"field": category_col, "type": "nominal",
                                      "scale": {"scheme": "tableau20"},
                                      "legend": {"title": category_col.replace("_", " ").title()}},
                            "opacity": {"condition": {"param": "highlight", "value": 1.0,
                                                       "empty": True},
                                        "value": 0.55},
                            "tooltip": [
                                {"field": category_col, "title": category_col.replace("_", " ").title()},
                                {"field": theta_col, "type": "quantitative",
                                 "format": ",.2f", "title": theta_col.replace("_", " ").title()},
                            ],
                        },
                    },
                    {
                        "mark": {"type": "text", "radius": 155, "fontWeight": "bold", "fontSize": 11},
                        "encoding": {
                            "theta": {"field": theta_col, "type": "quantitative", "stack": True},
                            "text": {"field": category_col, "type": "nominal"},
                            "color": {"field": category_col, "type": "nominal",
                                      "scale": {"scheme": "tableau20"}, "legend": None},
                        },
                    },
                ],
            }
            return {"format": "vega-lite", "spec": spec, "title": title,
                    "row_count": len(safe_rows), "mark": "arc"}

    # Correlation intent → scatter with hover, regression line, and labels.
    if intent_hint == "correlation" and len(numeric_cols_all) >= 2:
        x_num, y_num = numeric_cols_all[0], numeric_cols_all[1]
        label_col = next((c for c in cols if isinstance(safe_rows[0].get(c), str)), None)
        layers: list[dict[str, Any]] = []
        point_mark = {"type": "point", "size": 140, "filled": True,
                      "tooltip": True, "cursor": "pointer"}
        point_encoding = {
            "x": {"field": x_num, "type": "quantitative",
                  "title": x_num.replace("_", " ").title(),
                  "scale": {"zero": False}},
            "y": {"field": y_num, "type": "quantitative",
                  "title": y_num.replace("_", " ").title(),
                  "scale": {"zero": False}},
        }
        if label_col:
            point_encoding["tooltip"] = [
                {"field": label_col, "title": label_col.replace("_", " ").title()},
                {"field": x_num, "format": ",.3f"},
                {"field": y_num, "format": ",.3f"},
            ]
            point_encoding["color"] = {"field": label_col, "type": "nominal",
                                        "scale": {"scheme": "tableau20"}, "legend": None}
        layers.append({"mark": point_mark, "encoding": point_encoding})
        # Linear regression line for visible correlation cue.
        layers.append({
            "mark": {"type": "line", "color": "#888", "strokeDash": [4, 3], "opacity": 0.7},
            "transform": [{"regression": y_num, "on": x_num}],
            "encoding": {
                "x": {"field": x_num, "type": "quantitative"},
                "y": {"field": y_num, "type": "quantitative"},
            },
        })
        if label_col and len(safe_rows) <= 16:
            layers.append({
                "mark": {"type": "text", "dx": 7, "dy": -7, "fontSize": 10, "color": "#333"},
                "encoding": {
                    "x": {"field": x_num, "type": "quantitative"},
                    "y": {"field": y_num, "type": "quantitative"},
                    "text": {"field": label_col, "type": "nominal"},
                },
            })
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": safe_rows},
            "height": 360,
            "width": "container",
            "layer": layers,
        }
        return {"format": "vega-lite", "spec": spec, "title": title,
                "row_count": len(safe_rows), "mark": "point"}

    # Geographic map — emit a Leaflet payload (tile basemap + circle markers)
    # instead of a Vega-Lite geoshape. Vega-embed doesn't natively support
    # pan/zoom for `projection`-based maps; Leaflet does, and ships with
    # native scroll-zoom, click-drag pan, and OSM tile attribution.
    lat_col = next((c for c in cols if c.lower() in ("latitude", "location_lat", "lat")), None)
    lng_col = next((c for c in cols if c.lower() in ("longitude", "location_lng", "lng", "lon")), None)
    if lat_col and lng_col:
        size_col = next((c for c in cols if "capacity" in c.lower() and _is_numeric(c)), None)
        label_col = next(
            (c for c in cols if c.lower() in ("facility_name", "name")),
            next((c for c in cols if isinstance(safe_rows[0].get(c), str)), None),
        )
        points: list[dict[str, Any]] = []
        for r in safe_rows:
            la = r.get(lat_col)
            ln = r.get(lng_col)
            if not isinstance(la, (int, float)) or not isinstance(ln, (int, float)):
                continue
            point = {"lat": float(la), "lng": float(ln)}
            if label_col and r.get(label_col) is not None:
                point["label"] = str(r[label_col])
            if size_col and isinstance(r.get(size_col), (int, float)):
                point["size_value"] = float(r[size_col])
            # Pass-through extra attrs for the popup
            extras: dict[str, Any] = {}
            for c in cols:
                if c in (lat_col, lng_col):
                    continue
                v = r.get(c)
                if isinstance(v, (str, int, float, bool)) or v is None:
                    extras[c] = v
            point["attrs"] = extras
            points.append(point)
        if points:
            return {
                "format": "leaflet-map",
                "title": title,
                "row_count": len(points),
                "mark": "map",
                "points": points,
                "size_field": size_col,
                "label_field": label_col,
            }

    # Time-series line chart — point markers + tooltip + zoom/pan interactions.
    date_col = next(
        (c for c in cols if "date" in c.lower() or "_at" in c.lower() or "_hour" in c.lower()),
        None,
    )
    numeric_cols = [c for c in cols if _is_numeric(c)]
    if not numeric_cols:
        return None
    y_col = numeric_cols[0]
    if date_col:
        color_col = next((c for c in cols if isinstance(safe_rows[0].get(c), str) and c != date_col), None)
        encoding = {
            "x": {"field": date_col, "type": "temporal",
                  "title": date_col.replace("_", " ").title()},
            "y": {"field": y_col, "type": "quantitative",
                  "title": y_col.replace("_", " ").title()},
            "tooltip": [
                {"field": date_col, "type": "temporal", "title": "Time"},
                {"field": y_col, "format": ",.2f", "title": y_col.replace("_", " ").title()},
            ],
        }
        if color_col:
            encoding["color"] = {"field": color_col, "type": "nominal",
                                  "scale": {"scheme": "tableau20"}}
            encoding["tooltip"].insert(0, {"field": color_col})
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": safe_rows},
            "height": 340,
            "width": "container",
            "mark": {"type": "line", "point": {"filled": True, "size": 50},
                     "tooltip": True, "cursor": "pointer", "interpolate": "monotone"},
            "encoding": encoding,
            "params": [{"name": "grid", "select": "interval", "bind": "scales"}],
        }
        return {"format": "vega-lite", "spec": spec, "title": title,
                "row_count": len(safe_rows), "mark": "line"}

    # Categorical x-axis fallback → bar with hover-highlight + tooltip.
    x_col = next((c for c in cols if isinstance(safe_rows[0].get(c), str)), cols[0])
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": safe_rows},
        "height": 340,
        "width": "container",
        "params": [{
            "name": "highlight",
            "select": {"type": "point", "on": "mouseover", "fields": [x_col]},
            "value": [],
        }],
        "mark": {"type": "bar", "tooltip": True, "cursor": "pointer",
                 "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
        "encoding": {
            "x": {"field": x_col, "type": "nominal", "sort": "-y",
                  "title": x_col.replace("_", " ").title()},
            "y": {"field": y_col, "type": "quantitative",
                  "title": y_col.replace("_", " ").title()},
            "color": {"field": x_col, "type": "nominal",
                       "scale": {"scheme": "tableau20"}, "legend": None},
            "opacity": {"condition": {"param": "highlight", "value": 1.0, "empty": True},
                        "value": 0.55},
            "tooltip": [
                {"field": x_col, "title": x_col.replace("_", " ").title()},
                {"field": y_col, "format": ",.2f", "title": y_col.replace("_", " ").title()},
            ],
        },
    }
    return {"format": "vega-lite", "spec": spec, "title": title,
            "row_count": len(safe_rows), "mark": "bar"}


_OFF_TOPIC_PATTERNS = (
    # Geography / general knowledge — only triggers when paired with NO
    # in-domain words (energy, facility, solar, weather, etc.)
    "weather forecast for", "weather in",
    "stock price", "stock market", "bitcoin", "crypto",
    "recipe", "cooking", "movie", "film",
    "capital of", "population of",
    "translate ", "convert currency",
    # Code-help / general programming
    "write a function", "write a python", "write python code",
    "viết function", "viet function", "viết hàm", "viet ham",
    "viết code", "viet code", "code python", "code java",
    "sort an array", "sort mảng", "sort mang", "sort the array",
    # Sport / culture / entertainment
    "world cup", "olympics", "football", "premier league",
    "tesla stock", "apple stock",
    # Generic math expressions outside lakehouse domain
    "1 + 1", "1+1", "2 + 2", "2+2", "căn bậc", "can bac",
    # Vietnamese
    "thời tiết ở", "thoi tiet o", "tỷ giá", "ty gia",
    "công thức nấu", "cong thuc nau",
    "phở bò", "pho bo", "công thức nấu", "cong thuc nau",
    "bằng mấy", "bang may",
)
_IN_DOMAIN_KEYWORDS = (
    # Domain-distinctive only — avoid generic words like "weather" or
    # "forecast" that overlap with off-topic patterns ("weather forecast
    # for New York"). For weather/forecast we require pairing with a
    # facility / solar / grid term to count as in-domain.
    "energy", "facility", "facilities", "solar", "pv", "panel",
    "capacity factor", "mwh", "kwh", "irradiance",
    "aqi", "lakehouse", "databricks",
    "capacity mw", "capacity_mw",
    "ingestion", "ingestion job", "etl",
    "champion model", "ml model", "ml champion",
    "kpi", "fleet", "production mwh", "generation mwh",
    "rooftop", "photovoltaic",
    # Vietnamese (distinctive)
    "trạm", "tram", "sản lượng", "san luong", "công suất", "cong suat",
    "mặt trời", "mat troi", "điện mặt trời", "dien mat troi",
    "lakehouse", "databricks", "tổng quan", "tong quan",
    "hệ thống pv", "he thong pv",
)


_CONCEPTUAL_PATTERNS = (
    # English definitional / explainer triggers
    r"\bwhat\s+is\b",
    r"\bwhat\s+does\b",
    r"\bwhat\s+are\b",
    r"\bdefine\b",
    r"\bdefinition\s+of\b",
    r"\bexplain\s+(?:what|how|why)\b",
    r"\bhow\s+(?:do|does|is)\b.*\b(?:work|defined|calculated|computed|measured)\b",
    r"\bhow\s+to\s+(?:compute|calculate|measure)\b",
    r"\bformula\s+for\b",
    r"\bmeaning\s+of\b",
    # Vietnamese definitional triggers
    r"\blà\s+gì\b",
    r"\bla\s+gi\b",
    r"\bnghĩa\s+là\b",
    r"\bnghia\s+la\b",
    r"\bđịnh\s+nghĩa\b",
    r"\bdinh\s+nghia\b",
    r"\bcông\s+thức\b",
    r"\bcong\s+thuc\b",
    r"\bgiải\s+thích\b",
    r"\bgiai\s+thich\b",
    r"\bý\s+nghĩa\b",
    r"\by\s+nghia\b",
    r"\bcách\s+(?:tính|đo)\b",
    r"\bcach\s+(?:tinh|do)\b",
    # "đo cái gì" / "đo gì" / "đại lượng đo" — measurement-definition queries
    r"\bđo\s+(?:cái\s+)?gì\b",
    r"\bdo\s+(?:cai\s+)?gi\b",
    r"\bđại\s+lượng\b",
    r"\bdai\s+luong\b",
    # NOTE: "so sánh" / "difference between" / "khác nhau" patterns were
    # REMOVED — they fire on data-driven comparison queries
    # ("so sánh DARLSF và AVLSF về sản lượng") and force the conceptual
    # path which never runs SQL. Definitional comparisons still get caught
    # by the trailing "là gì" / "is" / "what is" patterns above.
)
_CONCEPTUAL_RE = re.compile("|".join(_CONCEPTUAL_PATTERNS), re.IGNORECASE)


def _is_conceptual_question(message: str) -> bool:
    """True for definitional / explainer questions that don't need data.

    Triggers ONLY when the question uses a textbook-style trigger phrase.
    A query like "show me the capacity factor of AVLSF" is NOT conceptual
    even though it mentions a metric name; "What is capacity factor" IS.

    Conservative on purpose — false positives here cost more (no data
    shown) than false negatives (model still answers via SQL path).
    """
    if not message:
        return False
    return bool(_CONCEPTUAL_RE.search(message))


def _is_off_topic(message: str) -> bool:
    if not message:
        return False
    low = message.lower()
    has_off = any(p in low for p in _OFF_TOPIC_PATTERNS)
    if not has_off:
        return False
    # If the message ALSO contains in-domain keywords, treat it as on-topic
    # (e.g. "weather in Sydney" is off, "weather impact on Sydney solar
    # facility" is on).
    has_in = any(k in low for k in _IN_DOMAIN_KEYWORDS)
    return not has_in


def _build_scope_redirect(language: str) -> str:
    if language == "vi":
        return (
            "Mình chỉ trả lời các câu hỏi liên quan đến hệ thống PV "
            "Lakehouse — sản lượng điện mặt trời, hiệu suất trạm, "
            "thời tiết tại các trạm, AQI, dự báo, mô hình ML và "
            "trạng thái pipeline. Bạn thử hỏi mình về một trong các "
            "chủ đề trên nhé. Ví dụ: \"Tổng quan hệ thống 7 ngày qua\", "
            "\"Top 5 trạm sản lượng cao nhất\", \"Dự báo sản lượng 7 ngày tới\"."
        )
    return (
        "I only answer questions about the PV Lakehouse — solar energy "
        "production, facility performance, weather at facilities, AQI, "
        "forecasts, ML models, and pipeline status. Try one of those "
        "instead, e.g. \"system overview last 7 days\", \"top 5 facilities "
        "by energy\", or \"forecast next 7 days\"."
    )


def _detect_language(message: str) -> str:
    """Quick VN/EN detector for engine routing.

    Heuristic: presence of Vietnamese-specific diacritics or known VN
    function-word tokens → 'vi'. Otherwise → 'en'. Avoids the
    query_rewriter mis-classifying English queries as Vietnamese when
    they happen to share a few token shapes (observed in production)."""
    if not message:
        return "en"
    # Vietnamese-only diacritic ranges (combining vowels with tone marks)
    vi_diacritics = "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴĂÂĐÊÔƠƯ"
    if any(ch in vi_diacritics for ch in message):
        return "vi"
    low = " " + message.lower() + " "
    vi_tokens = (
        " hệ ", " thống ", " trạm ", " ngày ", " tổng ", " sản ", " lượng ",
        " bao cao ", " cho toi ", " hien tai ", " tinh hinh ", " hieu suat ",
        " du bao ", " tong quan ", " chinh xac ",
    )
    if any(t in low for t in vi_tokens):
        return "vi"
    return "en"


_CHART_INTENT_KEYWORDS = (
    # English
    "chart", "graph", "plot", "visualize", "visualise", "visualization",
    "visualisation", "bar chart", "line chart", "map", "scatter",
    "histogram", "trend chart", "show me a chart", "draw",
    # Time-series intent — these queries shape implies line/area chart
    # even when the user doesn't say "chart" explicitly.
    "trend", "forecast", "over time", "by hour", "by day", "by month",
    "correlation", "vs.", " vs ",
    # Vietnamese (with/without diacritics)
    "biểu đồ", "bieu do", "đồ thị", "do thi", "trực quan", "truc quan",
    "vẽ", "ve ", "bản đồ", "ban do", "biểu diễn", "bieu dien",
    "dự báo", "du bao", "xu hướng", "xu huong",
    "theo giờ", "theo gio", "theo ngày", "theo ngay",
    "tương quan", "tuong quan", "mối quan hệ", "moi quan he",
)


def _detect_chart_intent(message: str) -> bool:
    """True if the user explicitly asked for a visualization."""
    if not message:
        return False
    low = message.lower()
    return any(kw in low for kw in _CHART_INTENT_KEYWORDS)


def _parse_user_hints(message: str) -> dict[str, Any]:
    """Extract structured parameters the model often forgets to substitute.

    Currently: window_days from "N ngày qua / last N days / past N days /
    N-day window / N day(s)". Returns {} if nothing matched.
    """
    if not message:
        return {}
    import re
    hints: dict[str, Any] = {}
    # Vietnamese: "7 ngày", "trong 14 ngày qua"
    # English:    "last 7 days", "past 14 days", "30-day", "30 day window"
    patterns = [
        r"(\d{1,3})\s*(?:ng[aà]y)\b",
        r"(?:last|past|previous|over\s+the\s+past)\s+(\d{1,3})\s*[- ]?day",
        r"(\d{1,3})[-\s]day(?:s)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            if 1 <= n <= 365:
                hints["window_days"] = n
                break
    return hints


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


def _infer_metric_from_sql(executed_sql: str, recall_top: dict[str, Any] | None) -> str | None:
    """Best-effort: tag a successful SQL with its source recall_metric name.

    Used to detect when last_sql_result is stale (model fetched the wrong
    metric's data, then user intent shifted). Returns the recall_top name
    if the executed SQL primary table matches the template's primary table.
    """
    if not executed_sql or not recall_top:
        return None
    template = str(recall_top.get("sql_template") or "")
    sql_fqn = _extract_table_from_sql(executed_sql)
    tmpl_fqn = _extract_table_from_sql(template)
    if sql_fqn and tmpl_fqn and sql_fqn == tmpl_fqn:
        return str(recall_top.get("name") or "")
    return None


def _extract_table_from_sql(sql: str) -> str:
    """Best-effort: pull the first real table FQN from a SELECT for source
    attribution. Skips FROM clauses that appear inside SQL functions like
    EXTRACT(HOUR FROM ts), CAST(x AS ... FROM ...), etc — the regex used
    to grab those column names instead of tables."""
    if not sql:
        return "lakehouse"
    import re
    # Strip FROM occurrences inside parentheses (function arg position) by
    # scanning paren depth and only accepting matches at depth == 0.
    depth = 0
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            # Match \bFROM\s+<ident> at the current position
            m = re.match(r"\bFROM\s+([a-zA-Z0-9_.]+)", sql[i:], re.IGNORECASE)
            if m:
                fqn = m.group(1)
                # Heuristic: real tables almost always have a dot (catalog.schema.table
                # or schema.table). Plain identifiers are usually CTEs or aliases.
                if "." in fqn:
                    return fqn
                # Keep scanning — maybe an inner FROM precedes the real one.
        i += 1
    # Fallback: original simple regex (catches single-segment table names too)
    m = re.search(r"\bFROM\s+([a-zA-Z0-9_.]+)", sql, re.IGNORECASE)
    return m.group(1) if m else "lakehouse"


def _extract_key_metrics(sql_result: dict[str, Any]) -> dict[str, Any]:
    """Pull a flat metrics dict from a sql result. If single row → that row.
    If multiple rows → first row + row_count. UI uses these for KPI cards.
    All values pass through `_to_json_safe` so date/Decimal don't break
    Postgres JSONB persistence."""
    rows = sql_result.get("rows") or []
    if not rows:
        return {}
    out: dict[str, Any] = {}
    if len(rows) == 1:
        for k, v in rows[0].items():
            safe = _to_json_safe(v)
            if isinstance(safe, (int, float, str, bool)) or safe is None:
                out[k] = safe
    else:
        out["row_count"] = len(rows)
        for k, v in rows[0].items():
            if isinstance(v, (int, float)):
                out[f"first_{k}"] = v
                break
    return out


def _build_data_table(sql_result: dict[str, Any]) -> dict[str, Any] | None:
    """Wrap rows + columns in the shape the v1 DataTable component expects.

    Rows are deep-converted via `_to_json_safe` so date/datetime/Decimal
    values from Databricks survive the JSON encoder downstream (DoneEvent
    serialisation + Postgres JSONB persistence both call json.dumps)."""
    rows = sql_result.get("rows") or []
    cols = sql_result.get("columns") or (list(rows[0].keys()) if rows else [])
    if not rows:
        return None
    safe_rows = [_to_json_safe(r) for r in rows[:200]]
    return {
        "columns": [{"key": c, "label": c} for c in cols],
        "rows": safe_rows,
        "row_count": len(rows),
        "truncated": len(rows) > 200,
    }
