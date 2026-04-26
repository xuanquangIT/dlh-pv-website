"""Tests for V2ChatEngine — the agentic loop that wires LLM tool calls
to v2 primitives.

Uses fake routers / dispatchers so no Databricks or real LLM is needed.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.services.solar_ai_chat.llm_client import (
    LLMToolResult,
    ToolCallRequest,
)
from app.services.solar_ai_chat.v2.dispatcher import V2DispatchResult
from app.services.solar_ai_chat.v2.engine import (
    V2ChatEngine,
    _extract_key_metrics,
    _extract_table_from_sql,
    _layer_from_fqn,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    """Scripted LLM turn — either tool calls OR a text answer."""
    calls: tuple = ()
    text: str | None = None


class FakeRouter:
    def __init__(self, scripted_turns: list[_Turn]):
        self._turns = list(scripted_turns)
        self.calls_seen = []

    def generate_with_tools(self, *, messages, tools, **_):
        self.calls_seen.append({"msg_count": len(messages), "tools_count": len(tools)})
        if not self._turns:
            return LLMToolResult(function_call=None, text="(scripted exhausted)",
                                 model_used="fake", fallback_used=False)
        turn = self._turns.pop(0)
        return LLMToolResult(
            function_call=turn.calls[0] if turn.calls else None,
            function_calls=turn.calls,
            text=turn.text,
            model_used="fake-gpt-4",
            fallback_used=False,
        )


class FakeDispatcher:
    def __init__(self, results_by_fn: dict[str, dict]):
        self._results = results_by_fn
        self.calls = []

    def execute(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        result = self._results.get(function_name, {"error": "not scripted"})
        return V2DispatchResult(
            function_name=function_name, ok="error" not in result,
            result=result, duration_ms=10,
        )


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


class V2EngineTextOnlyTests(unittest.TestCase):
    def test_no_tool_calls_returns_llm_text_directly(self):
        router = FakeRouter([_Turn(text="Xin chào! Tôi là Solar AI.")])
        dispatcher = FakeDispatcher({})
        engine = V2ChatEngine(router, dispatcher)

        result = engine.run(user_message="hello", language="vi")

        self.assertEqual(result.answer, "Xin chào! Tôi là Solar AI.")
        self.assertEqual(result.trace_steps, [])
        self.assertEqual(dispatcher.calls, [])
        self.assertIsNone(result.chart)


class V2EngineToolLoopTests(unittest.TestCase):
    def test_recall_metric_then_execute_sql_then_answer(self):
        router = FakeRouter([
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "top facilities"}),)),
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1 FROM pv.gold.mart_energy_daily"}),)),
            _Turn(text="Trạm A đứng đầu với 250 MWh."),
        ])
        dispatcher = FakeDispatcher({
            "recall_metric": {"matches": [{"name": "top_facilities_by_energy",
                                            "sql_template": "SELECT ..."}]},
            "execute_sql": {
                "rows": [{"facility_name": "A", "total_mwh": 250}],
                "row_count": 1,
                "columns": ["facility_name", "total_mwh"],
                "executed_sql": "SELECT facility_name, total_mwh FROM pv.gold.mart_energy_daily LIMIT 1000",
            },
        })

        engine = V2ChatEngine(router, dispatcher)
        result = engine.run(user_message="top trạm sản lượng", language="vi")

        self.assertEqual(result.answer, "Trạm A đứng đầu với 250 MWh.")
        self.assertEqual(len(result.trace_steps), 2)
        self.assertEqual(result.trace_steps[0]["primitive"], "recall_metric")
        self.assertEqual(result.trace_steps[1]["primitive"], "execute_sql")
        # SQL result populated key_metrics
        self.assertEqual(result.key_metrics["facility_name"], "A")
        self.assertEqual(result.key_metrics["total_mwh"], 250)
        # Source attribution from the SQL FROM clause
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].dataset, "pv.gold.mart_energy_daily")
        self.assertEqual(result.sources[0].layer, "Gold")
        # Data table mirrored from execute_sql rows
        self.assertIsNotNone(result.data_table)
        self.assertEqual(result.data_table["row_count"], 1)

    def test_render_visualization_populates_chart_payload(self):
        router = FakeRouter([
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT * FROM pv.gold.dim_facility"}),)),
            _Turn(calls=(ToolCallRequest(
                name="render_visualization",
                arguments={
                    "spec": {"mark": "geoshape"},
                    "data": [{"latitude": -34.5, "longitude": 146.3}],
                    "title": "Facility map",
                },
            ),)),
            _Turn(text="Đây là bản đồ 8 trạm."),
        ])
        dispatcher = FakeDispatcher({
            "execute_sql": {
                "rows": [{"latitude": -34.5, "longitude": 146.3}],
                "row_count": 1,
                "columns": ["latitude", "longitude"],
                "executed_sql": "SELECT latitude, longitude FROM pv.gold.dim_facility",
            },
            "render_visualization": {
                "format": "vega-lite",
                "spec": {"mark": "geoshape", "data": {"values": [{"latitude": -34.5, "longitude": 146.3}]}},
                "row_count": 1,
                "title": "Facility map",
            },
        })

        engine = V2ChatEngine(router, dispatcher)
        result = engine.run(user_message="map các trạm", language="vi")

        self.assertIsNotNone(result.chart)
        self.assertEqual(result.chart["format"], "vega-lite")
        self.assertEqual(result.chart["spec"]["mark"], "geoshape")
        self.assertEqual(result.chart["title"], "Facility map")

    def test_max_steps_forces_synthesis_turn(self):
        # Three identical tool calls in a row. The engine bans recall_metric
        # on the 2nd (exact_dup), then on the 3rd attempt the only call is to
        # a banned tool → triggers the "all calls banned" branch and forces
        # synthesis. Net dispatches: 2 (step 1 + step 2 dup re-dispatch).
        scripted = [
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),))
            for _ in range(3)
        ]
        scripted.append(_Turn(text="OK gathered enough."))   # synthesis
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({"recall_metric": {"matches": []}})

        engine = V2ChatEngine(router, dispatcher, max_steps=3)
        result = engine.run(user_message="thử test loop", language="en")

        # 2 dispatched calls (step 1 + step 2 re-dispatch on dup detection);
        # step 3 hits the banned-tool branch which forces synthesis. Since
        # the dispatcher returned no SQL rows (recall_metric only), the
        # forced synth uses the no-data fallback message rather than the
        # LLM thread (which would otherwise hedge).
        self.assertEqual(len(result.trace_steps), 2)
        self.assertIn("couldn't retrieve", result.answer.lower())

    def test_banned_tool_call_blocked_at_engine_layer(self):
        """Weak models (gemini-flash-lite) re-call banned tools even after
        we filter the schema. The engine MUST physically block dispatch and
        force synthesis instead of executing the banned call."""
        scripted = [
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),  # dup → ban
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),  # banned → synth
            _Turn(text="Synthesised."),
        ]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({"recall_metric": {"matches": []}})
        engine = V2ChatEngine(router, dispatcher, max_steps=5)

        result = engine.run(user_message="ping", language="en")

        # Dispatcher.calls captures every actual primitive execution.
        # Step 1 = 1 call, step 2 = 1 re-dispatch, step 3 = BLOCKED (no dispatch).
        # Total = 2 dispatcher executions, NOT 3.
        self.assertEqual(len(dispatcher.calls), 2)
        # No SQL rows ever surfaced, so forced synthesis returns the
        # deterministic no-data message instead of consuming the scripted
        # "Synthesised." turn (which would only fire if real data were present).
        self.assertIn("couldn't retrieve", result.answer.lower())

    def test_conceptual_question_skips_tools(self):
        """Definition queries ('Performance Ratio là gì', 'What is X')
        must skip the tool loop entirely — no SQL, no KPIs, no chart."""
        # Even though the router is scripted with a tool call, the engine
        # MUST short-circuit to the conceptual-answer path before the loop
        # starts. So dispatcher.calls stays empty.
        scripted = [_Turn(text="Performance Ratio (PR) là tỷ lệ năng lượng thực tế / lý thuyết.")]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({})
        engine = V2ChatEngine(router, dispatcher)

        result = engine.run(user_message="Performance Ratio là gì", language="vi")

        self.assertEqual(dispatcher.calls, [])  # no tool dispatched
        self.assertIn("Performance Ratio", result.answer)
        self.assertIsNone(result.chart)
        self.assertIsNone(result.data_table)
        self.assertEqual(result.key_metrics, {})

    def test_parallel_call_fanout_capped(self):
        """Some models (grok-4-fast) emit 5-6 tool_calls per turn. The
        engine MUST cap fan-out at MAX_PARALLEL_CALLS_PER_TURN (3) so an
        8-turn budget doesn't burn 30+ primitive dispatches in one go."""
        big_fanout = tuple(
            ToolCallRequest(name="inspect_table", arguments={"table_fqn": f"pv.gold.t{i}"})
            for i in range(6)
        )
        scripted = [_Turn(calls=big_fanout), _Turn(text="done")]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({
            "inspect_table": {"error": "Table not in semantic layer."},
        })
        engine = V2ChatEngine(router, dispatcher, max_steps=3)
        engine.run(user_message="probe", language="en")
        # Only 3 of the 6 fan-out calls should reach the dispatcher.
        self.assertEqual(len(dispatcher.calls), 3)

    def test_generic_placeholder_labels_trigger_hedge(self):
        """Regression: model received real wind data with facility_name
        column but answered 'Cơ sở 1: 15.2 m/s, Cơ sở 2: 14.8 m/s' —
        invented placeholder labels. Hedge detector must catch this and
        force the deterministic draft (which uses real names)."""
        scripted = [
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1 FROM pv.gold.x"}),)),
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1 FROM pv.gold.x"}),)),  # dup
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1 FROM pv.gold.x"}),)),  # banned → synth
            _Turn(text="Top 5: Cơ sở 1: 15.2 m/s, Cơ sở 2: 14.8 m/s, Cơ sở 3: 14.1 m/s."),
        ]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({"execute_sql": {
            "rows": [
                {"facility_name": "AVLSF", "avg_wind_speed_ms": 5.2},
                {"facility_name": "BOMENSF", "avg_wind_speed_ms": 4.8},
            ],
            "columns": ["facility_name", "avg_wind_speed_ms"],
        }})
        engine = V2ChatEngine(router, dispatcher, max_steps=5)
        result = engine.run(user_message="wind impact", language="vi")
        # Hedge detected → draft used. Draft must mention a real name.
        self.assertIn("AVLSF", result.answer)
        self.assertNotIn("Cơ sở 1", result.answer)
        self.assertTrue(result.fallback_used)

    def test_auto_execute_runs_recall_top_when_model_loops(self):
        """Weak models loop on recall_metric without ever calling execute_sql.
        When recall is banned, the engine MUST auto-execute the top match's
        rendered SQL so the user still gets data."""
        # Model calls recall_metric twice (identical → exact_dup), then on
        # turn 3 calls recall_metric again (banned). All_calls_banned fires →
        # _maybe_auto_execute should run the YAML SQL via dispatcher and
        # produce real data + a final synthesised answer.
        scripted = [
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),  # exact dup → ban
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),)),  # banned → synth
            _Turn(text="Total energy 112,134.77 MWh."),
        ]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({
            "recall_metric": {"matches": [{
                "name": "system_overview",
                "sql_template": "SELECT 1 AS total_energy_mwh FROM pv.gold.mart_system_kpi_daily WHERE kpi_date >= CURRENT_DATE - INTERVAL '{window_days}' DAY",
                "parameters": [{"name": "window_days", "default": 30}],
            }]},
            "execute_sql": {
                "rows": [{"total_energy_mwh": 112134.77}],
                "row_count": 1,
                "columns": ["total_energy_mwh"],
                "executed_sql": "SELECT 1 AS total_energy_mwh FROM pv.gold.mart_system_kpi_daily WHERE kpi_date >= CURRENT_DATE - INTERVAL '30' DAY",
            },
        })
        engine = V2ChatEngine(router, dispatcher, max_steps=5)

        result = engine.run(user_message="summarize", language="en")

        # The auto-executor MUST have called execute_sql with the rendered SQL
        execute_calls = [c for c in dispatcher.calls if c[0] == "execute_sql"]
        self.assertEqual(len(execute_calls), 1)
        self.assertIn("INTERVAL '30' DAY", execute_calls[0][1]["sql"])
        # And the engine should now have data → KPI cards populated
        self.assertEqual(result.key_metrics.get("total_energy_mwh"), 112134.77)
        # Sources extracted from the auto-executed SQL
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].dataset, "pv.gold.mart_system_kpi_daily")

    def test_hedging_synthesis_falls_back_to_draft(self):
        """If the LLM ignores data and returns a hedge/menu response, the
        engine falls back to the deterministic draft so the user still
        gets an answer with the actual numbers."""
        scripted = [
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1"}),)),
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1"}),)),  # dup
            _Turn(calls=(ToolCallRequest(name="execute_sql", arguments={"sql": "SELECT 1"}),)),  # banned → synth
            _Turn(text="I've prepared the query and would you like option 1, 2, or 3?"),
        ]
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({"execute_sql": {
            "rows": [{"total_energy_mwh": 112134.77, "facility_count": 8}],
            "columns": ["total_energy_mwh", "facility_count"],
        }})
        engine = V2ChatEngine(router, dispatcher, max_steps=5)

        result = engine.run(user_message="summarize", language="en")

        # Hedging detected → answer becomes the deterministic draft, which
        # MUST quote the actual numbers from the row.
        self.assertIn("112,134.77", result.answer)
        self.assertIn("8", result.answer)
        # And the model_used should reflect the fallback flag was raised.
        self.assertTrue(result.fallback_used)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HelperTests(unittest.TestCase):
    def test_extract_table_from_sql_basic(self):
        self.assertEqual(_extract_table_from_sql("SELECT * FROM pv.gold.dim_facility LIMIT 10"),
                         "pv.gold.dim_facility")
        self.assertEqual(_extract_table_from_sql(""), "lakehouse")

    def test_extract_table_skips_extract_function_from(self):
        """Regression: SELECT EXTRACT(HOUR FROM date_hour_local) AS h
        FROM pv.gold.fact_energy used to extract 'date_hour_local' instead
        of 'pv.gold.fact_energy' because the regex matched the inner FROM."""
        sql = (
            "SELECT EXTRACT(HOUR FROM date_hour_local) AS hour_of_day, "
            "AVG(energy_mwh) AS avg_energy_mwh "
            "FROM pv.gold.fact_energy "
            "WHERE date_hour_local >= CURRENT_TIMESTAMP - INTERVAL '30' DAY "
            "GROUP BY EXTRACT(HOUR FROM date_hour_local) "
            "ORDER BY hour_of_day"
        )
        self.assertEqual(_extract_table_from_sql(sql), "pv.gold.fact_energy")

    def test_extract_table_handles_with_cte(self):
        sql = (
            "WITH recent AS (SELECT * FROM pv.gold.mart_energy_daily WHERE energy_date >= '2026-01-01') "
            "SELECT facility_name, SUM(energy_mwh_daily) FROM recent GROUP BY facility_name"
        )
        # First real FROM with a dot-qualified name should win.
        self.assertEqual(_extract_table_from_sql(sql), "pv.gold.mart_energy_daily")

    def test_layer_from_fqn(self):
        self.assertEqual(_layer_from_fqn("pv.gold.dim_facility"), "Gold")
        self.assertEqual(_layer_from_fqn("pv.silver.energy_readings"), "Silver")
        self.assertEqual(_layer_from_fqn("foo"), "Gold")

    def test_extract_key_metrics_single_row_inlines(self):
        out = _extract_key_metrics({"rows": [{"x": 1, "y": "a"}]})
        self.assertEqual(out, {"x": 1, "y": "a"})

    def test_extract_key_metrics_multi_row_summary(self):
        out = _extract_key_metrics({"rows": [{"x": 1}, {"x": 2}]})
        self.assertEqual(out["row_count"], 2)
        self.assertEqual(out["first_x"], 1)


if __name__ == "__main__":
    unittest.main()
