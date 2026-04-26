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
        # 3 tool turns (matching max_steps) → loop exhausts → engine adds a
        # forced synthesis turn that should consume the next scripted reply.
        scripted = [
            _Turn(calls=(ToolCallRequest(name="recall_metric", arguments={"query": "x"}),))
            for _ in range(3)
        ]
        scripted.append(_Turn(text="OK gathered enough."))   # synthesis
        router = FakeRouter(scripted)
        dispatcher = FakeDispatcher({"recall_metric": {"matches": []}})

        engine = V2ChatEngine(router, dispatcher, max_steps=3)
        result = engine.run(user_message="thử test loop", language="en")

        self.assertEqual(len(result.trace_steps), 3)
        self.assertEqual(result.answer, "OK gathered enough.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HelperTests(unittest.TestCase):
    def test_extract_table_from_sql_basic(self):
        self.assertEqual(_extract_table_from_sql("SELECT * FROM pv.gold.dim_facility LIMIT 10"),
                         "pv.gold.dim_facility")
        self.assertEqual(_extract_table_from_sql(""), "lakehouse")

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
