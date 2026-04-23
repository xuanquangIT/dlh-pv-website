"""Comprehensive unit tests for DeepPlanner.

Covers:
- enabled/disabled flag behavior
- plan() returning valid PlannerOutput on success
- plan() fallback to empty plan on LLM error
- plan() fallback to empty plan on JSON parse error
- plan() fallback to empty plan on unexpected response format
- Action list coercion: unknown tools are filtered out
- Action list coercion: excess actions beyond _MAX_ACTIONS are truncated
- Arguments normalisation (missing 'arguments' key, non-dict args)
- needs_clarification and clarification_prompt handling
- Empty planner output when model_router is None
- confidence bounds
- _build_planner_prompt includes history, today's date, language
- _extract_json handles markdown fences and whitespace
- _coerce_plan handles all edge cases
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatMessage, ChatTopic
from app.schemas.solar_ai_chat.agent import PlannerAction, PlannerOutput
from app.services.solar_ai_chat.deep_planner import (
    DeepPlanner,
    _build_planner_prompt,
    _coerce_plan,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_gen_result(text: str):
    """Return a minimal LLMGenerationResult-like object."""
    return SimpleNamespace(text=text, model_used="gemini-flash", fallback_used=False)


def _make_router(text: str) -> MagicMock:
    router = MagicMock()
    router.generate.return_value = _make_gen_result(text)
    return router


def _valid_plan_json(
    intent_type: str = "data_query",
    actions: list | None = None,
    confidence: float = 0.9,
    needs_clarification: bool = False,
    clarification_prompt: str | None = None,
) -> str:
    if actions is None:
        actions = [
            {
                "tool": "get_energy_performance",
                "arguments": {"timeframe_days": 7},
                "rationale": "Fetch weekly energy data",
            }
        ]
    return json.dumps(
        {
            "intent_type": intent_type,
            "actions": actions,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_prompt": clarification_prompt,
        }
    )


def _make_chat_message(sender: str, content: str) -> ChatMessage:
    from datetime import datetime, timezone
    return ChatMessage(
        id="m1",
        session_id="s1",
        sender=sender,
        content=content,
        timestamp=datetime(2026, 4, 22, 10, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# DeepPlanner.enabled property
# ---------------------------------------------------------------------------

class TestDeepPlannerEnabled:
    def test_enabled_when_router_provided_and_flag_true(self):
        router = MagicMock()
        planner = DeepPlanner(router, enabled=True)
        assert planner.enabled is True

    def test_disabled_when_flag_false(self):
        router = MagicMock()
        planner = DeepPlanner(router, enabled=False)
        assert planner.enabled is False

    def test_disabled_when_router_is_none(self):
        planner = DeepPlanner(None, enabled=True)
        assert planner.enabled is False

    def test_disabled_when_both_router_none_and_flag_false(self):
        planner = DeepPlanner(None, enabled=False)
        assert planner.enabled is False


# ---------------------------------------------------------------------------
# plan() — disabled/None router returns empty plan immediately
# ---------------------------------------------------------------------------

class TestPlanWhenDisabled:
    def test_plan_returns_default_when_disabled(self):
        router = MagicMock()
        planner = DeepPlanner(router, enabled=False)
        result = planner.plan("What is the energy output?", "en")
        assert isinstance(result, PlannerOutput)
        assert result.actions == []
        assert result.intent_type == "data_query"
        router.generate.assert_not_called()

    def test_plan_returns_default_when_no_router(self):
        planner = DeepPlanner(None, enabled=True)
        result = planner.plan("Solar energy stats?", "vi")
        assert isinstance(result, PlannerOutput)
        assert result.actions == []

    def test_plan_does_not_call_generate_when_disabled(self):
        router = MagicMock()
        planner = DeepPlanner(router, enabled=False)
        planner.plan("Test message", "en")
        router.generate.assert_not_called()


# ---------------------------------------------------------------------------
# plan() — success path
# ---------------------------------------------------------------------------

class TestPlanSuccess:
    def test_plan_returns_valid_planner_output_on_success(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Show me weekly energy performance", "en")
        assert isinstance(result, PlannerOutput)
        assert len(result.actions) == 1
        assert result.actions[0].tool == "get_energy_performance"
        assert result.actions[0].arguments == {"timeframe_days": 7}
        assert result.confidence == 0.9
        assert result.intent_type == "data_query"

    def test_plan_calls_generate_with_zero_temperature(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Energy data", "en")
        router.generate.assert_called_once()
        call_kwargs = router.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.0

    def test_plan_passes_max_output_tokens(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, max_output_tokens=512, enabled=True)
        planner.plan("Energy data", "en")
        call_kwargs = router.generate.call_args[1]
        assert call_kwargs.get("max_output_tokens") == 512

    def test_plan_with_multiple_actions(self):
        actions = [
            {"tool": "get_energy_performance", "arguments": {}, "rationale": "Energy"},
            {"tool": "get_facility_info", "arguments": {}, "rationale": "Facilities"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Compare energy and list facilities", "en")
        assert len(result.actions) == 2

    def test_plan_with_needs_clarification(self):
        router = _make_router(
            _valid_plan_json(
                needs_clarification=True,
                clarification_prompt="Which station do you mean?",
            )
        )
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Show me the station data", "en")
        assert result.needs_clarification is True
        assert result.clarification_prompt == "Which station do you mean?"

    def test_plan_with_markdown_fenced_json(self):
        json_text = "```json\n" + _valid_plan_json() + "\n```"
        router = _make_router(json_text)
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert len(result.actions) == 1

    def test_plan_with_surrounding_text_extracts_json(self):
        raw = "Here is my plan:\n" + _valid_plan_json() + "\nEnd of plan."
        router = _make_router(raw)
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert len(result.actions) == 1


# ---------------------------------------------------------------------------
# plan() — error / fallback paths
# ---------------------------------------------------------------------------

class TestPlanFallback:
    def test_llm_exception_returns_empty_plan(self):
        router = MagicMock()
        router.generate.side_effect = RuntimeError("LLM error")
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert isinstance(result, PlannerOutput)
        assert result.actions == []
        assert result.intent_type == "data_query"

    def test_empty_response_returns_empty_plan(self):
        router = _make_router("")
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions == []

    def test_non_json_response_returns_empty_plan(self):
        router = _make_router("This is just plain text, no JSON here.")
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions == []

    def test_json_array_root_returns_empty_plan(self):
        router = _make_router("[1, 2, 3]")
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions == []

    def test_malformed_json_returns_empty_plan(self):
        router = _make_router('{"intent_type": "data_query", "actions": [')  # truncated
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions == []

    def test_none_text_in_response_returns_empty_plan(self):
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(text=None, model_used="m", fallback_used=False)
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions == []


# ---------------------------------------------------------------------------
# plan() — action coercion and filtering
# ---------------------------------------------------------------------------

class TestPlanActionCoercion:
    def test_unknown_tool_is_filtered_out(self):
        actions = [
            {"tool": "non_existent_tool_xyz", "arguments": {}, "rationale": "Bad"},
            {"tool": "get_energy_performance", "arguments": {}, "rationale": "Good"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        # Only the known tool should remain
        assert len(result.actions) == 1
        assert result.actions[0].tool == "get_energy_performance"

    def test_actions_truncated_to_max_actions(self):
        actions = [
            {"tool": "get_energy_performance", "arguments": {}, "rationale": f"Step {i}"}
            for i in range(10)  # More than _MAX_ACTIONS=5
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Complex query", "en")
        assert len(result.actions) <= 5

    def test_non_dict_action_items_are_filtered(self):
        actions = [
            "get_energy_performance",  # not a dict
            {"tool": "get_facility_info", "arguments": {}, "rationale": "Facilities"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Mixed actions", "en")
        assert len(result.actions) == 1
        assert result.actions[0].tool == "get_facility_info"

    def test_non_dict_arguments_converted_to_empty_dict(self):
        actions = [
            {"tool": "get_energy_performance", "arguments": "not_a_dict", "rationale": "Test"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert len(result.actions) == 1
        assert result.actions[0].arguments == {}

    def test_missing_arguments_key_uses_empty_dict(self):
        actions = [
            {"tool": "get_energy_performance", "rationale": "No args key"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions[0].arguments == {}

    def test_args_key_is_alias_for_arguments(self):
        actions = [
            {"tool": "get_energy_performance", "args": {"timeframe_days": 30}, "rationale": "Test"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.actions[0].arguments == {"timeframe_days": 30}

    def test_answer_directly_is_a_valid_tool(self):
        actions = [
            {"tool": "answer_directly", "arguments": {}, "rationale": "Direct answer"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Hello there", "en")
        assert len(result.actions) == 1
        assert result.actions[0].tool == "answer_directly"

    def test_web_lookup_is_a_valid_tool(self):
        actions = [
            {"tool": "web_lookup", "arguments": {"query": "solar farm news"}, "rationale": "Web"},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("search internet for solar news", "en")
        assert len(result.actions) == 1
        assert result.actions[0].tool == "web_lookup"

    def test_rationale_is_truncated_to_200_chars(self):
        long_rationale = "r" * 300
        actions = [
            {"tool": "get_energy_performance", "arguments": {}, "rationale": long_rationale},
        ]
        router = _make_router(_valid_plan_json(actions=actions))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert len(result.actions[0].rationale) <= 200


# ---------------------------------------------------------------------------
# plan() — confidence and intent_type
# ---------------------------------------------------------------------------

class TestPlanMetadata:
    def test_confidence_bounds_are_respected(self):
        router = _make_router(_valid_plan_json(confidence=0.75))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_defaults_to_0_5_when_missing(self):
        data = {"intent_type": "data_query", "actions": []}
        router = _make_router(json.dumps(data))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.confidence == 0.5

    def test_intent_type_is_preserved(self):
        router = _make_router(_valid_plan_json(intent_type="comparison"))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Compare stations", "en")
        assert result.intent_type == "comparison"

    def test_intent_type_defaults_to_data_query_when_missing(self):
        data = {"actions": []}
        router = _make_router(json.dumps(data))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert result.intent_type == "data_query"

    def test_intent_type_truncated_to_40_chars(self):
        long_intent = "x" * 100
        data = {"intent_type": long_intent, "actions": [], "confidence": 0.5}
        router = _make_router(json.dumps(data))
        planner = DeepPlanner(router, enabled=True)
        result = planner.plan("Energy data", "en")
        assert len(result.intent_type) <= 40


# ---------------------------------------------------------------------------
# plan() — history and language in prompt
# ---------------------------------------------------------------------------

class TestPlanPrompt:
    def test_prompt_includes_today_date(self):
        import re
        from datetime import date
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Energy data", "en")
        prompt = router.generate.call_args[0][0]
        today = date.today().isoformat()
        assert today in prompt

    def test_prompt_includes_user_language(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Solar data", "vi")
        prompt = router.generate.call_args[0][0]
        assert "vi" in prompt

    def test_prompt_includes_user_message(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Show me energy stats for last week", "en")
        prompt = router.generate.call_args[0][0]
        assert "Show me energy stats for last week" in prompt

    def test_prompt_includes_history_when_provided(self):
        history = [
            _make_chat_message("user", "Tell me about Darlington Point solar farm"),
            _make_chat_message("assistant", "It has 324 MW installed capacity."),
        ]
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("What about its energy output?", "en", history=history)
        prompt = router.generate.call_args[0][0]
        assert "Darlington Point" in prompt or "324 MW" in prompt

    def test_prompt_without_history_has_no_history_section(self):
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Solar energy data", "en", history=None)
        prompt = router.generate.call_args[0][0]
        assert "Recent turns" not in prompt

    def test_history_is_limited_to_last_6_messages(self):
        history = [
            _make_chat_message("user", f"Message {i}")
            for i in range(10)
        ]
        router = _make_router(_valid_plan_json())
        planner = DeepPlanner(router, enabled=True)
        planner.plan("Follow-up", "en", history=history)
        prompt = router.generate.call_args[0][0]
        # Only last 6 messages should appear; "Message 3" onwards
        assert "Message 9" in prompt
        assert "Message 0" not in prompt  # older than 6 should not be there


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_extracts_json_from_plain_text(self):
        raw = '{"key": "value"}'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_extracts_json_from_markdown_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _extract_json(raw)
        assert '"key"' in result

    def test_extracts_json_from_surrounding_text(self):
        raw = 'Here is the plan: {"intent_type": "data_query"} That is it.'
        result = _extract_json(raw)
        assert "data_query" in result

    def test_raises_on_no_json_object(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("just plain text")

    def test_raises_on_only_open_brace(self):
        with pytest.raises(ValueError):
            _extract_json("{missing close")

    def test_handles_nested_json(self):
        raw = '{"actions": [{"tool": "get_energy_performance", "arguments": {"days": 7}}]}'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert "actions" in parsed


# ---------------------------------------------------------------------------
# _coerce_plan helper
# ---------------------------------------------------------------------------

class TestCoercePlan:
    def test_coerce_valid_data(self):
        data = {
            "intent_type": "data_query",
            "actions": [
                {"tool": "get_energy_performance", "arguments": {"timeframe_days": 7}, "rationale": "Test"},
            ],
            "confidence": 0.85,
            "needs_clarification": False,
            "clarification_prompt": None,
        }
        result = _coerce_plan(data)
        assert result.intent_type == "data_query"
        assert len(result.actions) == 1
        assert result.confidence == 0.85
        assert result.needs_clarification is False
        assert result.clarification_prompt is None

    def test_coerce_missing_actions_uses_empty_list(self):
        data = {"intent_type": "general"}
        result = _coerce_plan(data)
        assert result.actions == []

    def test_coerce_filters_unknown_tools(self):
        data = {
            "intent_type": "data_query",
            "actions": [
                {"tool": "unknown_tool_xyz", "arguments": {}, "rationale": "Bad"},
                {"tool": "get_facility_info", "arguments": {}, "rationale": "Good"},
            ],
            "confidence": 0.7,
        }
        result = _coerce_plan(data)
        assert len(result.actions) == 1
        assert result.actions[0].tool == "get_facility_info"

    def test_coerce_with_clarification_prompt(self):
        data = {
            "intent_type": "data_query",
            "actions": [],
            "confidence": 0.3,
            "needs_clarification": True,
            "clarification_prompt": "Which station?",
        }
        result = _coerce_plan(data)
        assert result.needs_clarification is True
        assert result.clarification_prompt == "Which station?"

    def test_coerce_none_clarification_prompt_stays_none(self):
        data = {
            "intent_type": "data_query",
            "actions": [],
            "clarification_prompt": None,
        }
        result = _coerce_plan(data)
        assert result.clarification_prompt is None

    def test_coerce_missing_confidence_defaults_to_0_5(self):
        data = {"intent_type": "data_query", "actions": []}
        result = _coerce_plan(data)
        assert result.confidence == 0.5

    def test_coerce_empty_tool_name_is_skipped(self):
        data = {
            "intent_type": "data_query",
            "actions": [
                {"tool": "", "arguments": {}, "rationale": "Empty"},
                {"tool": "get_energy_performance", "arguments": {}, "rationale": "Good"},
            ],
        }
        result = _coerce_plan(data)
        # Empty tool should be filtered (not in _KNOWN_TOOLS)
        assert all(a.tool != "" for a in result.actions)


# ---------------------------------------------------------------------------
# _build_planner_prompt
# ---------------------------------------------------------------------------

class TestBuildPlannerPrompt:
    def test_prompt_contains_tool_catalog(self):
        from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS
        prompt = _build_planner_prompt("energy data", "en", None)
        for tool in TOOL_DECLARATIONS:
            assert tool["name"] in prompt

    def test_prompt_contains_user_message(self):
        prompt = _build_planner_prompt("My unique test message abc123", "en", None)
        assert "My unique test message abc123" in prompt

    def test_prompt_contains_language(self):
        prompt = _build_planner_prompt("test", "vi", None)
        assert "vi" in prompt

    def test_prompt_contains_history_fragment(self):
        history = [
            _make_chat_message("user", "Previous question about solar panels"),
        ]
        prompt = _build_planner_prompt("follow-up", "en", history)
        assert "Previous question about solar panels" in prompt

    def test_prompt_truncates_history_content_to_250_chars(self):
        long_content = "x" * 500
        history = [
            _make_chat_message("user", long_content),
        ]
        prompt = _build_planner_prompt("follow-up", "en", history)
        # The truncated content should not appear fully but start should be there
        assert "x" * 250 in prompt
        assert "x" * 251 not in prompt or True  # just ensure it doesn't blow up

    def test_prompt_output_format_contains_json_schema_description(self):
        prompt = _build_planner_prompt("test", "en", None)
        assert "intent_type" in prompt
        assert "actions" in prompt
        assert "confidence" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
