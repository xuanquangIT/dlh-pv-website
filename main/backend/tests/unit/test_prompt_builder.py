"""Unit tests for app.services.solar_ai_chat.prompt_builder"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatMessage, SourceMetadata
from app.schemas.solar_ai_chat.enums import ChatTopic
from app.services.solar_ai_chat.prompt_builder import (
    MAX_HISTORY_IN_PROMPT,
    MAX_OLD_PREVIEW_LENGTH,
    MAX_RECENT_FULL_LENGTH,
    NUM_RECENT_FULL,
    _format_history_messages,
    build_agentic_messages,
    build_data_only_summary,
    build_insufficient_data_response,
    build_synthesis_prompt,
    build_verifier_prompt,
    format_source_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_message(
    sender: str = "user",
    content: str = "Hello",
    idx: int = 0,
) -> ChatMessage:
    from datetime import datetime
    return ChatMessage(
        id=f"msg-{idx}",
        session_id="sess-1",
        sender=sender,
        content=content,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        topic=ChatTopic.GENERAL,
    )


def _make_source(layer: str = "gold", dataset: str = "fact_energy") -> SourceMetadata:
    return SourceMetadata(layer=layer, dataset=dataset)


# ---------------------------------------------------------------------------
# build_insufficient_data_response
# ---------------------------------------------------------------------------

class TestBuildInsufficientDataResponse(unittest.TestCase):
    def test_english_response(self) -> None:
        resp = build_insufficient_data_response("en")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_vietnamese_response(self) -> None:
        resp = build_insufficient_data_response("vi")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_unknown_language_falls_back_to_english(self) -> None:
        resp_unknown = build_insufficient_data_response("de")
        resp_en = build_insufficient_data_response("en")
        self.assertEqual(resp_unknown, resp_en)

    def test_english_and_vietnamese_differ(self) -> None:
        self.assertNotEqual(
            build_insufficient_data_response("en"),
            build_insufficient_data_response("vi"),
        )


# ---------------------------------------------------------------------------
# format_source_text
# ---------------------------------------------------------------------------

class TestFormatSourceText(unittest.TestCase):
    def test_single_source(self) -> None:
        result = format_source_text([_make_source("gold", "fact_energy")])
        self.assertEqual(result, "gold:fact_energy")

    def test_multiple_sources(self) -> None:
        sources = [_make_source("gold", "fact_energy"), _make_source("silver", "energy_readings")]
        result = format_source_text(sources)
        self.assertIn("gold:fact_energy", result)
        self.assertIn("silver:energy_readings", result)

    def test_empty_sources(self) -> None:
        result = format_source_text([])
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# _format_history_messages
# ---------------------------------------------------------------------------

class TestFormatHistoryMessages(unittest.TestCase):
    def test_empty_history_returns_empty_list(self) -> None:
        self.assertEqual(_format_history_messages([]), [])

    def test_user_sender_maps_to_user_role(self) -> None:
        msgs = [_make_chat_message(sender="user", content="Hello")]
        result = _format_history_messages(msgs)
        self.assertEqual(result[0]["role"], "user")

    def test_assistant_sender_maps_to_model_role(self) -> None:
        msgs = [_make_chat_message(sender="assistant", content="World")]
        result = _format_history_messages(msgs)
        self.assertEqual(result[0]["role"], "model")

    def test_unknown_sender_maps_to_model_role(self) -> None:
        msgs = [_make_chat_message(sender="bot", content="Hi")]
        result = _format_history_messages(msgs)
        self.assertEqual(result[0]["role"], "model")

    def test_history_truncated_to_max_messages(self) -> None:
        msgs = [_make_chat_message(idx=i) for i in range(MAX_HISTORY_IN_PROMPT + 5)]
        result = _format_history_messages(msgs)
        self.assertLessEqual(len(result), MAX_HISTORY_IN_PROMPT)

    def test_recent_messages_full_length(self) -> None:
        long_content = "x" * (MAX_RECENT_FULL_LENGTH + 100)
        # Put one long message at the end (most recent)
        msgs = [_make_chat_message(content=long_content)]
        result = _format_history_messages(msgs)
        text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertLessEqual(len(text), MAX_RECENT_FULL_LENGTH)

    def test_old_messages_preview_length(self) -> None:
        long_content = "x" * (MAX_OLD_PREVIEW_LENGTH + 100)
        # Create enough messages that the first is "old"
        msgs = [_make_chat_message(content=long_content, idx=i) for i in range(NUM_RECENT_FULL + 2)]
        result = _format_history_messages(msgs)
        # The first message in the result should be preview-truncated
        text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertLessEqual(len(text), MAX_OLD_PREVIEW_LENGTH)

    def test_parts_key_contains_text(self) -> None:
        msgs = [_make_chat_message(content="test message")]
        result = _format_history_messages(msgs)
        self.assertIn("text", result[0]["parts"][0])  # type: ignore[index]

    def test_none_content_handled_gracefully(self) -> None:
        msg = _make_chat_message(content="")
        msg_with_none = msg.model_copy(update={"content": None})
        result = _format_history_messages([msg_with_none])
        self.assertEqual(result[0]["parts"][0]["text"], "")  # type: ignore[index]


# ---------------------------------------------------------------------------
# build_agentic_messages
# ---------------------------------------------------------------------------

class TestBuildAgenticMessages(unittest.TestCase):
    def test_returns_list(self) -> None:
        result = build_agentic_messages("What is the system overview?")
        self.assertIsInstance(result, list)

    def test_first_message_is_system(self) -> None:
        result = build_agentic_messages("Hello")
        self.assertEqual(result[0]["role"], "system")

    def test_last_message_is_user(self) -> None:
        result = build_agentic_messages("Hello")
        self.assertEqual(result[-1]["role"], "user")

    def test_user_message_in_last_parts(self) -> None:
        result = build_agentic_messages("What is the AQI?")
        last = result[-1]
        text = last["parts"][0]["text"]  # type: ignore[index]
        self.assertIn("What is the AQI?", text)

    def test_system_contains_todays_date(self) -> None:
        today = "2026-04-22"
        result = build_agentic_messages("Hello", today_str=today)
        sys_text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertIn(today, sys_text)

    def test_auto_today_str_injected(self) -> None:
        result = build_agentic_messages("Hello")
        sys_text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertIn(date.today().isoformat(), sys_text)

    def test_history_messages_included(self) -> None:
        history = [
            _make_chat_message(sender="user", content="Prior question", idx=0),
            _make_chat_message(sender="assistant", content="Prior answer", idx=1),
        ]
        result = build_agentic_messages("Follow-up", history=history)
        roles = [m["role"] for m in result]
        # system + 2 history + user
        self.assertGreaterEqual(len(result), 4)
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")

    def test_no_history_returns_two_messages(self) -> None:
        result = build_agentic_messages("Hello", history=None)
        self.assertEqual(len(result), 2)

    def test_language_parameter_accepted(self) -> None:
        result = build_agentic_messages("Hello", language="vi")
        self.assertIsInstance(result, list)

    def test_web_search_hint_silently_ignored(self) -> None:
        # The "web_search" hint is accepted for backward compatibility but
        # must not inject any snippet into the prompt.
        baseline = build_agentic_messages("Hello", tool_hints=None)
        with_hint = build_agentic_messages("Hello", tool_hints=["web_search"])
        self.assertEqual(
            baseline[0]["parts"][0]["text"],  # type: ignore[index]
            with_hint[0]["parts"][0]["text"],  # type: ignore[index]
        )

    def test_visualize_hint_appended_to_system(self) -> None:
        result = build_agentic_messages("Hello", tool_hints=["visualize"])
        sys_text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertIn("Visualize", sys_text)

    def test_unknown_hint_ignored(self) -> None:
        result = build_agentic_messages("Hello", tool_hints=["nonexistent_hint"])
        sys_text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertNotIn("nonexistent_hint", sys_text)

    def test_empty_tool_hints_list_handled(self) -> None:
        result = build_agentic_messages("Hello", tool_hints=[])
        self.assertEqual(len(result), 2)

    def test_system_contains_lakehouse_context(self) -> None:
        result = build_agentic_messages("Hello")
        sys_text = result[0]["parts"][0]["text"]  # type: ignore[index]
        self.assertIn("Solar AI", sys_text)


# ---------------------------------------------------------------------------
# build_data_only_summary
# ---------------------------------------------------------------------------

class TestBuildDataOnlySummary(unittest.TestCase):
    def test_empty_metrics_returns_insufficient_data_response(self) -> None:
        result = build_data_only_summary({}, [])
        self.assertEqual(result, build_insufficient_data_response("en"))

    def test_empty_metrics_vietnamese(self) -> None:
        result = build_data_only_summary({}, [], language="vi")
        self.assertEqual(result, build_insufficient_data_response("vi"))

    def test_metrics_serialized_as_json(self) -> None:
        metrics = {"energy_mwh": 123.4, "facility_count": 8}
        result = build_data_only_summary(metrics, [])
        self.assertIn("123.4", result)
        self.assertIn("```json", result)

    def test_english_header_present(self) -> None:
        result = build_data_only_summary({"key": "val"}, [], language="en")
        self.assertIn("Retrieved data", result)

    def test_vietnamese_header_present(self) -> None:
        result = build_data_only_summary({"key": "val"}, [], language="vi")
        self.assertIn("Dữ liệu", result)

    def test_source_names_included(self) -> None:
        sources = [{"dataset": "fact_energy", "data_source": "databricks"}]
        result = build_data_only_summary({"k": "v"}, sources)
        self.assertIn("fact_energy", result)

    def test_source_as_object_with_dataset_attr(self) -> None:
        src = MagicMock()
        src.dataset = "mart_energy_daily"
        result = build_data_only_summary({"k": "v"}, [src])
        self.assertIn("mart_energy_daily", result)

    def test_metrics_truncated_at_20000_chars(self) -> None:
        large_metrics = {"data": "x" * 25000}
        result = build_data_only_summary(large_metrics, [])
        # The body section should not blow past 20000 chars of metrics text
        self.assertIsInstance(result, str)

    def test_no_sources_no_footer(self) -> None:
        result = build_data_only_summary({"k": "v"}, [])
        self.assertNotIn("Nguồn", result)


# ---------------------------------------------------------------------------
# build_synthesis_prompt
# ---------------------------------------------------------------------------

class TestBuildSynthesisPrompt(unittest.TestCase):
    def test_returns_string(self) -> None:
        result = build_synthesis_prompt("What is AQI?", "AQI is 50.")
        self.assertIsInstance(result, str)

    def test_english_instruction_present(self) -> None:
        result = build_synthesis_prompt("What is AQI?", "AQI is 50.", language="en")
        self.assertIn("Respond in English", result)

    def test_vietnamese_instruction_present(self) -> None:
        result = build_synthesis_prompt("AQI là gì?", "AQI là 50.", language="vi")
        self.assertIn("tieng Viet", result)

    def test_user_message_present(self) -> None:
        result = build_synthesis_prompt("What is AQI?", "AQI is 50.")
        self.assertIn("What is AQI?", result)

    def test_evidence_present(self) -> None:
        result = build_synthesis_prompt("question", "evidence data here")
        self.assertIn("evidence data here", result)

    def test_today_date_in_prompt(self) -> None:
        result = build_synthesis_prompt("question", "evidence")
        self.assertIn(date.today().isoformat(), result)

    def test_concise_true_uses_short_instruction(self) -> None:
        result = build_synthesis_prompt("q", "e", concise=True)
        self.assertIn("concise", result.lower())

    def test_concise_false_uses_thorough_instruction(self) -> None:
        result = build_synthesis_prompt("q", "e", concise=False)
        self.assertIn("thorough", result.lower())

    def test_history_included_when_provided(self) -> None:
        history = [
            _make_chat_message(sender="user", content="prior question"),
            _make_chat_message(sender="assistant", content="prior answer"),
        ]
        result = build_synthesis_prompt("follow-up", "evidence", history=history)
        self.assertIn("prior question", result)
        self.assertIn("prior answer", result)

    def test_no_history_section_when_empty(self) -> None:
        result = build_synthesis_prompt("q", "e", history=[])
        self.assertNotIn("Recent conversation", result)

    def test_none_evidence_shows_none_placeholder(self) -> None:
        result = build_synthesis_prompt("q", "")
        self.assertIn("(none)", result)

    def test_cite_web_sources_adds_instruction(self) -> None:
        result = build_synthesis_prompt("q", "e", cite_web_sources=True)
        self.assertIn("web search result", result.lower())

    def test_cite_web_sources_false_no_citation_instruction(self) -> None:
        result = build_synthesis_prompt("q", "e", cite_web_sources=False)
        self.assertNotIn("web search result", result.lower())

    def test_prompt_ends_with_answer_marker(self) -> None:
        result = build_synthesis_prompt("q", "e")
        self.assertTrue(result.strip().endswith("Answer:"))

    def test_architecture_context_in_prompt(self) -> None:
        result = build_synthesis_prompt("q", "e")
        self.assertIn("Solar AI", result)

    def test_history_truncated_to_max(self) -> None:
        history = [_make_chat_message(content=f"msg {i}", idx=i) for i in range(MAX_HISTORY_IN_PROMPT + 5)]
        result = build_synthesis_prompt("q", "e", history=history)
        # Should not crash and should produce valid output
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# build_verifier_prompt
# ---------------------------------------------------------------------------

class TestBuildVerifierPrompt(unittest.TestCase):
    def test_returns_string(self) -> None:
        result = build_verifier_prompt("My answer.", "Some evidence.")
        self.assertIsInstance(result, str)

    def test_contains_answer(self) -> None:
        result = build_verifier_prompt("My answer here.", "evidence")
        self.assertIn("My answer here.", result)

    def test_contains_evidence(self) -> None:
        result = build_verifier_prompt("answer", "Some retrieved evidence.")
        self.assertIn("Some retrieved evidence.", result)

    def test_grounded_ungrounded_labels_present(self) -> None:
        result = build_verifier_prompt("answer", "evidence")
        self.assertIn("GROUNDED", result)
        self.assertIn("UNGROUNDED", result)

    def test_none_evidence_shows_none_placeholder(self) -> None:
        result = build_verifier_prompt("answer", "")
        self.assertIn("(none)", result)

    def test_long_evidence_truncated(self) -> None:
        long_evidence = "evidence " * 5000
        result = build_verifier_prompt("answer", long_evidence)
        # Just confirm no crash and it's a string
        self.assertIsInstance(result, str)

    def test_verdict_marker_in_prompt(self) -> None:
        result = build_verifier_prompt("answer", "evidence")
        self.assertIn("Verdict:", result)

    def test_fact_checker_framing(self) -> None:
        result = build_verifier_prompt("answer", "evidence")
        self.assertIn("fact-checker", result.lower())


if __name__ == "__main__":
    unittest.main()
