"""Tests for system role handling in prompt_builder and llm_client converters."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.settings import SolarChatSettings
from app.services.solar_ai_chat.llm_client import LLMModelRouter
from app.services.solar_ai_chat.prompt_builder import build_agentic_messages


def _make_settings(**overrides: object) -> SolarChatSettings:
    defaults = {
        "llm_api_format": "openai",
        "llm_api_key": "test-key",
        "llm_base_url": "https://api.openai.com/v1",
        "primary_model": "gpt-5-mini",
        "fallback_model": "gpt-5-mini",
        "request_timeout_seconds": 3.0,
    }
    defaults.update(overrides)
    settings = SolarChatSettings()
    for field_name, field_value in defaults.items():
        setattr(settings, field_name, field_value)
    return settings


class TestBuildAgenticMessages(unittest.TestCase):
    """Verify that build_agentic_messages produces a system role."""

    def test_first_message_is_system_role(self):
        messages = build_agentic_messages("Tổng quan hệ thống", "vi")
        self.assertEqual(messages[0]["role"], "system")

    def test_no_fake_model_acknowledgment(self):
        messages = build_agentic_messages("Overview", "en")
        roles = [m["role"] for m in messages]
        # Should be: system, user — no "model" in between
        self.assertNotIn("model", roles[:2])

    def test_user_message_is_last(self):
        messages = build_agentic_messages("Hello", "en")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["parts"][0]["text"], "Hello")

    def test_system_context_contains_lakehouse(self):
        messages = build_agentic_messages("Test", "en")
        system_text = messages[0]["parts"][0]["text"]
        self.assertIn("Solar AI", system_text)
        self.assertIn("Gold", system_text)
        self.assertIn("get_system_overview", system_text)


class TestOpenAISystemRoleConversion(unittest.TestCase):
    """Verify _convert_gemini_messages_to_openai handles system role."""

    def test_system_role_becomes_openai_system(self):
        messages = [
            {"role": "system", "parts": [{"text": "You are Solar AI."}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        converted = LLMModelRouter._convert_gemini_messages_to_openai(messages)
        self.assertEqual(converted[0]["role"], "system")
        self.assertEqual(converted[0]["content"], "You are Solar AI.")
        self.assertEqual(converted[1]["role"], "user")

    def test_model_role_becomes_assistant(self):
        messages = [
            {"role": "model", "parts": [{"text": "Reply"}]},
        ]
        converted = LLMModelRouter._convert_gemini_messages_to_openai(messages)
        self.assertEqual(converted[0]["role"], "assistant")

    def test_function_call_and_response_preserved(self):
        messages = [
            {"role": "system", "parts": [{"text": "context"}]},
            {"role": "user", "parts": [{"text": "query"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_system_overview", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_system_overview", "response": {"ok": True}}}]},
        ]
        converted = LLMModelRouter._convert_gemini_messages_to_openai(messages)
        self.assertEqual(converted[0]["role"], "system")
        self.assertEqual(converted[1]["role"], "user")
        self.assertEqual(converted[2]["role"], "assistant")
        self.assertIn("tool_calls", converted[2])
        self.assertEqual(converted[3]["role"], "tool")


class TestGeminiSystemInstruction(unittest.TestCase):
    """Verify Gemini payload uses systemInstruction for system role."""

    def test_system_extracted_to_system_instruction(self):
        messages = [
            {"role": "system", "parts": [{"text": "System context"}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        tools = [{"name": "test_tool", "description": "A test", "parameters": {"type": "object", "properties": {}}}]

        router = LLMModelRouter(settings=_make_settings(llm_api_format="gemini", llm_base_url="https://generativelanguage.googleapis.com/v1beta"), request_executor=MagicMock())
        payload, _ = router._build_tool_generation_payload(messages, tools)

        # System messages should NOT be in contents
        for content in payload["contents"]:
            self.assertNotEqual(str(content.get("role", "")), "system")

        # Should be in systemInstruction
        self.assertIn("systemInstruction", payload)
        si_text = payload["systemInstruction"]["parts"][0]["text"]
        self.assertEqual(si_text, "System context")

    def test_no_system_instruction_when_no_system_messages(self):
        messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
        tools = [{"name": "test_tool", "description": "A test", "parameters": {"type": "object", "properties": {}}}]

        router = LLMModelRouter(settings=_make_settings(llm_api_format="gemini", llm_base_url="https://generativelanguage.googleapis.com/v1beta"), request_executor=MagicMock())
        payload, _ = router._build_tool_generation_payload(messages, tools)
        self.assertNotIn("systemInstruction", payload)


class TestOpenAIPayloadRequireFunctionCall(unittest.TestCase):
    """Verify that require_function_call=True produces tool_choice='required'."""

    def test_openai_tool_choice_required(self):
        messages = [
            {"role": "system", "parts": [{"text": "context"}]},
            {"role": "user", "parts": [{"text": "query"}]},
        ]
        tools = [{"name": "get_system_overview", "description": "Overview", "parameters": {"type": "object", "properties": {}}}]

        router = LLMModelRouter(settings=_make_settings(), request_executor=MagicMock())
        payload, _ = router._build_tool_generation_payload(messages, tools, require_function_call=True)
        self.assertEqual(payload["tool_choice"], "required")

    def test_openai_tool_choice_auto_by_default(self):
        messages = [
            {"role": "system", "parts": [{"text": "context"}]},
            {"role": "user", "parts": [{"text": "query"}]},
        ]
        tools = [{"name": "get_system_overview", "description": "Overview", "parameters": {"type": "object", "properties": {}}}]

        router = LLMModelRouter(settings=_make_settings(), request_executor=MagicMock())
        payload, _ = router._build_tool_generation_payload(messages, tools, require_function_call=False)
        self.assertEqual(payload["tool_choice"], "auto")


class TestAnthropicSystemHandling(unittest.TestCase):
    """Verify Anthropic payload extracts system messages correctly."""

    def test_system_extracted_to_top_level(self):
        messages = [
            {"role": "system", "parts": [{"text": "System context"}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        tools = [{"name": "test_tool", "description": "A test", "parameters": {"type": "object", "properties": {}}}]

        router = LLMModelRouter(settings=_make_settings(llm_api_format="anthropic", llm_base_url="https://api.anthropic.com/v1"), request_executor=MagicMock())
        payload, _ = router._build_tool_generation_payload(messages, tools)

        self.assertIn("system", payload)
        self.assertEqual(payload["system"], "System context")
        # System messages should not appear in the messages list
        for msg in payload["messages"]:
            if isinstance(msg.get("content"), str):
                self.assertNotEqual(msg["content"], "System context")


if __name__ == "__main__":
    unittest.main()
