"""Coverage for LLMModelRouter response parsers (Gemini / OpenAI / Anthropic)."""
from __future__ import annotations

import pytest

from app.services.solar_ai_chat.llm_client import LLMModelRouter


class TestGeminiToolResponseParser:
    def test_ignores_non_dict_parts(self) -> None:
        raw = {"candidates": [{"content": {"parts": ["not-a-dict", {"text": "hello"}]}}]}
        result = LLMModelRouter._parse_gemini_tool_response(raw, "m", False)
        assert result.text == "hello"
        assert result.function_call is None

    def test_function_call_with_non_dict_args_falls_back_empty(self) -> None:
        raw = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "tool_x", "args": "not-a-dict"}},
        ]}}]}
        result = LLMModelRouter._parse_gemini_tool_response(raw, "m", False)
        assert result.function_call is not None
        assert result.function_call.name == "tool_x"
        assert result.function_call.arguments == {}

    def test_function_call_with_valid_args(self) -> None:
        raw = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "tool_y", "args": {"k": 1}}},
        ]}}]}
        result = LLMModelRouter._parse_gemini_tool_response(raw, "m", False)
        assert result.function_call.name == "tool_y"
        assert result.function_call.arguments == {"k": 1}

    def test_no_parts_raises(self) -> None:
        with pytest.raises(RuntimeError):
            LLMModelRouter._parse_gemini_tool_response(
                {"candidates": [{"content": {"parts": []}}]}, "m", False,
            )


class TestOpenAIToolResponseParser:
    def test_no_choices_raises(self) -> None:
        with pytest.raises(RuntimeError, match="choices"):
            LLMModelRouter._parse_openai_tool_response({"choices": []}, "m", False)

    def test_invalid_message_raises(self) -> None:
        raw = {"choices": [{"message": "not-a-dict"}]}
        with pytest.raises(RuntimeError, match="message"):
            LLMModelRouter._parse_openai_tool_response(raw, "m", False)

    def test_tool_call_with_skip_empty_name(self) -> None:
        raw = {"choices": [{"message": {
            "tool_calls": [
                {"function": {"name": "", "arguments": "{}"}},
                {"function": {"name": "valid_tool", "arguments": '{"k": 1}'}},
            ],
        }}]}
        result = LLMModelRouter._parse_openai_tool_response(raw, "m", False)
        assert result.function_call.name == "valid_tool"
        assert result.function_call.arguments == {"k": 1}

    def test_tool_call_with_invalid_json_args(self) -> None:
        raw = {"choices": [{"message": {
            "tool_calls": [{"function": {"name": "tool_a", "arguments": "{not-json}"}}],
        }}]}
        result = LLMModelRouter._parse_openai_tool_response(raw, "m", False)
        assert result.function_call.arguments == {}

    def test_tool_call_with_dict_args(self) -> None:
        raw = {"choices": [{"message": {
            "tool_calls": [{"function": {"name": "tool_b", "arguments": {"x": 1}}}],
        }}]}
        result = LLMModelRouter._parse_openai_tool_response(raw, "m", False)
        assert result.function_call.arguments == {"x": 1}

    def test_tool_call_with_invalid_args_type(self) -> None:
        raw = {"choices": [{"message": {
            "tool_calls": [{"function": {"name": "tool_c", "arguments": 42}}],
        }}]}
        result = LLMModelRouter._parse_openai_tool_response(raw, "m", False)
        assert result.function_call.arguments == {}

    def test_text_content(self) -> None:
        raw = {"choices": [{"message": {"content": "Plain answer"}}]}
        result = LLMModelRouter._parse_openai_tool_response(raw, "m", False)
        assert result.function_call is None
        assert result.text == "Plain answer"
