from unittest.mock import MagicMock

import pytest

from app.core.settings import SolarChatSettings
from app.services.solar_ai_chat.llm_client import (
    LLMModelRouter,
    LLMToolResult,
    ModelUnavailableError,
    ToolCallNotSupportedError,
    _is_temporary_unavailable_error,
)


def _make_settings(**overrides: object) -> SolarChatSettings:
    defaults = {
        "llm_api_format": "gemini",
        "llm_api_key": "test-key",
        "llm_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "primary_model": "gemini-2.5-flash-lite",
        "fallback_model": "gemini-2.5-flash",
        "request_timeout_seconds": 3.0,
    }
    defaults.update(overrides)
    settings = SolarChatSettings()
    for field_name, field_value in defaults.items():
        setattr(settings, field_name, field_value)
    return settings


class TestParseToolResponse:
    def test_parses_function_call(self) -> None:
        raw = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_system_overview",
                            "args": {},
                        }
                    }]
                }
            }]
        }
        result = LLMModelRouter._parse_tool_response(raw, "test-model", False)
        assert isinstance(result, LLMToolResult)
        assert result.function_call is not None
        assert result.function_call.name == "get_system_overview"
        assert result.text is None

    def test_parses_text_response(self) -> None:
        raw = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Xin chao!"}]
                }
            }]
        }
        result = LLMModelRouter._parse_tool_response(raw, "test-model", False)
        assert result.text == "Xin chao!"
        assert result.function_call is None

    def test_parses_function_call_with_args(self) -> None:
        raw = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_extreme_aqi",
                            "args": {"query_type": "highest", "timeframe": "day"},
                        }
                    }]
                }
            }]
        }
        result = LLMModelRouter._parse_tool_response(raw, "test-model", True)
        assert result.function_call.name == "get_extreme_aqi"
        assert result.function_call.arguments == {"query_type": "highest", "timeframe": "day"}
        assert result.fallback_used is True

    def test_raises_on_empty_candidates(self) -> None:
        with pytest.raises(RuntimeError, match="candidates"):
            LLMModelRouter._parse_tool_response({"candidates": []}, "m", False)

    def test_raises_on_empty_parts(self) -> None:
        raw = {"candidates": [{"content": {"parts": []}}]}
        with pytest.raises(RuntimeError, match="parts"):
            LLMModelRouter._parse_tool_response(raw, "m", False)

    def test_raises_on_neither_function_nor_text(self) -> None:
        raw = {"candidates": [{"content": {"parts": [{"other": "value"}]}}]}
        with pytest.raises(RuntimeError, match="neither"):
            LLMModelRouter._parse_tool_response(raw, "m", False)


class TestGenerateWithTools:
    def test_primary_success(self) -> None:
        mock_executor = MagicMock()
        mock_executor.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "get_system_overview", "args": {}}}]
                }
            }]
        }
        router = LLMModelRouter(
            settings=_make_settings(),
            request_executor=mock_executor,
        )
        result = router.generate_with_tools(
            messages=[{"role": "user", "parts": [{"text": "test"}]}],
            tools=[{"name": "get_system_overview", "parameters": {}}],
        )
        assert result.function_call.name == "get_system_overview"
        assert result.model_used == "gemini-2.5-flash-lite"
        assert result.fallback_used is False

    def test_fallback_on_primary_failure(self) -> None:
        call_count = {"n": 0}

        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Primary down")
            return {
                "candidates": [{
                    "content": {"parts": [{"text": "Fallback answer"}]}
                }]
            }

        router = LLMModelRouter(
            settings=_make_settings(),
            request_executor=mock_executor,
        )
        result = router.generate_with_tools(
            messages=[{"role": "user", "parts": [{"text": "test"}]}],
            tools=[],
        )
        assert result.text == "Fallback answer"
        assert result.model_used == "gemini-2.5-flash"
        assert result.fallback_used is True

    def test_raises_when_both_fail(self) -> None:
        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            raise RuntimeError("Down")

        router = LLMModelRouter(
            settings=_make_settings(),
            request_executor=mock_executor,
        )
        with pytest.raises(ModelUnavailableError):
            router.generate_with_tools(
                messages=[{"role": "user", "parts": [{"text": "test"}]}],
                tools=[],
            )

    def test_no_api_key_raises(self) -> None:
        settings = _make_settings()
        router = LLMModelRouter(settings=settings)
        router._api_key = None
        with pytest.raises(ModelUnavailableError, match="API key"):
            router.generate_with_tools([], [])

    def test_tool_use_failed_raises_tooling_not_supported_without_cooldown(self) -> None:
        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            raise RuntimeError(
                "LLM request failed: HTTP 400 Bad Request; body={\"error\":{\"code\":\"tool_use_failed\"}}"
            )

        settings = _make_settings(
            llm_api_format="openai",
            llm_base_url="https://api.openai.com/v1",
            primary_model="llama-3.3-70b-versatile",
            fallback_model="llama-3.3-70b-versatile",
        )
        router = LLMModelRouter(
            settings=settings,
            request_executor=mock_executor,
        )

        with pytest.raises(ToolCallNotSupportedError):
            router.generate_with_tools(
                messages=[{"role": "user", "parts": [{"text": "test"}]}],
                tools=[{"name": "get_system_overview", "parameters": {}}],
            )

        assert router._cooldown_until == 0.0


class TestTemporaryUnavailableClassifier:
    def test_tool_use_failed_is_not_temporary_unavailable(self) -> None:
        err = RuntimeError("HTTP 400 invalid_request_error tool_use_failed")
        assert _is_temporary_unavailable_error(err) is False


class TestSendToolResult:
    def test_sends_multi_turn_messages(self) -> None:
        mock_executor = MagicMock()
        mock_executor.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "AQI cao nhat la 150."}]}
            }]
        }
        router = LLMModelRouter(
            settings=_make_settings(),
            request_executor=mock_executor,
        )
        messages = [
            {"role": "user", "parts": [{"text": "AQI cao nhat?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_extreme_aqi", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_extreme_aqi", "response": {"result": {}}}}]},
        ]
        result = router.send_tool_result(messages, "gemini-2.5-flash-lite")
        assert result.text == "AQI cao nhat la 150."
        assert result.model_used == "gemini-2.5-flash-lite"
