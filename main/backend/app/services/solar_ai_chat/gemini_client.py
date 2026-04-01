import json
import logging
import time
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.settings import SolarChatSettings

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when both primary and fallback models are unavailable."""


@dataclass(frozen=True)
class GeminiGenerationResult:
    text: str
    model_used: str
    fallback_used: bool


@dataclass(frozen=True)
class FunctionCallRequest:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class GeminiToolResult:
    function_call: FunctionCallRequest | None
    text: str | None
    model_used: str
    fallback_used: bool


class GeminiModelRouter:
    """Gemini client with primary/fallback model routing."""

    def __init__(
        self,
        settings: SolarChatSettings,
        request_executor: Callable[[str, dict[str, object], float], dict[str, object]] | None = None,
    ) -> None:
        self._api_key = settings.gemini_api_key
        self._base_url = settings.gemini_base_url.rstrip("/")
        self._primary_model = settings.primary_model
        self._fallback_model = settings.fallback_model
        self._timeout = settings.request_timeout_seconds
        self._request_executor = request_executor or self._execute_request

    def generate(self, prompt: str) -> GeminiGenerationResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        started = time.perf_counter()
        try:
            primary_answer = self._call_model(model_name=self._primary_model, prompt=prompt)
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "solar_chat_model_selection",
                extra={
                    "provider": "google_gemini",
                    "selected_model": self._primary_model,
                    "latency_ms": duration_ms,
                    "fallback_used": False,
                },
            )
            return GeminiGenerationResult(
                text=primary_answer,
                model_used=self._primary_model,
                fallback_used=False,
            )
        except Exception as primary_error:
            logger.warning(
                "solar_chat_primary_model_failed",
                extra={
                    "provider": "google_gemini",
                    "primary_model": self._primary_model,
                    "fallback_model": self._fallback_model,
                    "error": str(primary_error),
                },
            )

        fallback_started = time.perf_counter()
        try:
            fallback_answer = self._call_model(model_name=self._fallback_model, prompt=prompt)
            duration_ms = int((time.perf_counter() - fallback_started) * 1000)
            logger.info(
                "solar_chat_model_selection",
                extra={
                    "provider": "google_gemini",
                    "selected_model": self._fallback_model,
                    "latency_ms": duration_ms,
                    "fallback_used": True,
                },
            )
            return GeminiGenerationResult(
                text=fallback_answer,
                model_used=self._fallback_model,
                fallback_used=True,
            )
        except Exception as fallback_error:
            logger.error(
                "solar_chat_fallback_model_failed",
                extra={
                    "provider": "google_gemini",
                    "primary_model": self._primary_model,
                    "fallback_model": self._fallback_model,
                    "error": str(fallback_error),
                },
            )
            raise ModelUnavailableError(
                "Primary and fallback Gemini models are both unavailable."
            ) from fallback_error

    def generate_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> GeminiToolResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        payload = {
            "contents": messages,
            "tools": [{"function_declarations": tools}],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }

        started = time.perf_counter()
        try:
            raw = self._call_model_raw(self._primary_model, payload)
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "solar_chat_tool_call",
                extra={"model": self._primary_model, "latency_ms": duration_ms, "fallback": False},
            )
            return self._parse_tool_response(raw, self._primary_model, fallback_used=False)
        except Exception as primary_err:
            logger.warning("solar_chat_tool_primary_failed", extra={"error": str(primary_err)})

        fallback_started = time.perf_counter()
        try:
            raw = self._call_model_raw(self._fallback_model, payload)
            duration_ms = int((time.perf_counter() - fallback_started) * 1000)
            logger.info(
                "solar_chat_tool_call",
                extra={"model": self._fallback_model, "latency_ms": duration_ms, "fallback": True},
            )
            return self._parse_tool_response(raw, self._fallback_model, fallback_used=True)
        except Exception as fallback_err:
            logger.error("solar_chat_tool_fallback_failed", extra={"error": str(fallback_err)})
            raise ModelUnavailableError(
                "Primary and fallback Gemini models are both unavailable."
            ) from fallback_err

    def send_tool_result(
        self,
        messages: list[dict[str, object]],
        model_name: str,
    ) -> GeminiToolResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        payload = {"contents": messages}
        raw = self._call_model_raw(model_name, payload)
        return self._parse_tool_response(raw, model_name, fallback_used=(model_name == self._fallback_model))

    def _call_model_raw(self, model_name: str, payload: dict[str, object]) -> dict[str, object]:
        endpoint = f"{self._base_url}/models/{model_name}:generateContent?key={self._api_key}"
        return self._request_executor(endpoint, payload, self._timeout)

    @staticmethod
    def _parse_tool_response(
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> GeminiToolResult:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response does not contain candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini response does not contain parts.")

        first_part = parts[0]
        function_call = first_part.get("functionCall")
        if isinstance(function_call, dict):
            return GeminiToolResult(
                function_call=FunctionCallRequest(
                    name=function_call.get("name", ""),
                    arguments=function_call.get("args", {}),
                ),
                text=None,
                model_used=model_used,
                fallback_used=fallback_used,
            )

        text_value = first_part.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return GeminiToolResult(
                function_call=None,
                text=text_value.strip(),
                model_used=model_used,
                fallback_used=fallback_used,
            )

        raise RuntimeError("Gemini response contains neither functionCall nor text.")

    def _call_model(self, model_name: str, prompt: str) -> str:
        endpoint = f"{self._base_url}/models/{model_name}:generateContent?key={self._api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        }

        raw_response = self._request_executor(endpoint, payload, self._timeout)
        candidates = raw_response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response does not contain candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini response does not contain text content.")

        text_value = parts[0].get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise RuntimeError("Gemini response text is empty.")

        return text_value.strip()

    @staticmethod
    def _execute_request(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as request_error:
            raise RuntimeError(f"Gemini request failed: {request_error}") from request_error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError("Gemini response body is not valid JSON.") from decode_error

        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response JSON root must be an object.")

        return parsed
