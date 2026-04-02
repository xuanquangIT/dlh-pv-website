import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable
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
        request_executor: Callable[[str, dict[str, object], float, dict[str, str] | None], dict[str, object]] | None = None,
    ) -> None:
        self._api_key = settings.gemini_api_key
        self._base_url = settings.gemini_base_url.rstrip("/")
        self._primary_model = settings.primary_model
        self._fallback_model = settings.fallback_model
        self._timeout = settings.request_timeout_seconds
        # Optional injection for testing; defaults to the real HTTP transport
        self._request_executor = request_executor or self._execute_request

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> GeminiGenerationResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        payload: dict[str, object] = {
            "contents": [{"parts": [{"text": prompt}]}],
        }

        def _call_and_extract(model: str) -> str:
            raw = self._call_model_raw(model, payload)
            parts = self._extract_parts(raw)
            text_value = parts[0].get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                raise RuntimeError("Gemini response text is empty.")
            return text_value.strip()

        text, model_used, fallback_used = self._with_model_fallback(
            _call_and_extract, "solar_chat_model_selection",
        )
        return GeminiGenerationResult(
            text=text,
            model_used=model_used,
            fallback_used=fallback_used,
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> GeminiToolResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        payload: dict[str, object] = {
            "contents": messages,
            "tools": [{"function_declarations": tools}],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }

        raw, model_used, fallback_used = self._with_model_fallback(
            lambda model: self._call_model_raw(model, payload),
            "solar_chat_tool_call",
        )
        return self._parse_tool_response(raw, model_used, fallback_used)

    def send_tool_result(
        self,
        messages: list[dict[str, object]],
        model_name: str,
    ) -> GeminiToolResult:
        if not self._api_key:
            raise ModelUnavailableError("Gemini API key is not configured.")

        payload: dict[str, object] = {"contents": messages}
        raw = self._call_model_raw(model_name, payload)
        return self._parse_tool_response(
            raw, model_name,
            fallback_used=(model_name == self._fallback_model),
        )

    # ------------------------------------------------------------------
    # Model fallback routing (eliminates D3 duplication)
    # ------------------------------------------------------------------

    def _with_model_fallback(
        self,
        action: Callable[[str], Any],
        log_event: str,
    ) -> tuple[Any, str, bool]:
        """Try primary model, then fallback.

        Returns (result, model_used, fallback_used).
        """
        models = [
            (self._primary_model, False),
            (self._fallback_model, True),
        ]
        last_error: Exception = RuntimeError("No models configured.")

        for model, is_fallback in models:
            started = time.perf_counter()
            try:
                result = action(model)
                duration_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    log_event,
                    extra={
                        "provider": "google_gemini",
                        "selected_model": model,
                        "latency_ms": duration_ms,
                        "fallback_used": is_fallback,
                    },
                )
                return result, model, is_fallback
            except Exception as err:
                last_error = err
                log_fn = logger.warning if not is_fallback else logger.error
                log_fn(
                    "%s_failed", log_event,
                    extra={
                        "provider": "google_gemini",
                        "model": model,
                        "error": str(err),
                    },
                )

        raise ModelUnavailableError(
            "Primary and fallback Gemini models are both unavailable."
        ) from last_error

    # ------------------------------------------------------------------
    # Response parsing (eliminates D4 duplication)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_parts(raw: dict[str, object]) -> list[dict[str, object]]:
        """Extract validated parts list from a Gemini API response."""
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response does not contain candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini response does not contain parts.")
        return parts

    @classmethod
    def _parse_tool_response(
        cls,
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> GeminiToolResult:
        parts = cls._extract_parts(raw)
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

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _call_model_raw(
        self, model_name: str, payload: dict[str, object],
    ) -> dict[str, object]:
        # C3: API key sent as header, never in URL
        endpoint = f"{self._base_url}/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        return self._request_executor(endpoint, payload, self._timeout, headers)

    @staticmethod
    def _execute_request(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as request_error:
            raise RuntimeError(
                f"Gemini request failed: {request_error}"
            ) from request_error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError(
                "Gemini response body is not valid JSON."
            ) from decode_error

        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response JSON root must be an object.")

        return parsed
