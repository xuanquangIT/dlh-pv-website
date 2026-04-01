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
