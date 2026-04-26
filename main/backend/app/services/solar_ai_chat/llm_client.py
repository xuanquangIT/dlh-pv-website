import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.settings import SolarChatSettings

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_TEMP_UNAVAILABLE_COOLDOWN_SECONDS = 45.0
_TOOL_CALL_DISABLE_TTL_SECONDS = 300.0


class ModelUnavailableError(RuntimeError):
    """Raised when both primary and fallback models are unavailable."""


class ToolCallNotSupportedError(ModelUnavailableError):
    """Raised when configured models reject tool-calling payloads."""


@dataclass(frozen=True)
class LLMGenerationResult:
    text: str
    model_used: str
    fallback_used: bool


@dataclass(frozen=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMToolResult:
    function_call: ToolCallRequest | None
    text: str | None
    model_used: str
    fallback_used: bool
    # Optional: all tool calls returned by the model in a single turn. When
    # the provider supports parallel tool calls (OpenAI, Anthropic), this
    # carries every call; `function_call` holds the first one for back-compat.
    function_calls: tuple[ToolCallRequest, ...] = ()


class LLMModelRouter:
    """Provider-agnostic LLM client with primary/fallback model routing.

    Supported API formats:
    - openai: OpenAI-compatible /chat/completions format
    - anthropic: Anthropic /messages format
    - gemini: Gemini generateContent format
    """

    def __init__(
        self,
        settings: SolarChatSettings,
        request_executor: Callable[[str, dict[str, object], float], dict[str, object]] | None = None,
    ) -> None:
        self._api_format = settings.resolved_llm_api_format
        self._api_key = settings.llm_api_key
        self._base_url = settings.resolved_llm_base_url.rstrip("/")
        self._primary_model = settings.primary_model
        self._fallback_model = settings.fallback_model
        self._timeout = settings.request_timeout_seconds
        self._anthropic_version = settings.llm_anthropic_version
        self._default_max_output_tokens = max(1, int(settings.llm_default_max_output_tokens))
        self._tool_call_max_output_tokens = max(1, int(settings.llm_tool_call_max_output_tokens))
        self._request_executor = request_executor
        # Models that produced invalid tool invocations are skipped for a short TTL window.
        self._tool_call_disabled_models: dict[str, float] = {}
        self._cooldown_until = 0.0

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> LLMGenerationResult:
        effective_max_tokens = self._resolve_max_output_tokens(
            value=max_output_tokens,
            default_value=self._default_max_output_tokens,
        )

        if self._api_format == "gemini":
            payload: dict[str, object] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": effective_max_tokens,
                    "temperature": temperature,
                },
            }

            def _call_and_extract(model: str) -> str:
                raw = self._call_model_raw(model, payload)
                parts = self._extract_gemini_parts(raw)
                text_value = parts[0].get("text")
                if not isinstance(text_value, str) or not text_value.strip():
                    raise RuntimeError("LLM response text is empty.")
                return text_value.strip()

        elif self._api_format == "anthropic":
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": effective_max_tokens,
                "temperature": temperature,
            }

            def _call_and_extract(model: str) -> str:
                raw = self._call_model_raw(model, payload)
                return self._extract_anthropic_text(raw)

        else:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
            }

            def _call_and_extract(model: str) -> str:
                raw = self._call_model_raw(model, payload)
                return self._extract_openai_text(raw)

        text, model_used, fallback_used = self._with_model_fallback(
            _call_and_extract,
            "solar_chat_model_selection",
        )
        return LLMGenerationResult(
            text=text,
            model_used=model_used,
            fallback_used=fallback_used,
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        require_function_call: bool = False,
        max_output_tokens: int | None = None,
    ) -> LLMToolResult:
        payload, skip_models = self._build_tool_generation_payload(
            messages,
            tools,
            require_function_call=require_function_call,
            max_output_tokens=max_output_tokens,
        )
        raw, model_used, fallback_used = self._with_model_fallback(
            lambda model: self._call_model_raw(model, payload),
            "solar_chat_tool_call",
            max_attempts=1,
            skip_models=skip_models,
        )
        return self._parse_tool_result_by_format(raw, model_used, fallback_used)

    def send_tool_result(
        self,
        messages: list[dict[str, object]],
        model_name: str,
        *,
        max_output_tokens: int | None = None,
    ) -> LLMToolResult:
        """Backward-compatible helper for tests and legacy call sites."""
        payload = self._build_tool_result_payload(
            messages,
            max_output_tokens=max_output_tokens,
        )
        raw = self._call_model_raw(model_name, payload)
        return self._parse_tool_result_by_format(raw, model_name, False)

    def _build_tool_generation_payload(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        require_function_call: bool = False,
        max_output_tokens: int | None = None,
    ) -> tuple[dict[str, object], set[str] | None]:
        effective_max_tokens = self._resolve_max_output_tokens(
            value=max_output_tokens,
            default_value=self._tool_call_max_output_tokens,
        )

        if self._api_format == "gemini":
            function_call_mode = "ANY" if require_function_call else "AUTO"
            # Gemini uses systemInstruction for system messages
            system_parts: list[dict[str, object]] = []
            contents: list[dict[str, object]] = []
            for msg in messages:
                if str(msg.get("role", "")) == "system":
                    parts = msg.get("parts")
                    if isinstance(parts, list):
                        system_parts.extend(parts)
                else:
                    contents.append(msg)
            payload: dict[str, object] = {
                "contents": contents,
                "tools": [{"function_declarations": tools}],
                "tool_config": {"function_calling_config": {"mode": function_call_mode}},
                "generationConfig": {
                    "maxOutputTokens": effective_max_tokens,
                    "temperature": 0.0,
                },
            }
            if system_parts:
                payload["systemInstruction"] = {"parts": system_parts}
            return payload, None

        if self._api_format == "anthropic":
            # Anthropic uses top-level "system" parameter for system messages
            system_texts: list[str] = []
            non_system_messages: list[dict[str, object]] = []
            for msg in messages:
                if str(msg.get("role", "")) == "system":
                    parts = msg.get("parts")
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                t = p.get("text")
                                if isinstance(t, str) and t.strip():
                                    system_texts.append(t)
                else:
                    non_system_messages.append(msg)
            payload = {
                "messages": self._convert_gemini_messages_to_anthropic(non_system_messages),
                "tools": self._convert_gemini_tools_to_anthropic(tools),
                "max_tokens": effective_max_tokens,
                "temperature": 0.0,
            }
            if system_texts:
                payload["system"] = "\n\n".join(system_texts)
            if require_function_call:
                payload["tool_choice"] = {"type": "any"}
            return payload, None

        tool_choice = "required" if require_function_call else "auto"
        payload = {
            "messages": self._convert_gemini_messages_to_openai(messages),
            "tools": self._convert_gemini_tools_to_openai(tools),
            "tool_choice": tool_choice,
            "temperature": 0.0,
            "max_tokens": effective_max_tokens,
        }
        return payload, self._active_tool_call_disabled_models()

    def _build_tool_result_payload(
        self,
        messages: list[dict[str, object]],
        *,
        max_output_tokens: int | None = None,
    ) -> dict[str, object]:
        effective_max_tokens = self._resolve_max_output_tokens(
            value=max_output_tokens,
            default_value=self._tool_call_max_output_tokens,
        )

        if self._api_format == "gemini":
            system_parts: list[dict[str, object]] = []
            contents: list[dict[str, object]] = []
            for msg in messages:
                if str(msg.get("role", "")) == "system":
                    parts = msg.get("parts")
                    if isinstance(parts, list):
                        system_parts.extend(parts)
                else:
                    contents.append(msg)
            result_payload: dict[str, object] = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": effective_max_tokens,
                    "temperature": 0.0,
                },
            }
            if system_parts:
                result_payload["systemInstruction"] = {"parts": system_parts}
            return result_payload
        if self._api_format == "anthropic":
            system_texts: list[str] = []
            non_system: list[dict[str, object]] = []
            for msg in messages:
                if str(msg.get("role", "")) == "system":
                    parts = msg.get("parts")
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                t = p.get("text")
                                if isinstance(t, str) and t.strip():
                                    system_texts.append(t)
                else:
                    non_system.append(msg)
            anthro_payload: dict[str, object] = {
                "messages": self._convert_gemini_messages_to_anthropic(non_system),
                "max_tokens": effective_max_tokens,
                "temperature": 0.0,
            }
            if system_texts:
                anthro_payload["system"] = "\n\n".join(system_texts)
            return anthro_payload
        return {
            "messages": self._convert_gemini_messages_to_openai(messages),
            "temperature": 0.0,
            "max_tokens": effective_max_tokens,
        }

    @staticmethod
    def _resolve_max_output_tokens(value: int | None, default_value: int) -> int:
        if value is None:
            return max(1, int(default_value))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(default_value))

    def _parse_tool_result_by_format(
        self,
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> LLMToolResult:
        if self._api_format == "gemini":
            return self._parse_gemini_tool_response(raw, model_used, fallback_used)
        if self._api_format == "anthropic":
            return self._parse_anthropic_tool_response(raw, model_used, fallback_used)
        return self._parse_openai_tool_response(raw, model_used, fallback_used)

    def _active_tool_call_disabled_models(self) -> set[str]:
        now = time.monotonic()
        expired_models = [
            model
            for model, disabled_until in self._tool_call_disabled_models.items()
            if disabled_until <= now
        ]
        for model in expired_models:
            self._tool_call_disabled_models.pop(model, None)
        return {
            model
            for model, disabled_until in self._tool_call_disabled_models.items()
            if disabled_until > now
        }

    def _with_model_fallback(
        self,
        action: Callable[[str], Any],
        log_event: str,
        max_attempts: int = 3,
        skip_models: set[str] | None = None,
    ) -> tuple[Any, str, bool]:
        now = time.monotonic()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            raise ModelUnavailableError(
                f"{self._api_format} temporarily unavailable (cooldown {remaining:.1f}s remaining)."
            )

        models = [(self._primary_model, False)]
        if self._fallback_model and self._fallback_model != self._primary_model:
            models.append((self._fallback_model, True))

        if skip_models:
            full_model_set = list(models)
            models = [
                (model_name, is_fallback)
                for model_name, is_fallback in models
                if model_name not in skip_models
            ]
            if not models:
                # All models are temporarily disabled; fall back to full set as a last resort.
                models = full_model_set

        if not models:
            raise ModelUnavailableError("No eligible models available for this request.")

        last_error: Exception = RuntimeError("No models configured.")
        saw_temporary_unavailable = False
        saw_tool_use_failed = False

        effective_attempts = max(1, int(max_attempts))
        for attempt in range(1, effective_attempts + 1):
            for model_index, (model, is_fallback) in enumerate(models):
                started = time.perf_counter()
                try:
                    result = action(model)
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    logger.info(
                        log_event,
                        extra={
                            "provider": self._api_format,
                            "selected_model": model,
                            "latency_ms": duration_ms,
                            "fallback_used": is_fallback,
                        },
                    )
                    return result, model, is_fallback
                except Exception as err:
                    last_error = err
                    error_text = str(err)
                    if (
                        log_event == "solar_chat_tool_call"
                        and self._api_format == "openai"
                        and _is_tool_use_failed_error(err)
                    ):
                        saw_tool_use_failed = True
                        disabled_until = time.monotonic() + _TOOL_CALL_DISABLE_TTL_SECONDS
                        self._tool_call_disabled_models[model] = disabled_until
                        logger.warning(
                            "Tool-call disabled for model=%s until %.1f due to provider invalid tool invocation.",
                            model,
                            disabled_until,
                        )
                    is_temporary_unavailable = _is_temporary_unavailable_error(err)
                    saw_temporary_unavailable = saw_temporary_unavailable or is_temporary_unavailable
                    error_kind = "rate_limited" if _is_rate_limit_error(err) else "request_failed"
                    has_more_models = model_index < (len(models) - 1)
                    if has_more_models:
                        logger.info(
                            "%s_failed provider=%s model=%s fallback_used=%s kind=%s will_try_next_model=True error=%s",
                            log_event,
                            self._api_format,
                            model,
                            is_fallback,
                            error_kind,
                            error_text,
                            extra={
                                "provider": self._api_format,
                                "model": model,
                                "error": error_text,
                                "kind": error_kind,
                                "fallback_used": is_fallback,
                            },
                        )
                    else:
                        logger.warning(
                            "%s_failed provider=%s model=%s fallback_used=%s kind=%s error=%s",
                            log_event,
                            self._api_format,
                            model,
                            is_fallback,
                            error_kind,
                            error_text,
                            extra={
                                "provider": self._api_format,
                                "model": model,
                                "error": error_text,
                                "kind": error_kind,
                                "fallback_used": is_fallback,
                            },
                        )

            last_err_str = str(last_error)
            if attempt < effective_attempts and (
                "429" in last_err_str
                or "503" in last_err_str
                or "529" in last_err_str
                or "Too Many" in last_err_str
            ):
                sleep_sec = 2.0 * attempt
                logger.warning(
                    "All models failed (429/503/529). Retrying %d/%d in %.1fs...",
                    attempt,
                    effective_attempts,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
                continue

            break

        if saw_temporary_unavailable:
            self._cooldown_until = time.monotonic() + _TEMP_UNAVAILABLE_COOLDOWN_SECONDS
            logger.warning(
                "%s temporary outage detected. Entering cooldown for %.0fs.",
                self._api_format,
                _TEMP_UNAVAILABLE_COOLDOWN_SECONDS,
            )

        if log_event == "solar_chat_tool_call" and saw_tool_use_failed and not saw_temporary_unavailable:
            raise ToolCallNotSupportedError(
                "Configured model rejected tool-calling payload."
            ) from last_error

        raise ModelUnavailableError("Primary and fallback models are both unavailable.") from last_error

    @staticmethod
    def _extract_gemini_parts(raw: dict[str, object]) -> list[dict[str, object]]:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("LLM response does not contain candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("LLM response does not contain parts.")
        return parts

    @staticmethod
    def _extract_openai_text(raw: dict[str, object]) -> str:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response does not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response text is empty.")
        return content.strip()

    @staticmethod
    def _extract_anthropic_text(raw: dict[str, object]) -> str:
        content = raw.get("content")
        if not isinstance(content, list) or not content:
            raise RuntimeError("LLM response does not contain content blocks.")
        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(str(t) for t in text_blocks if str(t).strip()).strip()
        if not text:
            raise RuntimeError("LLM response text is empty.")
        return text

    @classmethod
    def _parse_gemini_tool_response(
        cls,
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> LLMToolResult:
        parts = cls._extract_gemini_parts(raw)
        calls: list[ToolCallRequest] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                arguments = function_call.get("args", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append(
                    ToolCallRequest(
                        name=str(function_call.get("name", "")),
                        arguments=arguments,
                    )
                )
        if calls:
            return LLMToolResult(
                function_call=calls[0],
                text=None,
                model_used=model_used,
                fallback_used=fallback_used,
                function_calls=tuple(calls),
            )

        text_segments = [
            str(part.get("text", "")).strip()
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str) and str(part.get("text", "")).strip()
        ]
        if text_segments:
            return LLMToolResult(
                function_call=None,
                text="\n".join(text_segments),
                model_used=model_used,
                fallback_used=fallback_used,
            )

        raise RuntimeError("LLM response contains neither functionCall nor text.")

    @classmethod
    def _parse_openai_tool_response(
        cls,
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> LLMToolResult:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response does not contain choices.")

        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("LLM response message is invalid.")

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            parsed_calls: list[ToolCallRequest] = []
            for call in tool_calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = str(function.get("name", ""))
                if not name:
                    continue
                raw_arguments = function.get("arguments", "{}")
                arguments: dict[str, object]
                if isinstance(raw_arguments, str):
                    try:
                        loaded = json.loads(raw_arguments)
                        arguments = loaded if isinstance(loaded, dict) else {}
                    except json.JSONDecodeError:
                        arguments = {}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    arguments = {}
                parsed_calls.append(ToolCallRequest(name=name, arguments=arguments))

            if parsed_calls:
                return LLMToolResult(
                    function_call=parsed_calls[0],
                    text=None,
                    model_used=model_used,
                    fallback_used=fallback_used,
                    function_calls=tuple(parsed_calls),
                )

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return LLMToolResult(
                function_call=None,
                text=content.strip(),
                model_used=model_used,
                fallback_used=fallback_used,
            )

        raise RuntimeError("LLM response contains neither tool call nor text.")

    @classmethod
    def _parse_anthropic_tool_response(
        cls,
        raw: dict[str, object],
        model_used: str,
        fallback_used: bool,
    ) -> LLMToolResult:
        content = raw.get("content")
        if not isinstance(content, list) or not content:
            raise RuntimeError("LLM response does not contain content blocks.")

        tool_calls: list[ToolCallRequest] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name", ""))
                raw_input = block.get("input", {})
                arguments = raw_input if isinstance(raw_input, dict) else {}
                tool_calls.append(ToolCallRequest(name=name, arguments=arguments))
        if tool_calls:
            return LLMToolResult(
                function_call=tool_calls[0],
                text=None,
                model_used=model_used,
                fallback_used=fallback_used,
                function_calls=tuple(tool_calls),
            )

        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(str(t) for t in text_blocks if str(t).strip()).strip()
        if text:
            return LLMToolResult(
                function_call=None,
                text=text,
                model_used=model_used,
                fallback_used=fallback_used,
            )

        raise RuntimeError("LLM response contains neither tool call nor text.")

    _parse_tool_response = _parse_gemini_tool_response

    @staticmethod
    def _convert_gemini_tools_to_openai(
        tools: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name", "")),
                        "description": str(tool.get("description", "")),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return converted

    @staticmethod
    def _convert_gemini_tools_to_anthropic(
        tools: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            converted.append(
                {
                    "name": str(tool.get("name", "")),
                    "description": str(tool.get("description", "")),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _convert_gemini_messages_to_openai(
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = []
        tool_call_index = 0
        last_tool_call_id_by_name: dict[str, str] = {}

        for message in messages:
            role = str(message.get("role", "user"))
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue

                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    if role == "system":
                        openai_role = "system"
                    elif role == "model":
                        openai_role = "assistant"
                    else:
                        openai_role = "user"
                    converted.append(
                        {
                            "role": openai_role,
                            "content": text,
                        }
                    )
                    continue

                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    tool_call_index += 1
                    call_id = f"call_{tool_call_index}"
                    function_name = str(function_call.get("name", ""))
                    function_args = function_call.get("args", {})
                    if isinstance(function_args, str):
                        args_text = function_args
                    else:
                        try:
                            args_text = json.dumps(function_args, ensure_ascii=False)
                        except TypeError:
                            args_text = "{}"

                    converted.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": args_text,
                                    },
                                }
                            ],
                        }
                    )
                    last_tool_call_id_by_name[function_name] = call_id
                    continue

                function_response = part.get("functionResponse")
                if isinstance(function_response, dict):
                    function_name = str(function_response.get("name", ""))
                    function_payload = function_response.get("response", {})
                    try:
                        function_content = json.dumps(function_payload, ensure_ascii=False)
                    except TypeError:
                        function_content = str(function_payload)

                    tool_call_id = last_tool_call_id_by_name.get(function_name)
                    if not tool_call_id:
                        tool_call_index += 1
                        tool_call_id = f"call_orphan_{tool_call_index}"

                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": function_content,
                        }
                    )

        return converted

    @staticmethod
    def _convert_gemini_messages_to_anthropic(
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = []
        tool_call_index = 0
        last_tool_call_id_by_name: dict[str, str] = {}

        for message in messages:
            role = str(message.get("role", "user"))
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue

                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    converted.append(
                        {
                            "role": "assistant" if role == "model" else "user",
                            "content": [{"type": "text", "text": text}],
                        }
                    )
                    continue

                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    tool_call_index += 1
                    call_id = f"toolu_{tool_call_index}"
                    function_name = str(function_call.get("name", ""))
                    function_args = function_call.get("args", {})
                    if isinstance(function_args, dict):
                        input_payload = function_args
                    else:
                        input_payload = {}

                    converted.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call_id,
                                    "name": function_name,
                                    "input": input_payload,
                                }
                            ],
                        }
                    )
                    last_tool_call_id_by_name[function_name] = call_id
                    continue

                function_response = part.get("functionResponse")
                if isinstance(function_response, dict):
                    function_name = str(function_response.get("name", ""))
                    function_payload = function_response.get("response", {})
                    try:
                        function_content = json.dumps(function_payload, ensure_ascii=False)
                    except TypeError:
                        function_content = str(function_payload)

                    tool_use_id = last_tool_call_id_by_name.get(function_name)
                    if not tool_use_id:
                        tool_call_index += 1
                        tool_use_id = f"toolu_orphan_{tool_call_index}"

                    converted.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": function_content,
                                }
                            ],
                        }
                    )

        return converted

    def _call_model_raw(
        self,
        model_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        if self._api_format == "gemini":
            endpoint = f"{self._base_url}/models/{model_name}:generateContent"
            if self._request_executor is not None:
                return self._request_executor(endpoint, payload, self._timeout)
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["x-goog-api-key"] = self._api_key
            return self._execute_request(endpoint, payload, self._timeout, headers)

        if self._api_format == "anthropic":
            endpoint = f"{self._base_url}/messages"
            request_payload = dict(payload)
            request_payload["model"] = model_name
            if self._request_executor is not None:
                return self._request_executor(endpoint, request_payload, self._timeout)
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": self._anthropic_version,
            }
            if self._api_key:
                headers["x-api-key"] = self._api_key
            return self._execute_request(endpoint, request_payload, self._timeout, headers)

        endpoint = f"{self._base_url}/chat/completions"
        request_payload = dict(payload)
        request_payload["model"] = model_name
        if self._request_executor is not None:
            return self._request_executor(endpoint, request_payload, self._timeout)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # OpenRouter-specific optional headers for leaderboard attribution.
        # Harmless on other OpenAI-compatible endpoints (they ignore unknown headers).
        if "openrouter.ai" in self._base_url:
            import os
            referer = os.environ.get("SOLAR_CHAT_OPENROUTER_REFERER", "http://localhost:8000").strip()
            title = os.environ.get("SOLAR_CHAT_OPENROUTER_APP_NAME", "PV Lakehouse Solar Chat").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        return self._execute_request(endpoint, request_payload, self._timeout, headers)

    @staticmethod
    def _execute_request(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "pv-lakehouse-solar-ai-chat/1.0",
        }
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
        except HTTPError as request_error:
            error_body = ""
            try:
                raw_error_body = request_error.read()
                if raw_error_body:
                    error_body = raw_error_body.decode("utf-8", errors="replace")
            except Exception:
                error_body = ""

            trimmed_error_body = error_body.strip()
            if len(trimmed_error_body) > 1000:
                trimmed_error_body = trimmed_error_body[:1000] + "..."

            if trimmed_error_body:
                raise RuntimeError(
                    f"LLM request failed: HTTP {request_error.code} {request_error.reason}; body={trimmed_error_body}"
                ) from request_error

            raise RuntimeError(
                f"LLM request failed: HTTP {request_error.code} {request_error.reason}"
            ) from request_error
        except (URLError, TimeoutError) as request_error:
            raise RuntimeError(f"LLM request failed: {request_error}") from request_error

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError("LLM response body is not valid JSON.") from decode_error

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON root must be an object.")

        return parsed


# Backward compatible aliases to avoid breaking existing imports/tests.
GeminiGenerationResult = LLMGenerationResult
FunctionCallRequest = ToolCallRequest
GeminiToolResult = LLMToolResult
GeminiModelRouter = LLMModelRouter


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error)
    return (
        "HTTP Error 429" in text
        or "Too Many Requests" in text
        or "rate limit" in text.lower()
    )


def _is_service_unavailable_error(error: Exception) -> bool:
    text = str(error)
    return (
        "HTTP Error 503" in text
        or "HTTP Error 529" in text
        or "Service Unavailable" in text
        or "overloaded" in text.lower()
    )


def _is_temporary_unavailable_error(error: Exception) -> bool:
    if _is_tool_use_failed_error(error):
        return False
    return _is_rate_limit_error(error) or _is_service_unavailable_error(error)


def _is_tool_use_failed_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "tool_use_failed" in text
        or "failed to call a function" in text
        or "invalid tool invocation" in text
    )
