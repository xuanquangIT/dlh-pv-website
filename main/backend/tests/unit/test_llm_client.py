import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.settings import SolarChatSettings
from app.services.solar_ai_chat.llm_client import LLMModelRouter, ModelUnavailableError


class LLMModelRouterTests(unittest.TestCase):
    def _settings(self) -> SolarChatSettings:
        settings = SolarChatSettings()
        settings.llm_api_format = "gemini"
        settings.llm_api_key = "fake-key"
        settings.llm_base_url = "https://generativelanguage.googleapis.com/v1beta"
        settings.primary_model = "gemini-2.5-flash-lite"
        settings.fallback_model = "gemini-2.5-flash"
        settings.request_timeout_seconds = 1.0
        return settings

    def test_uses_fallback_when_primary_unavailable(self) -> None:
        calls: list[str] = []

        def executor(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
            calls.append(url)
            if "flash-lite" in url:
                raise RuntimeError("primary model unavailable")
            return {"candidates": [{"content": {"parts": [{"text": "fallback reply"}]}}]}

        router = LLMModelRouter(settings=self._settings(), request_executor=executor)
        result = router.generate("Test prompt")

        self.assertEqual(result.model_used, "gemini-2.5-flash")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.text, "fallback reply")
        self.assertEqual(len(calls), 2)

    def test_raises_when_both_models_fail(self) -> None:
        def executor(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
            raise RuntimeError("all models down")

        router = LLMModelRouter(settings=self._settings(), request_executor=executor)

        with self.assertRaises(ModelUnavailableError):
            router.generate("Test prompt")


class GeminiThoughtSignatureRoundTripTests(unittest.TestCase):
    """Gemini 3 returns HTTP 400 when a function_call part is echoed back
    without its `thoughtSignature`. The router must capture it during parse
    so the engine can restore it on the next turn."""

    def test_thought_signature_captured_from_part(self) -> None:
        from app.services.solar_ai_chat.engine import ChatEngine

        raw = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "recall_metric",
                                          "args": {"query": "energy"},
                                          "thoughtSignature": "sig-on-fc"}},
                    ]
                }
            }]
        }
        result = LLMModelRouter._parse_gemini_tool_response(raw, "gemini-3-test", False)
        fc = result.function_call
        assert fc is not None
        self.assertEqual(fc.provider_metadata, {"thoughtSignature": "sig-on-fc"})

        echoed = ChatEngine._format_assistant_tool_calls([fc])
        self.assertEqual(echoed["parts"][0]["thought_signature"], "sig-on-fc")

    def test_no_signature_when_provider_did_not_supply(self) -> None:
        from app.services.solar_ai_chat.engine import ChatEngine

        raw = {
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "f", "args": {}}}]
                }
            }]
        }
        result = LLMModelRouter._parse_gemini_tool_response(raw, "gemini-2.5", False)
        fc = result.function_call
        assert fc is not None
        self.assertIsNone(fc.provider_metadata)

        echoed = ChatEngine._format_assistant_tool_calls([fc])
        self.assertNotIn("thought_signature", echoed["parts"][0])


if __name__ == "__main__":
    unittest.main()
