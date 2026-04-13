"""Tests for agentic tool loop guardrails and answer synthesis.

Updated for the new ReAct-based agentic architecture:
- LLM text without tool calls is accepted as the final answer
- RBAC denies disallowed tools and returns an error to the LLM
- build_data_only_summary() is used when LLM is unavailable
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.solar_ai_chat import ChatTopic


def _general_intent_service() -> MagicMock:
    """Intent service mock that returns GENERAL (no pre-fetch)."""
    svc = MagicMock()
    svc.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.GENERAL, confidence=0.0
    )
    return svc

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.llm_client import LLMToolResult, ToolCallRequest


class AgenticLoopAnswerTests(unittest.TestCase):
    def test_llm_text_response_is_used_as_final_answer(self) -> None:
        """When LLM returns text without any tool call, that text is the answer."""
        model_router = MagicMock()
        model_router.generate_with_tools.return_value = LLMToolResult(
            function_call=None,
            text="Darlington Point la tram lon nhat voi 324 MW.",
            model_used="gemini-flash",
            fallback_used=False,
        )

        service = SolarAIChatService(
            repository=MagicMock(),
            intent_service=_general_intent_service(),
            model_router=model_router,
            history_repository=None,
        )

        response = service.handle_query(
            SolarChatRequest(
                message="Tram co cong suat lon nhat la tram gi",
                role=ChatRole.DATA_ENGINEER,
                session_id=None,
            )
        )

        self.assertEqual(response.answer, "Darlington Point la tram lon nhat voi 324 MW.")
        self.assertEqual(response.model_used, "gemini-flash")
        self.assertFalse(response.fallback_used)

    def test_tool_call_followed_by_answer(self) -> None:
        """LLM calls a tool then synthesises; both are reflected in the response."""
        repository = MagicMock()
        repository.fetch_topic_metrics.return_value = (
            {"facility_count": 2, "facilities": [{"facility_name": "Alpha", "total_capacity_mw": 100.0}]},
            [{"layer": "Gold", "dataset": "gold.dim_facility", "data_source": "databricks"}],
        )

        model_router = MagicMock()
        model_router.generate_with_tools.side_effect = [
            LLMToolResult(
                function_call=ToolCallRequest(name="get_facility_info", arguments={}),
                text=None,
                model_used="gemini-flash",
                fallback_used=False,
            ),
            LLMToolResult(
                function_call=None,
                text="He thong co 2 tram, lon nhat la Alpha voi 100 MW.",
                model_used="gemini-flash",
                fallback_used=False,
            ),
        ]

        service = SolarAIChatService(
            repository=repository,
            intent_service=_general_intent_service(),
            model_router=model_router,
            history_repository=None,
        )

        response = service.handle_query(
            SolarChatRequest(
                message="Cac tram hien tai la gi?",
                role=ChatRole.DATA_ENGINEER,
                session_id=None,
            )
        )

        self.assertIn("Alpha", response.answer)
        self.assertEqual(response.model_used, "gemini-flash")
        self.assertFalse(response.fallback_used)
        # Tool was called then synthesised
        self.assertEqual(model_router.generate_with_tools.call_count, 2)
        repository.fetch_topic_metrics.assert_called_once()


class RbacGuardTests(unittest.TestCase):
    def test_denied_tool_appends_error_and_llm_continues(self) -> None:
        """When a tool is denied by RBAC, an error is returned to the LLM
        which should then answer without that tool's data."""
        model_router = MagicMock()
        model_router.generate_with_tools.side_effect = [
            # LLM requests a tool that viewer role does not have access to
            LLMToolResult(
                function_call=ToolCallRequest(name="get_pipeline_status", arguments={}),
                text=None,
                model_used="gemini-flash",
                fallback_used=False,
            ),
            # LLM continues after receiving access-denied error
            LLMToolResult(
                function_call=None,
                text="Xin loi, toi khong co quyen truy cap du lieu nay.",
                model_used="gemini-flash",
                fallback_used=False,
            ),
        ]

        service = SolarAIChatService(
            repository=MagicMock(),
            intent_service=_general_intent_service(),
            model_router=model_router,
            history_repository=None,
        )

        response = service.handle_query(
            SolarChatRequest(
                message="Pipeline status la gi?",
                role=ChatRole.ML_ENGINEER,  # ML_ENGINEER does not have get_pipeline_status
                session_id=None,
            )
        )

        self.assertIn("quyen", response.answer.lower())
        self.assertEqual(model_router.generate_with_tools.call_count, 2)


if __name__ == "__main__":
    unittest.main()
