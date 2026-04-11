import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.llm_client import LLMToolResult
from app.services.solar_ai_chat.prompt_builder import build_fallback_summary


class FacilityFallbackSummaryTests(unittest.TestCase):
    def test_facility_summary_includes_highest_capacity_station(self) -> None:
        metrics = {
            "facility_count": 2,
            "facilities": [
                {
                    "facility_name": "Alpha",
                    "location_lat": -35.0,
                    "location_lng": 149.0,
                    "total_capacity_mw": 120.5,
                    "timezone_name": "Australia/Eastern",
                    "timezone_utc_offset": "UTC+10:00",
                },
                {
                    "facility_name": "Beta",
                    "location_lat": -34.0,
                    "location_lng": 150.0,
                    "total_capacity_mw": 320.0,
                    "timezone_name": "Australia/Eastern",
                    "timezone_utc_offset": "UTC+10:00",
                },
            ],
        }

        summary = build_fallback_summary(ChatTopic.FACILITY_INFO, metrics, [])

        self.assertIn("Beta (320.0 MW)", summary)
        self.assertIn("Múi giờ các trạm hiện tại", summary)
        self.assertIn("Australia/Eastern (UTC+10:00)", summary)


class ToolPathGuardTests(unittest.TestCase):
    def test_text_without_tools_for_data_question_forces_deterministic_fallback(self) -> None:
        repository = MagicMock()
        repository.fetch_topic_metrics.return_value = (
            {
                "facility_count": 2,
                "facilities": [
                    {
                        "facility_name": "Alpha",
                        "location_lat": -35.0,
                        "location_lng": 149.0,
                        "total_capacity_mw": 120.5,
                    },
                    {
                        "facility_name": "Beta",
                        "location_lat": -34.0,
                        "location_lng": 150.0,
                        "total_capacity_mw": 320.0,
                    },
                ],
            },
            [{"layer": "Gold", "dataset": "gold.dim_facility", "data_source": "databricks"}],
        )

        intent_service = MagicMock()
        intent_service.detect_intent.return_value = SimpleNamespace(
            topic=ChatTopic.FACILITY_INFO,
            confidence=0.91,
        )

        model_router = MagicMock()
        model_router.generate_with_tools.return_value = LLMToolResult(
            function_call=None,
            text="Tram lon nhat the gioi la dap thuy dien ...",
            model_used="claude-haiku-4.5",
            fallback_used=False,
        )

        service = SolarAIChatService(
            repository=repository,
            intent_service=intent_service,
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

        self.assertEqual(response.topic, ChatTopic.FACILITY_INFO)
        self.assertEqual(response.model_used, "deterministic-summary")
        self.assertTrue(response.fallback_used)
        self.assertIn("Beta (320.0 MW)", response.answer)
        repository.fetch_topic_metrics.assert_called_once_with(ChatTopic.FACILITY_INFO)


if __name__ == "__main__":
    unittest.main()
