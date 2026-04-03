import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app
from app.api.solar_ai_chat.routes import get_solar_ai_chat_service, _get_history_repository
from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat import ChatHistoryRepository
from app.repositories.solar_ai_chat import SolarChatRepository
from app.services.solar_ai_chat import SolarAIChatService
from app.services.solar_ai_chat import VietnameseIntentService

_TEST_HISTORY_DIR: Path | None = None


def _get_test_history_repository() -> ChatHistoryRepository:
    global _TEST_HISTORY_DIR
    if _TEST_HISTORY_DIR is None:
        _TEST_HISTORY_DIR = Path(tempfile.mkdtemp(prefix="chat_history_test_"))
    return ChatHistoryRepository(storage_dir=_TEST_HISTORY_DIR)


def get_test_solar_ai_chat_service() -> SolarAIChatService:
    settings = SolarChatSettings()
    settings.gemini_api_key = None
    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=VietnameseIntentService(),
        model_router=None,
        history_repository=_get_test_history_repository(),
    )


class SolarAIChatApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = create_app()
        app.dependency_overrides[get_solar_ai_chat_service] = get_test_solar_ai_chat_service
        app.dependency_overrides[_get_history_repository] = _get_test_history_repository
        cls.client = TestClient(app)
        cls._trino_patcher = patch(
            "app.repositories.solar_ai_chat.chat_repository.SolarChatRepository._execute_query",
            side_effect=ConnectionError("Trino not available in tests"),
        )
        cls._trino_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._trino_patcher.stop()

    def test_returns_system_overview_for_viewer(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi tong quan he thong, san luong va r-squared",
                "role": "viewer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "system_overview")
        self.assertEqual(body["role"], "viewer")
        self.assertGreater(len(body["sources"]), 0)
        self.assertIn("model_used", body)

    def test_enforces_rbac_for_pipeline_status(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi trang thai pipeline va ETA",
                "role": "viewer",
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("not allowed", response.json()["detail"])

    def test_returns_lowest_aqi_station_for_date(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem AQI nao thap nhat cac tram vao ngay 5/1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "data_quality_issues")
        self.assertEqual(body["key_metrics"]["query_date"], "2026-01-05")
        self.assertIn("lowest_station", body["key_metrics"])
        self.assertIn("lowest_aqi_value", body["key_metrics"])

    def test_returns_highest_aqi_station_for_date(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem AQI nao cao nhat cac tram vao ngay 2/1/2026",
                "role": "data_engineer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "data_quality_issues")
        self.assertEqual(body["key_metrics"]["query_date"], "2026-01-02")
        self.assertIn("highest_station", body["key_metrics"])
        self.assertIn("highest_aqi_value", body["key_metrics"])

    def test_returns_highest_energy_station_for_date(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem energy cao nhat cac tram vao ngay 2/1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "energy_performance")
        self.assertEqual(body["key_metrics"]["query_date"], "2026-01-02")
        self.assertEqual(body["key_metrics"]["extreme_metric"], "energy")
        self.assertIn("highest_station", body["key_metrics"])
        self.assertIn("highest_energy_mwh", body["key_metrics"])

    def test_returns_lowest_weather_station_for_date(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem nhiet do thap nhat cac tram vao ngay 2/1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "energy_performance")
        self.assertEqual(body["key_metrics"]["query_date"], "2026-01-02")
        self.assertEqual(body["key_metrics"]["extreme_metric"], "weather")
        self.assertIn("lowest_station", body["key_metrics"])
        self.assertIn("lowest_weather_value", body["key_metrics"])

    def test_supports_hour_timeframe_for_aqi(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem AQI cao nhat theo 1 gio vao ngay 2/1/2026",
                "role": "data_engineer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "aqi")
        self.assertEqual(body["key_metrics"]["timeframe"], "hour")
        self.assertIn("period_label", body["key_metrics"])

    def test_supports_24h_timeframe_for_energy(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem energy thap nhat theo 24 gio vao ngay 2/1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "energy")
        self.assertEqual(body["key_metrics"]["timeframe"], "24h")
        self.assertIn("period_label", body["key_metrics"])

    def test_supports_week_timeframe_for_energy(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem energy cao nhat theo tuan ngay 2/1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "energy")
        self.assertEqual(body["key_metrics"]["timeframe"], "week")
        self.assertIn("period_label", body["key_metrics"])

    def test_supports_month_timeframe_for_weather(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem nhiet do cao nhat theo thang 1/2026",
                "role": "data_analyst",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "weather")
        self.assertEqual(body["key_metrics"]["timeframe"], "month")
        self.assertIn("period_label", body["key_metrics"])

    def test_resolves_solar_radiation_metric_for_hourly_query(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem buc xa mat troi cao nhat cua tram trong 1 gio vao ngay 5/2/2026",
                "role": "data_engineer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "weather")
        self.assertEqual(body["key_metrics"]["weather_metric"], "shortwave_radiation")
        self.assertEqual(body["key_metrics"]["timeframe"], "hour")

    def test_keeps_wind_metric_when_user_asks_for_wind(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem toc do gio cao nhat cua tram trong 1 gio vao ngay 5/2/2026",
                "role": "data_engineer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "weather")
        self.assertEqual(body["key_metrics"]["weather_metric"], "wind_speed_10m")
        self.assertEqual(body["key_metrics"]["timeframe"], "hour")

    def test_hourly_solar_radiation_highest_and_lowest_are_distinct(self) -> None:
        highest_response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem buc xa mat troi cao nhat cua tram trong 1 gio vao ngay 5/2/2025",
                "role": "data_engineer",
            },
        )
        lowest_response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem buc xa mat troi thap nhat cua tram trong 1 gio vao ngay 5/2/2025",
                "role": "data_engineer",
            },
        )

        self.assertEqual(highest_response.status_code, 200)
        self.assertEqual(lowest_response.status_code, 200)

        highest_body = highest_response.json()
        lowest_body = lowest_response.json()

        highest_value = highest_body["key_metrics"]["highest_weather_value"]
        lowest_value = lowest_body["key_metrics"]["lowest_weather_value"]

        self.assertEqual(highest_body["key_metrics"]["weather_metric"], "shortwave_radiation")
        self.assertEqual(lowest_body["key_metrics"]["weather_metric"], "shortwave_radiation")
        self.assertGreater(highest_value, lowest_value)
        self.assertGreater(highest_value, 0.0)

    def test_supports_specific_hour_phrase_for_weather_query(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi buc xa mat troi cao nhat cua tram vao luc 5 gio chieu vao ngay 5/2/2025",
                "role": "viewer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["topic"], "energy_performance")
        self.assertEqual(body["key_metrics"]["extreme_metric"], "weather")
        self.assertEqual(body["key_metrics"]["weather_metric"], "shortwave_radiation")
        self.assertEqual(body["key_metrics"]["timeframe"], "hour")
        self.assertEqual(body["key_metrics"]["specific_hour"], 17)
        self.assertEqual(body["key_metrics"]["query_date"], "2025-02-05")
        self.assertEqual(body["key_metrics"]["period_label"], "2025-02-05T17:00")

    def test_supports_year_timeframe_for_aqi(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi xem AQI thap nhat theo nam 2026",
                "role": "data_engineer",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["key_metrics"]["extreme_metric"], "aqi")
        self.assertEqual(body["key_metrics"]["timeframe"], "year")
        self.assertIn("period_label", body["key_metrics"])

    def test_admin_can_access_all_topics(self) -> None:
        admin_queries = [
            {"message": "Cho toi tong quan he thong", "role": "admin"},
            {"message": "Top nha may hieu suat cao nhat", "role": "admin"},
            {"message": "So sanh mo hinh GBT-v4.2 voi v4.1", "role": "admin"},
            {"message": "Cho toi trang thai pipeline va ETA", "role": "admin"},
            {"message": "Du bao 72 gio va khoang tin cay", "role": "admin"},
            {"message": "Cho toi danh sach co so diem thap va nguyen nhan", "role": "admin"},
        ]
        for query in admin_queries:
            response = self.client.post("/solar-ai-chat/query", json=query)
            self.assertEqual(
                response.status_code,
                200,
                msg=f"Admin should access topic for: {query['message']}",
            )

    def test_data_engineer_denied_ml_model(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi tham so mo hinh GBT-v4.2 va so sanh voi v4.1",
                "role": "data_engineer",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("not allowed", response.json()["detail"])

    def test_ml_engineer_denied_pipeline_status(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi trang thai pipeline, tien do cac stage va ETA",
                "role": "ml_engineer",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("not allowed", response.json()["detail"])

    def test_viewer_denied_data_quality_issues(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi danh sach co so diem chat luong thap va nguyen nhan",
                "role": "viewer",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("not allowed", response.json()["detail"])

    def test_returns_warning_message_when_model_router_absent(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi tong quan he thong hom nay",
                "role": "viewer",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_used"], "deterministic-summary")
        self.assertIsNotNone(body["warning_message"])
        self.assertIn("Gemini API key", body["warning_message"])

    def test_standard_queries_under_four_seconds(self) -> None:
        standard_queries = [
            {
                "message": "Cho toi tong quan he thong hom nay",
                "role": "viewer",
            },
            {
                "message": "Du bao 72 gio va khoang tin cay theo ngay",
                "role": "data_analyst",
            },
            {
                "message": "So sanh mo hinh GBT-v4.2 voi v4.1",
                "role": "ml_engineer",
            },
            {
                "message": "Cho toi danh sach co so diem thap va nguyen nhan",
                "role": "data_engineer",
            },
        ]

        for query in standard_queries:
            started = time.perf_counter()
            response = self.client.post("/solar-ai-chat/query", json=query)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            self.assertEqual(response.status_code, 200)
            body = response.json()

            self.assertLess(
                body["latency_ms"],
                4000,
                msg=f"Service latency exceeded 4s for query: {query['message']}",
            )
            self.assertLess(
                elapsed_ms,
                4000,
                msg=f"End-to-end latency exceeded 4s for query: {query['message']}",
            )


class ChatSessionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global _TEST_HISTORY_DIR
        _TEST_HISTORY_DIR = Path(tempfile.mkdtemp(prefix="chat_session_api_test_"))
        app = create_app()
        app.dependency_overrides[get_solar_ai_chat_service] = get_test_solar_ai_chat_service
        app.dependency_overrides[_get_history_repository] = _get_test_history_repository
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        global _TEST_HISTORY_DIR
        if _TEST_HISTORY_DIR and _TEST_HISTORY_DIR.exists():
            shutil.rmtree(_TEST_HISTORY_DIR, ignore_errors=True)
            _TEST_HISTORY_DIR = None

    def test_create_and_list_sessions(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/sessions",
            json={"role": "viewer", "title": "Test session"},
        )
        self.assertEqual(response.status_code, 201)
        session = response.json()
        self.assertEqual(session["title"], "Test session")
        self.assertEqual(session["role"], "viewer")
        self.assertEqual(session["message_count"], 0)

        list_response = self.client.get("/solar-ai-chat/sessions")
        self.assertEqual(list_response.status_code, 200)
        sessions = list_response.json()
        session_ids = [s["session_id"] for s in sessions]
        self.assertIn(session["session_id"], session_ids)

    def test_get_session_detail(self) -> None:
        create = self.client.post(
            "/solar-ai-chat/sessions",
            json={"role": "admin", "title": "Detail test"},
        )
        session_id = create.json()["session_id"]

        detail = self.client.get(f"/solar-ai-chat/sessions/{session_id}")
        self.assertEqual(detail.status_code, 200)
        body = detail.json()
        self.assertEqual(body["session_id"], session_id)
        self.assertEqual(body["messages"], [])

    def test_get_nonexistent_session_returns_404(self) -> None:
        response = self.client.get("/solar-ai-chat/sessions/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_delete_session(self) -> None:
        create = self.client.post(
            "/solar-ai-chat/sessions",
            json={"role": "viewer", "title": "To delete"},
        )
        session_id = create.json()["session_id"]

        delete = self.client.delete(f"/solar-ai-chat/sessions/{session_id}")
        self.assertEqual(delete.status_code, 204)

        get = self.client.get(f"/solar-ai-chat/sessions/{session_id}")
        self.assertEqual(get.status_code, 404)

    def test_delete_nonexistent_session_returns_404(self) -> None:
        response = self.client.delete("/solar-ai-chat/sessions/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_query_persists_messages_in_session(self) -> None:
        create = self.client.post(
            "/solar-ai-chat/sessions",
            json={"role": "viewer", "title": "Persist test"},
        )
        session_id = create.json()["session_id"]

        self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi tong quan he thong",
                "role": "viewer",
                "session_id": session_id,
            },
        )

        detail = self.client.get(f"/solar-ai-chat/sessions/{session_id}")
        body = detail.json()
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["messages"][0]["sender"], "user")
        self.assertEqual(body["messages"][1]["sender"], "assistant")

    def test_fork_session_copies_messages(self) -> None:
        create = self.client.post(
            "/solar-ai-chat/sessions",
            json={"role": "admin", "title": "Original"},
        )
        session_id = create.json()["session_id"]

        self.client.post(
            "/solar-ai-chat/query",
            json={
                "message": "Cho toi tong quan he thong",
                "role": "admin",
                "session_id": session_id,
            },
        )

        fork = self.client.post(
            f"/solar-ai-chat/sessions/{session_id}/fork",
            json={"title": "Forked session", "role": "admin"},
        )
        self.assertEqual(fork.status_code, 201)
        fork_body = fork.json()
        self.assertEqual(fork_body["title"], "Forked session")
        self.assertEqual(fork_body["message_count"], 2)

        fork_detail = self.client.get(f"/solar-ai-chat/sessions/{fork_body['session_id']}")
        self.assertEqual(len(fork_detail.json()["messages"]), 2)

    def test_fork_nonexistent_session_returns_404(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/sessions/nonexistent/fork",
            json={"title": "Fork"},
        )
        self.assertEqual(response.status_code, 404)


class FunctionCallingIntegrationTests(unittest.TestCase):
    """Integration tests for the Gemini Function Calling flow with mocked Gemini API."""

    @classmethod
    def setUpClass(cls) -> None:
        app = create_app()
        app.dependency_overrides[_get_history_repository] = _get_test_history_repository
        cls.client = TestClient(app)
        cls._trino_patcher = patch(
            "app.repositories.solar_ai_chat.chat_repository.SolarChatRepository._execute_query",
            side_effect=ConnectionError("Trino not available in tests"),
        )
        cls._trino_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._trino_patcher.stop()

    def _make_service_with_mock_router(self, responses: list[dict]) -> SolarAIChatService:
        from app.services.solar_ai_chat.gemini_client import GeminiModelRouter
        call_index = {"n": 0}

        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            idx = call_index["n"]
            call_index["n"] += 1
            return responses[idx]

        settings = SolarChatSettings()
        settings.gemini_api_key = "test-key-fc"
        router = GeminiModelRouter(settings=settings, request_executor=mock_executor)

        return SolarAIChatService(
            repository=SolarChatRepository(settings=SolarChatSettings()),
            intent_service=VietnameseIntentService(),
            model_router=router,
            history_repository=_get_test_history_repository(),
        )

    def test_tool_call_flow_system_overview(self) -> None:
        gemini_responses = [
            {
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
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "He thong dang hoat dong tot voi san luong 4000 MWh."}]
                    }
                }]
            },
        ]
        service = self._make_service_with_mock_router(gemini_responses)
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Cho toi tong quan he thong", "role": "admin"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["topic"], "system_overview")
        self.assertIn("4000 MWh", body["answer"])
        self.assertIn("gemini", body["model_used"])
        self.assertEqual(body["intent_confidence"], 0.95)

    def test_tool_call_flow_extreme_aqi(self) -> None:
        gemini_responses = [
            {
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
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "AQI cao nhat hom nay la 120 tai tram Facility_01."}]
                    }
                }]
            },
        ]
        service = self._make_service_with_mock_router(gemini_responses)
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "AQI cao nhat hom nay", "role": "admin"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["topic"], "data_quality_issues")
        self.assertIn("120", body["answer"])

    def test_direct_text_answer_without_tool_call(self) -> None:
        gemini_responses = [
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Xin chao! Toi la tro ly Solar AI Chat."}]
                    }
                }]
            },
        ]
        service = self._make_service_with_mock_router(gemini_responses)
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Xin chao!", "role": "viewer"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Xin chao", body["answer"])
        self.assertEqual(body["intent_confidence"], 0.85)

    def test_rbac_enforced_via_tool_call(self) -> None:
        gemini_responses = [
            {
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
            },
        ]
        service = self._make_service_with_mock_router(gemini_responses)
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "AQI cao nhat", "role": "viewer"},
        )
        self.assertEqual(response.status_code, 403)

    def test_falls_back_to_regex_when_gemini_unavailable(self) -> None:
        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            raise RuntimeError("Service unavailable")

        settings = SolarChatSettings()
        settings.gemini_api_key = "test-key"
        from app.services.solar_ai_chat.gemini_client import GeminiModelRouter
        router = GeminiModelRouter(settings=settings, request_executor=mock_executor)

        service = SolarAIChatService(
            repository=SolarChatRepository(settings=SolarChatSettings()),
            intent_service=VietnameseIntentService(),
            model_router=router,
            history_repository=_get_test_history_repository(),
        )
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Cho toi tong quan he thong", "role": "admin"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["topic"], "system_overview")
        self.assertIn("warning_message", body)


class RagApiIntegrationTests(unittest.TestCase):
    """Integration tests for RAG document ingestion and stats endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        app = create_app()
        app.dependency_overrides[_get_history_repository] = _get_test_history_repository
        cls.client = TestClient(app)

    def test_document_stats_returns_empty_when_no_pg(self) -> None:
        from app.api.solar_ai_chat.routes import _get_vector_repository
        self.client.app.dependency_overrides[_get_vector_repository] = lambda: None
        response = self.client.get("/solar-ai-chat/documents/stats")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_chunks"], 0)
        self.assertEqual(body["by_doc_type"], {})

    def test_ingest_rejects_missing_api_key(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/documents/ingest",
            json={"file_path": "/nonexistent.txt", "doc_type": "incident_report"},
        )
        self.assertIn(response.status_code, [400, 503])

    def test_ingest_rejects_invalid_doc_type(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/documents/ingest",
            json={"file_path": "/tmp/test.txt", "doc_type": "invalid_type"},
        )
        self.assertEqual(response.status_code, 422)

    def test_delete_document_returns_503_when_no_pg(self) -> None:
        from app.api.solar_ai_chat.routes import _get_vector_repository
        self.client.app.dependency_overrides[_get_vector_repository] = lambda: None
        response = self.client.delete("/solar-ai-chat/documents/nonexistent.txt")
        self.assertEqual(response.status_code, 503)

    def test_search_documents_tool_via_query_when_rag_unavailable(self) -> None:
        """When RAG infra is not configured, search_documents tool returns 'not configured' message."""
        gemini_responses = [
            {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "name": "search_documents",
                                "args": {"query": "su co thang 3"},
                            }
                        }]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Xin loi, chuc nang tim kiem tai lieu chua duoc cau hinh."}]
                    }
                }]
            },
        ]
        from app.services.solar_ai_chat.gemini_client import GeminiModelRouter
        call_index = {"n": 0}

        def mock_executor(url: str, payload: dict, timeout: float) -> dict:
            idx = call_index["n"]
            call_index["n"] += 1
            return gemini_responses[idx]

        settings = SolarChatSettings()
        settings.gemini_api_key = "test-key-rag"
        router = GeminiModelRouter(settings=settings, request_executor=mock_executor)

        service = SolarAIChatService(
            repository=SolarChatRepository(settings=SolarChatSettings()),
            intent_service=VietnameseIntentService(),
            model_router=router,
            history_repository=_get_test_history_repository(),
            vector_repo=None,
            embedding_client=None,
        )
        from app.api.solar_ai_chat.routes import get_solar_ai_chat_service
        self.client.app.dependency_overrides[get_solar_ai_chat_service] = lambda: service

        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Tim tai lieu su co thang 3", "role": "admin"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("tai lieu", body["answer"].lower())


if __name__ == "__main__":
    unittest.main()
