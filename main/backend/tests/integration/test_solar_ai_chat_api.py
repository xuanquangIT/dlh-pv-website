import sys
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app
from app.api.solar_ai_chat import get_solar_ai_chat_service
from app.core.settings import SolarChatSettings
from app.repositories.solar_chat_repository import SolarChatRepository
from app.services.solar_ai_chat_service import SolarAIChatService
from app.services.solar_chat_intent_service import VietnameseIntentService


def get_test_solar_ai_chat_service() -> SolarAIChatService:
    settings = SolarChatSettings()
    settings.gemini_api_key = None
    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=VietnameseIntentService(),
        model_router=None,
    )


class SolarAIChatApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = create_app()
        app.dependency_overrides[get_solar_ai_chat_service] = get_test_solar_ai_chat_service
        cls.client = TestClient(app)

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


if __name__ == "__main__":
    unittest.main()
