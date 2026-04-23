import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatTopic
from app.services.solar_ai_chat.intent_service import VietnameseIntentService, normalize_vietnamese_text


class VietnameseIntentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = VietnameseIntentService()

    def test_detects_system_overview_for_vietnamese_message(self) -> None:
        result = self.service.detect_intent("Cho toi tong quan he thong va san luong hien tai")
        self.assertEqual(result.topic, ChatTopic.SYSTEM_OVERVIEW)
        self.assertGreaterEqual(result.confidence, 0.5)

    def test_detects_system_overview_for_english_message(self) -> None:
        result = self.service.detect_intent("Now give me the current system overview")
        self.assertEqual(result.topic, ChatTopic.SYSTEM_OVERVIEW)

    def test_detects_system_overview_for_overview_r_squared_message(self) -> None:
        result = self.service.detect_intent(
            "Give me an overview of the entire solar system with facility count, output, and quality score"
        )
        self.assertEqual(result.topic, ChatTopic.SYSTEM_OVERVIEW)

    def test_detects_forecast_72h_topic(self) -> None:
        result = self.service.detect_intent("Du bao 72 gio va khoang tin cay")
        self.assertEqual(result.topic, ChatTopic.FORECAST_72H)

    def test_detects_forecast_72h_for_english_message(self) -> None:
        result = self.service.detect_intent("Show me expected energy production for next 72 hours")
        self.assertEqual(result.topic, ChatTopic.FORECAST_72H)

    def test_detects_data_quality_aqi_question(self) -> None:
        result = self.service.detect_intent("AQI nao thap nhat cac tram vao ngay 5/1/2026")
        self.assertEqual(result.topic, ChatTopic.DATA_QUALITY_ISSUES)

    def test_detects_data_quality_for_english_low_score_alerts(self) -> None:
        result = self.service.detect_intent("Any low-score facilities or data quality alerts today")
        self.assertEqual(result.topic, ChatTopic.DATA_QUALITY_ISSUES)

    def test_detects_data_quality_highest_aqi_question(self) -> None:
        result = self.service.detect_intent("AQI nao cao nhat cac tram vao ngay 2/1/2026")
        self.assertEqual(result.topic, ChatTopic.DATA_QUALITY_ISSUES)

    def test_detects_energy_extreme_question(self) -> None:
        result = self.service.detect_intent("San luong energy cao nhat cac tram vao ngay 2/1/2026")
        self.assertEqual(result.topic, ChatTopic.ENERGY_PERFORMANCE)

    def test_detects_energy_performance_for_best_performance_question(self) -> None:
        result = self.service.detect_intent("Tram co performance tot nhat la tram nao")
        self.assertEqual(result.topic, ChatTopic.ENERGY_PERFORMANCE)

    def test_detects_energy_performance_for_vietnamese_top_facility_comparison(self) -> None:
        result = self.service.detect_intent("So sánh 2 facilities lớn nhất mà hệ thống đang theo dõi")
        self.assertEqual(result.topic, ChatTopic.ENERGY_PERFORMANCE)

    def test_detects_energy_performance_for_english_top_facility_comparison(self) -> None:
        result = self.service.detect_intent("Compare top 2 facilities by production output")
        self.assertEqual(result.topic, ChatTopic.ENERGY_PERFORMANCE)

    def test_detects_weather_extreme_question(self) -> None:
        result = self.service.detect_intent("Nhiet do weather thap nhat cac tram vao ngay 2/1/2026")
        self.assertEqual(result.topic, ChatTopic.ENERGY_PERFORMANCE)

    def test_detects_facility_info_for_timezone_question(self) -> None:
        result = self.service.detect_intent("Mui gio cua cac tram hien tai")
        self.assertEqual(result.topic, ChatTopic.FACILITY_INFO)

    def test_detects_pipeline_status_for_english_pipeline_alerts(self) -> None:
        result = self.service.detect_intent("Any pipeline alerts in the latest ETL run")
        self.assertEqual(result.topic, ChatTopic.PIPELINE_STATUS)

    def test_returns_general_for_unsupported_intent(self) -> None:
        result = self.service.detect_intent("Ke cho toi mot cau chuyen vui")
        self.assertEqual(result.topic, ChatTopic.GENERAL)
        self.assertLessEqual(result.confidence, 0.5)

    def test_returns_general_for_greeting(self) -> None:
        result = self.service.detect_intent("hello")
        self.assertEqual(result.topic, ChatTopic.GENERAL)

    def test_normalize_vietnamese_text_strips_diacritics(self) -> None:
        result = normalize_vietnamese_text("Năm nay chất lượng tốt")
        self.assertEqual(result, "nam nay chat luong tot")

    def test_normalize_vietnamese_text_is_ascii_only(self) -> None:
        result = normalize_vietnamese_text("Nhiệt độ tối đa")
        self.assertTrue(result.isascii())


if __name__ == "__main__":
    unittest.main()
