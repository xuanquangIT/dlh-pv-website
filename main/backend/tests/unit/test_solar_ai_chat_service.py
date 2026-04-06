import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.solar_ai_chat.chat_service import SolarAIChatService


class ExtractExtremeMetricQueryTests(unittest.TestCase):
    def test_detects_lowest_aqi(self) -> None:
        result = SolarAIChatService._extract_extreme_metric_query(
            "Cho toi AQI thap nhat ngay 5/1/2026"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.query_type, "lowest")
        self.assertEqual(result.metric_name, "aqi")

    def test_detects_highest_energy(self) -> None:
        result = SolarAIChatService._extract_extreme_metric_query(
            "San luong energy cao nhat theo tuan ngay 2/1/2026"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.query_type, "highest")
        self.assertEqual(result.metric_name, "energy")

    def test_lowest_wins_when_message_contains_both_markers(self) -> None:
        result = SolarAIChatService._extract_extreme_metric_query(
            "Khong phai cao nhat ma la thap nhat AQI ngay 2/1/2026"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.query_type, "lowest")

    def test_returns_none_when_no_extreme_marker(self) -> None:
        result = SolarAIChatService._extract_extreme_metric_query(
            "Cho toi tong quan he thong"
        )
        self.assertIsNone(result)


class ExtractTimeframeTests(unittest.TestCase):
    def test_year_detected_from_theo_nam_phrase(self) -> None:
        result = SolarAIChatService._extract_timeframe("aqi thap nhat theo nam 2026")
        self.assertEqual(result, "year")

    def test_year_detected_from_nam_year_number(self) -> None:
        result = SolarAIChatService._extract_timeframe("aqi thap nhat nam 2026")
        self.assertEqual(result, "year")

    def test_viet_nam_does_not_trigger_year(self) -> None:
        result = SolarAIChatService._extract_timeframe(
            "aqi thap nhat cac tram viet nam ngay 5/1/2026"
        )
        self.assertEqual(result, "day")

    def test_week_timeframe_detected(self) -> None:
        result = SolarAIChatService._extract_timeframe("energy cao nhat theo tuan")
        self.assertEqual(result, "week")

    def test_month_timeframe_detected(self) -> None:
        result = SolarAIChatService._extract_timeframe("nhiet do thap nhat theo thang 1/2026")
        self.assertEqual(result, "month")

    def test_24h_timeframe_detected(self) -> None:
        result = SolarAIChatService._extract_timeframe("energy thap nhat theo 24 gio")
        self.assertEqual(result, "24h")

    def test_hour_timeframe_detected_from_1_gio(self) -> None:
        result = SolarAIChatService._extract_timeframe("aqi cao nhat trong 1 gio")
        self.assertEqual(result, "hour")

    def test_specific_hour_overrides_all_timeframes(self) -> None:
        result = SolarAIChatService._extract_timeframe(
            "buc xa mat troi cao nhat nam 2026", specific_hour=17
        )
        self.assertEqual(result, "hour")

    def test_defaults_to_day_when_no_timeframe_marker(self) -> None:
        result = SolarAIChatService._extract_timeframe("aqi thap nhat ngay 5/1/2026")
        self.assertEqual(result, "day")


if __name__ == "__main__":
    unittest.main()
