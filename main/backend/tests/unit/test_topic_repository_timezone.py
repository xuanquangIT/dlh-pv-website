import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.repositories.solar_ai_chat.topic_repository import TopicRepository


class TopicRepositoryTimezoneTests(unittest.TestCase):
    def test_derives_eastern_timezone_for_east_australia_coordinates(self) -> None:
        timezone_name, timezone_offset = TopicRepository._derive_timezone_from_coordinates(-34.0, 150.0)
        self.assertEqual(timezone_name, "Australia/Eastern")
        self.assertEqual(timezone_offset, "UTC+10:00")

    def test_derives_central_timezone_for_central_australia_coordinates(self) -> None:
        timezone_name, timezone_offset = TopicRepository._derive_timezone_from_coordinates(-31.0, 133.0)
        self.assertEqual(timezone_name, "Australia/Central")
        self.assertEqual(timezone_offset, "UTC+09:30")

    def test_derives_approx_timezone_for_out_of_scope_coordinates(self) -> None:
        timezone_name, timezone_offset = TopicRepository._derive_timezone_from_coordinates(40.7, -74.0)
        self.assertEqual(timezone_name, "UTC (approx)")
        self.assertEqual(timezone_offset, "UTC-05:00")


if __name__ == "__main__":
    unittest.main()
