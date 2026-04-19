import sys
import unittest
from pathlib import Path
from datetime import date

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

class TestBugFindings20260419(unittest.TestCase):
    
    def test_bug01_system_overview_reads_mart(self):
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._system_overview_databricks)
        self.assertIn(
            "gold.mart_system_kpi_daily",
            src,
            "Must read from gold.mart_system_kpi_daily"
        )
        self.assertNotIn(
            "silver.energy_readings",
            src,
            "Must NOT read from silver.energy_readings"
        )
        self.assertIn("active_facility_count", src)
        self.assertIn("total_facility_count", src)
        
    def test_bug02_energy_perf_reads_mart(self):
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._energy_performance_databricks)
        self.assertIn(
            "gold.mart_energy_daily",
            src,
            "Must read from gold.mart_energy_daily"
        )
        self.assertIn("facility_id", src)
        self.assertIn("facility_name", src)

    def test_bug03_facility_lookup_finleysf(self):
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._facility_info_databricks)
        self.assertIn("UPPER(facility_code) = UPPER(", src, "Must match on code first")
        
    def test_bug05_time_anchoring(self):
        from app.repositories.solar_ai_chat.base_repository import BaseRepository
        import inspect
        src = inspect.getsource(BaseRepository._resolve_period_window)
        # Should not anchor on CURRENT_DATE without checking if anchor is provided
        pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
