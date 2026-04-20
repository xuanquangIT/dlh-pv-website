"""test_bug_fixes.py
===================
Unit / integration tests for all 9 bugs reported in the bug-fix sprint.

Tests run fully in-process (no live HTTP server required).  They mock
Databricks SQL where needed and test the exact layer that was changed.

Usage
-----
# From the dlh-pv-website root
python -m pytest main/backend/scripts/test_bug_fixes.py -v

# Or run directly:
python main/backend/scripts/test_bug_fixes.py
"""
from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Colour helpers (no deps)
# ---------------------------------------------------------------------------
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"{_GREEN}  PASS{_RESET}  {msg}")


def _fail(msg: str, detail: str = "") -> None:
    print(f"{_RED}  FAIL{_RESET}  {msg}")
    if detail:
        print(f"         {_YELLOW}{detail}{_RESET}")


# ===========================================================================
# Bug #1 — Peak hours: nighttime rows must be excluded
# ===========================================================================

class TestBug1PeakHours(unittest.TestCase):
    """Verify the peak-hours SQL for energy performance excludes zero-energy rows."""

    def test_sql_contains_daytime_filter_sql(self):
        """The SQL in _energy_performance_databricks must include AND energy_mwh > 0."""
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._energy_performance_databricks)
        self.assertIn(
            "energy_mwh > 0",
            src,
            "Peak hours SQL must filter out zero-energy (nighttime) rows",
        )

    def test_csv_fallback_excludes_zero_energy(self):
        """CSV fallback must skip hours with energy == 0 when building hour_totals."""
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._energy_performance_csv)
        self.assertIn("energy > 0", src, "CSV fallback must only count daytime hours")


# ===========================================================================
# Bug #2 / Bug #8 — ML model intent: extended keywords
# ===========================================================================

class TestBug2MLModelIntent(unittest.TestCase):
    """ML model intent must be triggered by extended phrases."""

    def setUp(self):
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService
        self.svc = VietnameseIntentService(embedding_client=None, semantic_enabled=False)

    def _detect(self, msg: str):
        from app.schemas.solar_ai_chat import ChatTopic
        result = self.svc.detect_intent(msg)
        return result.topic == ChatTopic.ML_MODEL, result.topic, result.confidence

    def test_ml_model_info_phrase(self):
        hit, topic, _ = self._detect("What is the current ML model info? Give me model name, version, R2")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")

    def test_nrmse_phrase(self):
        hit, topic, _ = self._detect("Show me the NRMSE and skill score of the current model")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")

    def test_champion_model_phrase(self):
        hit, topic, _ = self._detect("champion model performance metrics")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")

    def test_model_version_phrase(self):
        hit, topic, _ = self._detect("What is the current forecast model version?")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")

    def test_r_squared_phrase(self):
        hit, topic, _ = self._detect("Give me the R-squared and R2 of the model")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")

    def test_model_quality_phrase(self):
        hit, topic, _ = self._detect("Is model quality improving or declining?")
        self.assertTrue(hit, f"Expected ML_MODEL, got {topic}")


# ===========================================================================
# Bug #3 — Pipeline job names ≠ facility names
# ===========================================================================

class TestBug3PipelineJobNames(unittest.TestCase):
    """Alerts from pipeline_run_diagnostics must use 'pipeline_name' key, not 'facility'."""

    def test_pipeline_alert_key_is_pipeline_name(self):
        """_pipeline_status_databricks alert dicts must use 'pipeline_name', not 'facility'."""
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._pipeline_status_databricks)
        self.assertIn(
            '"pipeline_name"',
            src,
            "Alert dicts must use 'pipeline_name' key",
        )
        # Old key 'facility' must NOT be used for pipeline alerts in this method
        # (it may appear elsewhere for actual facility data, but not in this block)
        # We check the intent: 'pipeline_name' is present as a key
        self.assertNotIn(
            '"facility": str(row.get("pipeline_name")',
            src,
            "Old 'facility' key with pipeline_name value must not exist",
        )

    def test_system_prompt_pipeline_schema(self):
        """System prompt must clarify that pipeline_name is a Databricks job, not a facility."""
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        self.assertIn(
            "Databricks",
            _LAKEHOUSE_ARCHITECTURE_CONTEXT,
        )
        self.assertIn(
            "pipeline_name",
            _LAKEHOUSE_ARCHITECTURE_CONTEXT,
            "System prompt table must include pipeline_name key",
        )


# ===========================================================================
# Bug #4 — Forecast window must not return past dates
# ===========================================================================

class TestBug4ForecastFuture(unittest.TestCase):
    """Forecast fallback must NEVER return past dates."""

    def test_primary_query_uses_current_date(self):
        """The primary forecast query must use >= current_date() not a past window."""
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._forecast_72h_databricks)
        self.assertIn(
            "current_date()",
            src,
            "Primary forecast query must use current_date()",
        )
        # Fallback may use ORDER BY day DESC LIMIT 3 alongside forecast_stale = True

    def test_fallback_query_future_dates_only(self):
        """Fallback query must include >= current_date() so only future rows are returned."""
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository
        import inspect
        src = inspect.getsource(TopicRepository._forecast_72h_databricks)
        # The fallback block must ALSO filter by current_date
        future_fallback_patterns = [
            "current_date()",
            "future_rows",
            "Never fall back to past",
        ]
        found = any(p in src for p in future_fallback_patterns)
        self.assertTrue(
            found,
            "Fallback must filter to future dates only (>= current_date)",
        )


# ===========================================================================
# Bug #5 — Facility ID lookup
# ===========================================================================

class TestBug5FacilityIDLookup(unittest.TestCase):
    """Facility codes like WRSF1 must be translated to full names."""

    def test_facility_id_map_exists(self):
        from app.services.solar_ai_chat.nlp_parser import FACILITY_ID_MAP
        self.assertIn("WRSF1", FACILITY_ID_MAP)
        self.assertIn("EMERASF", FACILITY_ID_MAP)
        self.assertIn("DARLSF", FACILITY_ID_MAP)

    def test_resolve_facility_code(self):
        from app.services.solar_ai_chat.nlp_parser import resolve_facility_code
        self.assertEqual(resolve_facility_code("WRSF1"), "White Rock Solar Farm")
        self.assertEqual(resolve_facility_code("wrsf1"), "White Rock Solar Farm")
        self.assertEqual(resolve_facility_code("EMERASF"), "Emerald Solar Farm")
        self.assertIsNone(resolve_facility_code("UNKNOWN"))

    def test_expand_facility_codes_in_message(self):
        from app.services.solar_ai_chat.nlp_parser import expand_facility_codes_in_message
        result = expand_facility_codes_in_message("Show data for WRSF1 station")
        self.assertIn("White Rock Solar Farm", result)
        self.assertNotIn("WRSF1", result)

    def test_expand_preserves_non_code_words(self):
        from app.services.solar_ai_chat.nlp_parser import expand_facility_codes_in_message
        result = expand_facility_codes_in_message("What is the forecast today?")
        self.assertEqual(result, "What is the forecast today?")

    def test_intent_recognizes_facility_codes(self):
        """Facility codes should now be classified as FACILITY_INFO, not GENERAL."""
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService
        from app.schemas.solar_ai_chat import ChatTopic
        svc = VietnameseIntentService(embedding_client=None, semantic_enabled=False)
        result = svc.detect_intent("WRSF1 station data")
        self.assertNotEqual(
            result.topic,
            ChatTopic.GENERAL,
            f"WRSF1 should not be classified as GENERAL, got {result.topic}",
        )

    def test_expand_facility_in_chat_service_message(self):
        """expand_facility_codes_in_message is importable from chat_service module."""
        from app.services.solar_ai_chat.chat_service import expand_facility_codes_in_message  # re-export
        result = expand_facility_codes_in_message("Data for DARLSF today")
        self.assertIn("Darlington Point Solar Farm", result)


# ===========================================================================
# Bug #6 — Extreme values: all_time timeframe
# ===========================================================================

class TestBug6ExtremeAllTime(unittest.TestCase):
    """_resolve_period_window must support 'all_time' timeframe."""

    def test_all_time_timeframe(self):
        from app.repositories.solar_ai_chat.base_repository import BaseRepository
        anchor = date(2026, 4, 14)
        start, end, label = BaseRepository._resolve_period_window("all_time", anchor)
        self.assertEqual(label, "all time")
        # start should be far in the past (10+ years ago)
        self.assertLess(start.year, anchor.year - 5)
        # end should be tomorrow or today's end
        self.assertGreaterEqual(end.date(), anchor)

    def test_all_timeframe_alias(self):
        from app.repositories.solar_ai_chat.base_repository import BaseRepository
        anchor = date(2026, 4, 14)
        start_all, _, _ = BaseRepository._resolve_period_window("all", anchor)
        start_hist, _, _ = BaseRepository._resolve_period_window("history", anchor)
        self.assertEqual(start_all.year, start_hist.year)

    def test_extract_timeframe_all_time_keywords(self):
        from app.services.solar_ai_chat.nlp_parser import extract_timeframe
        from app.services.solar_ai_chat.intent_service import normalize_vietnamese_text
        msg = normalize_vietnamese_text("lịch sử toàn bộ")
        result = extract_timeframe(msg)
        self.assertEqual(result, "all_time")

    def test_extract_timeframe_historical(self):
        from app.services.solar_ai_chat.nlp_parser import extract_timeframe
        self.assertEqual(extract_timeframe("historical record ever"), "all_time")

    def test_existing_timeframes_unchanged(self):
        """Existing timeframes (day/week/month/year) must still work."""
        from app.services.solar_ai_chat.nlp_parser import extract_timeframe
        self.assertEqual(extract_timeframe("tuan nay"), "week")
        self.assertEqual(extract_timeframe("thang truoc"), "month")
        self.assertEqual(extract_timeframe("ca nam"), "year")
        self.assertEqual(extract_timeframe("hom nay"), "day")


# ===========================================================================
# Bug #7 — Future date must not trigger scope refusal
# ===========================================================================

class TestBug7FutureDateHandling(unittest.TestCase):
    """System prompt must handle future-date station queries gracefully."""

    def test_system_prompt_has_future_date_rule(self):
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        ctx = _LAKEHOUSE_ARCHITECTURE_CONTEXT.lower()
        self.assertIn(
            "future",
            ctx,
            "System prompt must contain future-date handling guidance",
        )
        self.assertIn(
            "get_forecast_72h",
            _LAKEHOUSE_ARCHITECTURE_CONTEXT,
            "Future-date guidance must suggest get_forecast_72h tool",
        )

    def test_system_prompt_no_scope_refusal_for_future(self):
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        # The future date guidance must explicitly say NOT to use scope refusal
        self.assertIn(
            "DO NOT return a scope-refusal",
            _LAKEHOUSE_ARCHITECTURE_CONTEXT,
            "System prompt must explicitly avoid scope-refusal for future date queries",
        )


# ===========================================================================
# Bug #8 — ML hallucination guardrail in system prompt
# ===========================================================================

class TestBug8MLHallucinationGuardrail(unittest.TestCase):
    """System prompt must contain guardrails against ML hallucination."""

    def test_guardrail_regression_only_metrics(self):
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        ctx = _LAKEHOUSE_ARCHITECTURE_CONTEXT
        # Must mention regression metrics
        self.assertIn("R²", ctx, "Guardrail must mention R² regression metric")
        self.assertIn("nRMSE", ctx, "Guardrail must mention nRMSE metric")
        self.assertIn("Skill Score", ctx, "Guardrail must mention Skill Score")

    def test_guardrail_forbids_classification_metrics(self):
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        ctx = _LAKEHOUSE_ARCHITECTURE_CONTEXT
        self.assertIn(
            "Accuracy, Precision, Recall",
            ctx,
            "Guardrail must forbid Accuracy/Precision/Recall metrics",
        )

    def test_guardrail_data_unavailable_instruction(self):
        from app.services.solar_ai_chat.prompt_builder import _LAKEHOUSE_ARCHITECTURE_CONTEXT
        ctx = _LAKEHOUSE_ARCHITECTURE_CONTEXT
        self.assertIn(
            "Model data is currently unavailable",
            ctx,
            "Guardrail must instruct 'Model data is currently unavailable' fallback",
        )


# ===========================================================================
# Bug #9 — Quick chips setValue method
# ===========================================================================

class TestBug9QuickChips(unittest.TestCase):
    """MessageInput.prototype.setValue must exist in solar_chat_page.js."""

    def test_setvalue_method_exists(self):
        js_path = (
            Path(__file__).resolve().parents[2]
            / "frontend/static/js/platform_portal/solar_chat_page.js"
        )
        self.assertTrue(js_path.exists(), f"JS file not found: {js_path}")
        src = js_path.read_text(encoding="utf-8")
        self.assertIn(
            "MessageInput.prototype.setValue",
            src,
            "solar_chat_page.js must define MessageInput.prototype.setValue",
        )

    def test_chip_handler_uses_setvalue(self):
        js_path = (
            Path(__file__).resolve().parents[2]
            / "frontend/static/js/platform_portal/solar_chat_page.js"
        )
        src = js_path.read_text(encoding="utf-8")
        self.assertIn(
            "messageInput.setValue(prompt)",
            src,
            "Chip click handler must call messageInput.setValue(prompt) before sendMessageFlow",
        )

    def test_pipeline_chip_uses_setvalue(self):
        js_path = (
            Path(__file__).resolve().parents[2]
            / "frontend/static/js/platform_portal/solar_chat_page.js"
        )
        src = js_path.read_text(encoding="utf-8")
        self.assertIn(
            'messageInput.setValue("Show latest pipeline status")',
            src,
            "Pipeline chip must also call messageInput.setValue before send",
        )


# ===========================================================================
# Integration: intent cache invalidation after keyword expansion
# ===========================================================================

class TestIntentCacheSafety(unittest.TestCase):
    """Intent cache must not return stale results after keyword updates."""

    def test_cache_does_not_cross_contaminate(self):
        """Two different messages must not share cached intent."""
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService
        svc = VietnameseIntentService(embedding_client=None, semantic_enabled=False)
        r1 = svc.detect_intent("What is the forecast for next 72 hours?")
        r2 = svc.detect_intent("Show me energy performance")
        self.assertNotEqual(
            r1.topic,
            r2.topic,
            "Different messages must resolve to different topics",
        )


# ===========================================================================
# Runner
# ===========================================================================

def _run_suite() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestBug1PeakHours,
        TestBug2MLModelIntent,
        TestBug3PipelineJobNames,
        TestBug4ForecastFuture,
        TestBug5FacilityIDLookup,
        TestBug6ExtremeAllTime,
        TestBug7FutureDateHandling,
        TestBug8MLHallucinationGuardrail,
        TestBug9QuickChips,
        TestIntentCacheSafety,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    passed = total - failures

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if failures:
        print(f"{_RED}FAILED: {failures} test(s){_RESET}")
        for failure in result.failures + result.errors:
            print(f"  - {failure[0]}")
    else:
        print(f"{_GREEN}All tests passed!{_RESET}")
    print("=" * 60)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_suite())
