"""Comprehensive unit tests for TopicRepository and its module-level helpers.

Strategy:
- All Databricks I/O is patched via _execute_query / _safe_execute_query so
  tests run entirely in-process without a real connection.
- CSV-path helpers are patched where needed.
- Each public / semi-public method is covered by at least one positive case;
  edge cases (empty rows, None values, unknown aliases, etc.) are also tested.
"""
from __future__ import annotations

import sys
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup + stub heavy dependencies before any project import
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Stub databricks.sql so imports succeed without the real SDK.
try:
    import databricks  # noqa: F401
except ImportError:
    _db_pkg = types.ModuleType("databricks")
    _db_pkg.__path__ = []
    _db_sql = types.ModuleType("databricks.sql")
    _db_sql.connect = MagicMock()
    sys.modules.setdefault("databricks", _db_pkg)
    sys.modules.setdefault("databricks.sql", _db_sql)

# bcrypt stub (mirrors conftest logic)
if "bcrypt" not in sys.modules:
    bcrypt_stub = types.ModuleType("bcrypt")
    bcrypt_stub.gensalt = lambda rounds=12: b"stub-salt"
    bcrypt_stub.hashpw = lambda password, salt: password + b"." + salt
    bcrypt_stub.checkpw = lambda password, hashed: True
    sys.modules["bcrypt"] = bcrypt_stub

from app.repositories.solar_ai_chat.topic_repository import (  # noqa: E402
    TopicRepository,
    _apply_energy_focus,
    _normalize_focus,
)
from app.core.settings import SolarChatSettings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**kwargs) -> SolarChatSettings:
    defaults = dict(
        llm_api_format="gemini",
        llm_api_key="test-key",
        databricks_host=None,
        databricks_token=None,
        databricks_sql_http_path=None,
        uc_catalog="pv",
        uc_silver_schema="silver",
        uc_gold_schema="gold",
        trino_catalog="postgresql",
        analytics_lookback_days=30,
    )
    defaults.update(kwargs)
    return SolarChatSettings.model_construct(**defaults)


def _make_repo(**kwargs) -> TopicRepository:
    settings = _make_settings(**kwargs)
    repo = TopicRepository.__new__(TopicRepository)
    repo._settings = settings
    repo._trino_catalog = "iceberg"
    repo._catalog = "pv"
    repo._silver_schema = "silver"
    repo._gold_schema = "gold"
    return repo


# ---------------------------------------------------------------------------
# 1. Module-level helpers
# ---------------------------------------------------------------------------

class TestNormalizeFocus(unittest.TestCase):
    def test_none_returns_overview(self):
        self.assertEqual(_normalize_focus(None), "overview")

    def test_canonical_overview(self):
        self.assertEqual(_normalize_focus("overview"), "overview")

    def test_canonical_energy(self):
        self.assertEqual(_normalize_focus("energy"), "energy")

    def test_canonical_capacity(self):
        self.assertEqual(_normalize_focus("capacity"), "capacity")

    def test_alias_capacity_factor(self):
        self.assertEqual(_normalize_focus("capacity_factor"), "capacity")

    def test_alias_capacity_factor_pct(self):
        self.assertEqual(_normalize_focus("capacity_factor_pct"), "capacity")

    def test_alias_mwh(self):
        self.assertEqual(_normalize_focus("mwh"), "energy")

    def test_alias_production(self):
        self.assertEqual(_normalize_focus("production"), "energy")

    def test_alias_summary(self):
        self.assertEqual(_normalize_focus("summary"), "overview")

    def test_alias_general(self):
        self.assertEqual(_normalize_focus("general"), "overview")

    def test_alias_all(self):
        self.assertEqual(_normalize_focus("all"), "overview")

    def test_alias_cf(self):
        self.assertEqual(_normalize_focus("cf"), "capacity")

    def test_alias_efficiency(self):
        self.assertEqual(_normalize_focus("efficiency"), "capacity")

    def test_unknown_value_defaults_to_overview(self):
        self.assertEqual(_normalize_focus("completely_unknown_xyz"), "overview")

    def test_strips_whitespace_and_lowercases(self):
        self.assertEqual(_normalize_focus("  ENERGY  "), "energy")

    def test_hyphen_replaced_by_underscore(self):
        self.assertEqual(_normalize_focus("capacity-factor"), "capacity")

    def test_empty_string_defaults_to_overview(self):
        self.assertEqual(_normalize_focus(""), "overview")


class TestApplyEnergyFocus(unittest.TestCase):
    """Tests for the _apply_energy_focus shaping function."""

    BASE_METRICS = {
        "all_facilities": [
            {"facility_id": "A", "facility": "Alpha", "energy_mwh": 100.0, "capacity_factor_pct": 0.25},
            {"facility_id": "B", "facility": "Beta",  "energy_mwh": 80.0,  "capacity_factor_pct": 0.20},
        ],
        "top_facilities": [
            {"facility_id": "A", "facility": "Alpha", "energy_mwh": 100.0, "capacity_factor_pct": 0.25},
        ],
        "bottom_facilities": [
            {"facility_id": "B", "facility": "Beta", "energy_mwh": 80.0, "capacity_factor_pct": 0.20},
        ],
        "top_performance_ratio_facilities": [
            {"facility": "Alpha", "performance_ratio_pct": 90.0},
        ],
        "tomorrow_forecast_mwh": 50.0,
        "facility_count": 2,
        "peak_hours": [{"hour": 12, "energy_mwh": 30.0}],
        "window_days": 30,
    }

    def _copy(self):
        import copy
        return copy.deepcopy(self.BASE_METRICS)

    def test_energy_focus_removes_capacity_factor_from_rows(self):
        out = _apply_energy_focus(self._copy(), "energy")
        for row in out["all_facilities"]:
            self.assertNotIn("capacity_factor_pct", row)
            self.assertIn("energy_mwh", row)

    def test_overview_focus_removes_performance_ratio_list(self):
        out = _apply_energy_focus(self._copy(), "overview")
        self.assertNotIn("top_performance_ratio_facilities", out)

    def test_capacity_focus_removes_energy_mwh_from_rows(self):
        out = _apply_energy_focus(self._copy(), "capacity")
        for row in out["all_facilities"]:
            self.assertNotIn("energy_mwh", row)
            self.assertIn("capacity_factor_pct", row)

    def test_capacity_focus_removes_forecast_and_peak_keys(self):
        out = _apply_energy_focus(self._copy(), "capacity")
        self.assertNotIn("tomorrow_forecast_mwh", out)
        self.assertNotIn("peak_hours", out)

    def test_zero_tomorrow_forecast_suppressed_in_energy_path(self):
        m = self._copy()
        m["tomorrow_forecast_mwh"] = 0.0
        out = _apply_energy_focus(m, "energy")
        self.assertNotIn("tomorrow_forecast_mwh", out)

    def test_non_zero_tomorrow_forecast_kept_in_energy_path(self):
        out = _apply_energy_focus(self._copy(), "energy")
        self.assertIn("tomorrow_forecast_mwh", out)
        self.assertEqual(out["tomorrow_forecast_mwh"], 50.0)

    def test_limit_applied_to_all_facilities(self):
        m = self._copy()
        out = _apply_energy_focus(m, "energy", limit=1)
        self.assertEqual(len(out["all_facilities"]), 1)

    def test_limit_drops_top_bottom_facilities_and_updates_count(self):
        out = _apply_energy_focus(self._copy(), "energy", limit=1)
        self.assertNotIn("top_facilities", out)
        self.assertNotIn("bottom_facilities", out)
        self.assertEqual(out["facility_count"], 1)

    def test_capacity_limit_also_updates_count_and_drops_top_bottom(self):
        out = _apply_energy_focus(self._copy(), "capacity", limit=1)
        self.assertNotIn("top_facilities", out)
        self.assertNotIn("bottom_facilities", out)
        self.assertEqual(out["facility_count"], 1)

    def test_missing_list_keys_do_not_crash(self):
        """When rows are missing entirely, the function should not raise."""
        m = {"facility_count": 0, "window_days": 30}
        out = _apply_energy_focus(m, "overview")
        self.assertIsInstance(out, dict)

    def test_none_focus_treated_as_overview(self):
        out = _apply_energy_focus(self._copy(), None)
        self.assertNotIn("top_performance_ratio_facilities", out)


# ---------------------------------------------------------------------------
# 2. TopicRepository._lookback_days
# ---------------------------------------------------------------------------

class TestLookbackDays(unittest.TestCase):
    def test_returns_configured_value(self):
        repo = _make_repo(analytics_lookback_days=14)
        self.assertEqual(repo._lookback_days(), 14)

    def test_minimum_is_one(self):
        repo = _make_repo(analytics_lookback_days=0)
        self.assertEqual(repo._lookback_days(), 1)

    def test_invalid_setting_falls_back_to_30(self):
        repo = _make_repo()
        repo._settings = SimpleNamespace(analytics_lookback_days="bad")
        self.assertEqual(repo._lookback_days(), 30)

    def test_missing_attribute_falls_back_to_30(self):
        repo = _make_repo()
        repo._settings = SimpleNamespace()
        self.assertEqual(repo._lookback_days(), 30)


# ---------------------------------------------------------------------------
# 3. TopicRepository._safe_execute_query
# ---------------------------------------------------------------------------

class TestSafeExecuteQuery(unittest.TestCase):
    def test_returns_rows_on_success(self):
        repo = _make_repo()
        expected = [{"col": "val"}]
        with patch.object(repo, "_execute_query", return_value=expected):
            result = repo._safe_execute_query("SELECT 1")
        self.assertEqual(result, expected)

    def test_returns_empty_list_on_exception(self):
        repo = _make_repo()
        with patch.object(repo, "_execute_query", side_effect=RuntimeError("boom")):
            result = repo._safe_execute_query("SELECT bad")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 4. _general_greeting
# ---------------------------------------------------------------------------

class TestGeneralGreeting(unittest.TestCase):
    def test_returns_available_topics_excluding_general(self):
        repo = _make_repo()
        metrics, sources = repo._general_greeting()
        self.assertIn("available_topics", metrics)
        self.assertNotIn("general", metrics["available_topics"])
        self.assertEqual(sources[0]["layer"], "Gold")


# ---------------------------------------------------------------------------
# 5. _system_overview_databricks
# ---------------------------------------------------------------------------

_SYSTEM_KPI_ROW = {
    "total_energy_mwh": 1234.5,
    "champion_r2": 0.912,
    "data_completeness_pct": 98.7,
    "active_facility_count": 7,
    "total_facility_count": 8,
    "avg_capacity_factor_pct": 0.31,
    "created_at": "2026-04-20T10:00",
}


class TestSystemOverviewDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _call(self, rows):
        with patch.object(self.repo, "_execute_query", return_value=rows), \
             patch.object(self.repo, "_resolve_latest_datetime", return_value="2026-04-20T10:00"):
            return self.repo._system_overview_databricks(30)

    def test_happy_path_parses_row(self):
        result = self._call([_SYSTEM_KPI_ROW])
        self.assertAlmostEqual(result["production_output_mwh"], 1234.5)
        self.assertAlmostEqual(result["r_squared"], 0.912)
        self.assertEqual(result["facility_count"], 8)
        self.assertEqual(result["active_facility_count"], 7)
        self.assertEqual(result["window_days"], 30)

    def test_empty_rows_returns_zeros(self):
        result = self._call([])
        self.assertEqual(result["production_output_mwh"], 0.0)
        self.assertIsNone(result["r_squared"])
        self.assertEqual(result["facility_count"], 0)

    def test_none_champion_r2_handled(self):
        row = {**_SYSTEM_KPI_ROW, "champion_r2": None}
        result = self._call([row])
        self.assertIsNone(result["r_squared"])

    def test_none_numeric_fields_become_zero(self):
        row = {
            "total_energy_mwh": None,
            "champion_r2": None,
            "data_completeness_pct": None,
            "active_facility_count": None,
            "total_facility_count": None,
            "avg_capacity_factor_pct": None,
            "created_at": None,
        }
        result = self._call([row])
        self.assertEqual(result["production_output_mwh"], 0.0)
        self.assertEqual(result["data_quality_score"], 0.0)


# ---------------------------------------------------------------------------
# 6. _system_overview (arguments / timeframe override)
# ---------------------------------------------------------------------------

class TestSystemOverview(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def test_timeframe_days_argument_overrides_lookback(self):
        captured = {}

        def _fake_db(days):
            captured["days"] = days
            return {
                "production_output_mwh": 0.0, "r_squared": None,
                "data_quality_score": 0.0, "facility_count": 0,
                "active_facility_count": 0, "avg_capacity_factor_pct": 0.0,
                "window_days": days, "latest_data_timestamp": None,
            }

        with patch.object(self.repo, "_system_overview_databricks", side_effect=_fake_db), \
             patch.object(self.repo, "_with_databricks_query",
                          side_effect=lambda label, qfn, fb, src: (qfn(), src)):
            self.repo._system_overview({"timeframe_days": 90})

        self.assertEqual(captured["days"], 90)

    def test_timeframe_days_never_shrinks_below_default(self):
        captured = {}

        def _fake_db(days):
            captured["days"] = days
            return {"window_days": days}

        with patch.object(self.repo, "_system_overview_databricks", side_effect=_fake_db), \
             patch.object(self.repo, "_with_databricks_query",
                          side_effect=lambda label, qfn, fb, src: (qfn(), src)):
            self.repo._system_overview({"timeframe_days": 1})

        self.assertGreaterEqual(captured["days"], self.repo._lookback_days())


# ---------------------------------------------------------------------------
# 7. _energy_performance_databricks
# ---------------------------------------------------------------------------

class TestEnergyPerformanceDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()
        self._fac_rows = [
            {"facility_id": "A", "facility_name": "Alpha", "total_mwh": 200.0, "capacity_factor_pct": 0.30},
            {"facility_id": "B", "facility_name": "Beta",  "total_mwh": 150.0, "capacity_factor_pct": 0.25},
            {"facility_id": "C", "facility_name": "Gamma", "total_mwh": 100.0, "capacity_factor_pct": None},
        ]
        self._peak_rows = [{"hr": 12, "total_mwh": 80.0}, {"hr": 13, "total_mwh": 60.0}]
        self._forecast_rows = [{"tomorrow_mwh": 50.0}]

    def _call(self, fac=None, peak=None, top_ratio=None, forecast=None):
        fac = fac if fac is not None else self._fac_rows
        peak = peak if peak is not None else self._peak_rows
        top_ratio = top_ratio if top_ratio is not None else []
        forecast = forecast if forecast is not None else self._forecast_rows

        side_effects = iter([fac, peak])
        safe_side_effects = iter([top_ratio, forecast])

        with patch.object(self.repo, "_resolve_latest_date", return_value=date(2026, 4, 20)), \
             patch.object(self.repo, "_execute_query", side_effect=side_effects), \
             patch.object(self.repo, "_safe_execute_query", side_effect=safe_side_effects):
            return self.repo._energy_performance_databricks(30)

    def test_facility_count_matches_rows(self):
        result = self._call()
        self.assertEqual(result["facility_count"], 3)

    def test_top_facilities_is_first_three_by_desc_mwh(self):
        result = self._call()
        self.assertEqual(len(result["top_facilities"]), 3)
        self.assertEqual(result["top_facilities"][0]["facility"], "Alpha")

    def test_bottom_facilities_excludes_top_names(self):
        result = self._call()
        top_names = {f["facility"] for f in result["top_facilities"]}
        for b in result["bottom_facilities"]:
            self.assertNotIn(b["facility"], top_names)

    def test_peak_hours_populated(self):
        result = self._call()
        self.assertEqual(result["peak_hours"][0]["hour"], 12)

    def test_tomorrow_forecast_from_first_query(self):
        result = self._call()
        self.assertAlmostEqual(result["tomorrow_forecast_mwh"], 50.0)

    def test_no_capacity_factor_handled_gracefully(self):
        """Row with None capacity_factor_pct should not include the key."""
        result = self._call()
        gamma = next(f for f in result["all_facilities"] if f["facility"] == "Gamma")
        self.assertNotIn("capacity_factor_pct", gamma)

    def test_fallback_forecast_when_primary_empty(self):
        """When forecast query returns empty, falls back to historical daily average."""
        daily_rows = [
            {"daily_mwh": 40.0},
            {"daily_mwh": 60.0},
        ]
        top_ratio: list = []
        safe_effects = iter([top_ratio, [], daily_rows])
        with patch.object(self.repo, "_resolve_latest_date", return_value=date(2026, 4, 20)), \
             patch.object(self.repo, "_execute_query", side_effect=iter([self._fac_rows, self._peak_rows])), \
             patch.object(self.repo, "_safe_execute_query", side_effect=safe_effects):
            result = self.repo._energy_performance_databricks(30)
        self.assertAlmostEqual(result["tomorrow_forecast_mwh"], 50.0)


# ---------------------------------------------------------------------------
# 8. _energy_performance (focus + limit forwarding)
# ---------------------------------------------------------------------------

class TestEnergyPerformance(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()
        self._raw_metrics = {
            "all_facilities": [
                {"facility_id": "A", "facility": "Alpha", "energy_mwh": 10.0, "capacity_factor_pct": 0.5},
            ],
            "top_facilities": [],
            "bottom_facilities": [],
            "facility_count": 1,
            "peak_hours": [],
            "top_performance_ratio_facilities": [],
            "tomorrow_forecast_mwh": 5.0,
            "window_days": 30,
        }

    def _call(self, args=None):
        with patch.object(self.repo, "_with_databricks_query",
                          return_value=(self._raw_metrics, [])):
            return self.repo._energy_performance(args)

    def test_capacity_focus_applied(self):
        metrics, _ = self._call({"focus": "capacity"})
        for row in metrics["all_facilities"]:
            self.assertNotIn("energy_mwh", row)

    def test_limit_applied(self):
        # Add extra facilities so truncation is visible
        self._raw_metrics["all_facilities"] = [
            {"facility_id": str(i), "facility": f"F{i}", "energy_mwh": float(i), "capacity_factor_pct": 0.1}
            for i in range(5)
        ]
        metrics, _ = self._call({"focus": "energy", "limit": 2})
        self.assertEqual(len(metrics["all_facilities"]), 2)

    def test_invalid_limit_ignored(self):
        metrics, _ = self._call({"limit": "bad"})
        self.assertIsNotNone(metrics)

    def test_invalid_timeframe_days_ignored(self):
        metrics, _ = self._call({"timeframe_days": "NaN"})
        self.assertIsNotNone(metrics)

    def test_no_arguments_uses_defaults(self):
        metrics, _ = self._call(None)
        self.assertIsNotNone(metrics)


# ---------------------------------------------------------------------------
# 9. _ml_model_databricks
# ---------------------------------------------------------------------------

_ML_ROW = {
    "model_name": "GBT-Solar",
    "model_version": "v3",
    "approach": "C:+MultiLag",
    "eval_date": "2026-04-15",
    "r2": 0.923,
    "skill_score": 0.41,
    "nrmse_pct": 12.3,
}


class TestMlModelDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _call(self, primary_rows, previous_rows=None):
        previous_rows = previous_rows or []
        effects = iter([primary_rows, previous_rows])
        with patch.object(self.repo, "_safe_execute_query", side_effect=effects):
            return self.repo._ml_model_databricks()

    def test_happy_path_populates_all_fields(self):
        result = self._call([_ML_ROW])
        self.assertEqual(result["model_name"], "GBT-Solar")
        self.assertEqual(result["model_version"], "v3")
        self.assertAlmostEqual(result["comparison"]["current_r_squared"], 0.923)

    def test_empty_rows_returns_unknown(self):
        # Both primary and fallback return empty
        with patch.object(self.repo, "_safe_execute_query", return_value=[]):
            result = self.repo._ml_model_databricks()
        self.assertEqual(result["model_name"], "Unknown")

    def test_previous_r2_delta_computed(self):
        previous = [{"model_version": "v2", "r2": 0.90}]
        result = self._call([_ML_ROW], previous)
        self.assertIsNotNone(result["comparison"]["previous_r_squared"])
        self.assertAlmostEqual(result["comparison"]["delta_r_squared"],
                               round(0.923 - 0.90, 4))

    def test_is_fallback_model_detected(self):
        fallback_row = {**_ML_ROW, "model_name": "GBT-Solar:fallback", "model_version": "v3"}
        with patch.object(self.repo, "_safe_execute_query",
                          side_effect=iter([[fallback_row], [], []])):
            result = self.repo._ml_model_databricks()
        self.assertTrue(result["is_fallback_model"])

    def test_skill_score_none_preserved(self):
        row = {**_ML_ROW, "skill_score": None}
        result = self._call([row])
        self.assertIsNone(result["comparison"]["skill_score"])

    def test_nrmse_none_preserved(self):
        row = {**_ML_ROW, "nrmse_pct": None}
        result = self._call([row])
        self.assertIsNone(result["comparison"]["nrmse_pct"])


# ---------------------------------------------------------------------------
# 10. _build_ml_metrics (static)
# ---------------------------------------------------------------------------

class TestBuildMlMetrics(unittest.TestCase):
    def test_structure(self):
        result = TopicRepository._build_ml_metrics(0.9, 0.98)
        self.assertEqual(result["model_name"], "GBT")
        self.assertIn("comparison", result)
        self.assertAlmostEqual(result["comparison"]["current_r_squared"], 0.9)

    def test_baseline_r2_is_smaller(self):
        result = TopicRepository._build_ml_metrics(0.9, 0.98)
        self.assertLess(result["comparison"]["previous_r_squared"],
                        result["comparison"]["current_r_squared"])


# ---------------------------------------------------------------------------
# 11. _pipeline_status_databricks
# ---------------------------------------------------------------------------

class TestPipelineStatusDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _make_run_row(self, status="SUCCESS", bronze_fail=0, silver_fail=0,
                      forecast_rows=10, pipeline_name="incremental_job"):
        return {
            "run_timestamp_utc": "2026-04-20T08:00",
            "pipeline_name": pipeline_name,
            "status": status,
            "bronze_failed_events": bronze_fail,
            "silver_quality_failed_checks": silver_fail,
            "forecast_rows_generated": forecast_rows,
        }

    def test_successful_run_gives_100_pct_progress(self):
        run_row = self._make_run_row()
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[run_row], []]):
            result = self.repo._pipeline_status_databricks()
        self.assertEqual(result["stage_progress"]["bronze"], 100.0)
        self.assertEqual(result["stage_progress"]["silver"], 100.0)

    def test_bronze_failures_lower_bronze_progress(self):
        run_row = self._make_run_row(bronze_fail=5)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[run_row], []]):
            result = self.repo._pipeline_status_databricks()
        self.assertLess(result["stage_progress"]["bronze"], 100.0)

    def test_zero_forecast_rows_lowers_gold_progress(self):
        run_row = self._make_run_row(forecast_rows=0)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[run_row], []]):
            result = self.repo._pipeline_status_databricks()
        self.assertLess(result["stage_progress"]["gold"], 100.0)

    def test_alerts_generated_for_non_success_rows(self):
        run_row = self._make_run_row(status="FAILED", bronze_fail=3)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[run_row], []]):
            result = self.repo._pipeline_status_databricks()
        self.assertGreater(len(result["alerts"]), 0)
        self.assertIn("pipeline_name", result["alerts"][0])

    def test_alerts_capped_at_five(self):
        run_rows = [self._make_run_row(status="FAILED", pipeline_name=f"job_{i}") for i in range(10)]
        with patch.object(self.repo, "_safe_execute_query", side_effect=[run_rows, []]):
            result = self.repo._pipeline_status_databricks()
        self.assertLessEqual(len(result["alerts"]), 5)

    def test_window_days_included(self):
        run_row = self._make_run_row()
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[run_row], []]):
            result = self.repo._pipeline_status_databricks()
        self.assertIn("window_days", result)


# ---------------------------------------------------------------------------
# 12. _build_pipeline_metrics (static)
# ---------------------------------------------------------------------------

class TestBuildPipelineMetrics(unittest.TestCase):
    def test_eta_minutes_when_behind(self):
        result = TopicRepository._build_pipeline_metrics(80.0, 1000, 800, [])
        self.assertGreater(result["eta_minutes"], 0)

    def test_no_eta_when_caught_up(self):
        result = TopicRepository._build_pipeline_metrics(100.0, 500, 500, [])
        self.assertEqual(result["eta_minutes"], 0)

    def test_serving_is_gold_minus_three(self):
        result = TopicRepository._build_pipeline_metrics(90.0, 100, 90, [])
        self.assertAlmostEqual(result["stage_progress"]["serving"],
                               round(max(90.0 - 3.0, 0.0), 2))


# ---------------------------------------------------------------------------
# 13. _forecast_72h_databricks
# ---------------------------------------------------------------------------

class TestForecast72hDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _make_forecast_rows(self, n=3):
        today = date.today()
        return [
            {"day": (today + timedelta(days=i)).isoformat(), "expected_mwh": 100.0 + i * 5}
            for i in range(n)
        ]

    def test_happy_path_returns_three_days(self):
        rows = self._make_forecast_rows(3)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[rows, []]), \
             patch.object(self.repo, "_latest_forecast_uncertainty_factor", return_value=0.10):
            result = self.repo._forecast_72h_databricks()
        self.assertEqual(len(result["daily_forecast"]), 3)

    def test_confidence_interval_present(self):
        rows = self._make_forecast_rows(3)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[rows, []]), \
             patch.object(self.repo, "_latest_forecast_uncertainty_factor", return_value=0.10):
            result = self.repo._forecast_72h_databricks()
        ci = result["daily_forecast"][0]["confidence_interval"]
        self.assertIn("low", ci)
        self.assertIn("high", ci)

    def test_fewer_than_three_rows_triggers_secondary_query(self):
        partial = self._make_forecast_rows(2)
        full = self._make_forecast_rows(3)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[partial, full]), \
             patch.object(self.repo, "_latest_forecast_uncertainty_factor", return_value=0.10):
            result = self.repo._forecast_72h_databricks()
        self.assertEqual(len(result["daily_forecast"]), 3)

    def test_empty_primary_and_secondary_marks_stale(self):
        stale = self._make_forecast_rows(3)
        with patch.object(self.repo, "_safe_execute_query", side_effect=[[], [], stale]), \
             patch.object(self.repo, "_latest_forecast_uncertainty_factor", return_value=0.10):
            result = self.repo._forecast_72h_databricks()
        self.assertTrue(result.get("forecast_stale"))

    def test_all_empty_returns_empty_forecast(self):
        with patch.object(self.repo, "_safe_execute_query", return_value=[]):
            result = self.repo._forecast_72h_databricks()
        self.assertEqual(result["daily_forecast"], [])
        self.assertTrue(result.get("forecast_stale"))


# ---------------------------------------------------------------------------
# 14. _build_forecast (static)
# ---------------------------------------------------------------------------

class TestBuildForecast(unittest.TestCase):
    def test_builds_three_day_forecast(self):
        base = date(2026, 4, 20)
        result = TopicRepository._build_forecast(base, 100.0)
        self.assertEqual(len(result["daily_forecast"]), 3)

    def test_dates_are_sequential(self):
        base = date(2026, 4, 20)
        result = TopicRepository._build_forecast(base, 100.0)
        days = [r["date"] for r in result["daily_forecast"]]
        expected = [(base + timedelta(days=i + 1)).isoformat() for i in range(3)]
        self.assertEqual(days, expected)

    def test_confidence_interval_low_less_than_expected(self):
        result = TopicRepository._build_forecast(date(2026, 4, 20), 100.0)
        for day in result["daily_forecast"]:
            self.assertLess(day["confidence_interval"]["low"], day["expected_mwh"])


# ---------------------------------------------------------------------------
# 15. _build_forecast_from_rows (static)
# ---------------------------------------------------------------------------

class TestBuildForecastFromRows(unittest.TestCase):
    def test_caps_at_three_rows(self):
        rows = [{"day": f"2026-04-{20+i}", "expected_mwh": 100.0} for i in range(5)]
        result = TopicRepository._build_forecast_from_rows(rows, 0.1)
        self.assertEqual(len(result), 3)

    def test_confidence_interval_uses_uncertainty_factor(self):
        rows = [{"day": "2026-04-21", "expected_mwh": 100.0}]
        result = TopicRepository._build_forecast_from_rows(rows, 0.2)
        self.assertAlmostEqual(result[0]["confidence_interval"]["low"], 80.0)
        self.assertAlmostEqual(result[0]["confidence_interval"]["high"], 120.0)

    def test_none_expected_mwh_treated_as_zero(self):
        rows = [{"day": "2026-04-21", "expected_mwh": None}]
        result = TopicRepository._build_forecast_from_rows(rows, 0.1)
        self.assertEqual(result[0]["expected_mwh"], 0.0)


# ---------------------------------------------------------------------------
# 16. _latest_forecast_uncertainty_factor
# ---------------------------------------------------------------------------

class TestLatestForecastUncertaintyFactor(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def test_uses_nrmse_from_db(self):
        with patch.object(self.repo, "_safe_execute_query", return_value=[{"nrmse_pct": 20.0}]):
            factor = self.repo._latest_forecast_uncertainty_factor()
        self.assertAlmostEqual(factor, 0.20)

    def test_clamps_to_minimum(self):
        with patch.object(self.repo, "_safe_execute_query", return_value=[{"nrmse_pct": 0.1}]):
            factor = self.repo._latest_forecast_uncertainty_factor()
        self.assertGreaterEqual(factor, 0.03)

    def test_clamps_to_maximum(self):
        with patch.object(self.repo, "_safe_execute_query", return_value=[{"nrmse_pct": 9999.0}]):
            factor = self.repo._latest_forecast_uncertainty_factor()
        self.assertLessEqual(factor, 0.35)

    def test_empty_rows_returns_default(self):
        with patch.object(self.repo, "_safe_execute_query", return_value=[]):
            factor = self.repo._latest_forecast_uncertainty_factor()
        self.assertAlmostEqual(factor, 0.12)


# ---------------------------------------------------------------------------
# 17. _data_quality_databricks
# ---------------------------------------------------------------------------

class TestDataQualityDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _call(self, quality_rows, issue_rows=None):
        issue_rows = issue_rows or []
        with patch.object(self.repo, "_execute_query",
                          side_effect=[quality_rows, issue_rows]), \
             patch.object(self.repo, "_resolve_latest_datetime", return_value=""):
            return self.repo._data_quality_databricks(30)

    def test_happy_path_populates_scores(self):
        quality_rows = [{"facility": "Alpha", "avg_score": 98.5}]
        result = self._call(quality_rows)
        self.assertEqual(len(result["facility_quality_scores"]), 1)
        self.assertAlmostEqual(result["facility_quality_scores"][0]["quality_score"], 98.5)

    def test_all_good_produces_summary(self):
        quality_rows = [{"facility": "Alpha", "avg_score": 99.0}]
        result = self._call(quality_rows)
        self.assertIn("summary", result)

    def test_low_score_facility_appears_in_list(self):
        quality_rows = [{"facility": "Alpha", "avg_score": 80.0}]
        issue_rows = [{"facility": "Alpha", "quality_issues": "missing_data|stale"}]
        result = self._call(quality_rows, issue_rows)
        self.assertEqual(len(result["low_score_facilities"]), 1)
        self.assertEqual(result["low_score_facilities"][0]["facility"], "Alpha")

    def test_empty_rows_returns_no_data_message(self):
        result = self._call([])
        self.assertIn("summary", result)
        self.assertEqual(result["facility_quality_scores"], [])

    def test_low_score_capped_at_five(self):
        quality_rows = [{"facility": f"F{i}", "avg_score": 50.0} for i in range(10)]
        result = self._call(quality_rows)
        self.assertLessEqual(len(result["low_score_facilities"]), 5)


# ---------------------------------------------------------------------------
# 18. _facility_info_databricks
# ---------------------------------------------------------------------------

_FAC_ROW = {
    "facility_code": "WRSF1",
    "facility_name": "Warrego Solar Farm 1",
    "region": "NSW",
    "state": "NSW",
    "location_lat": -32.5,
    "location_lng": 150.1,
    "total_capacity_mw": 100.0,
    "total_capacity_registered_mw": 98.0,
    "total_capacity_maximum_mw": 102.0,
}


class TestFacilityInfoDatabricks(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def _call(self, rows, facility_name=None):
        with patch.object(self.repo, "_safe_execute_query", return_value=rows):
            return self.repo._facility_info_databricks(facility_name)

    def test_all_facilities_when_no_name(self):
        result = self._call([_FAC_ROW])
        self.assertEqual(result["facility_count"], 1)
        self.assertEqual(result["facilities"][0]["facility_code"], "WRSF1")

    def test_facility_name_filter(self):
        result = self._call([_FAC_ROW], facility_name="WRSF1")
        self.assertEqual(result["facilities"][0]["facility_code"], "WRSF1")

    def test_empty_rows_falls_back_to_csv(self):
        """When DB returns nothing, should fall back to CSV path."""
        fallback_result = {"facility_count": 0, "facilities": []}
        with patch.object(self.repo, "_safe_execute_query", return_value=[]), \
             patch.object(self.repo, "_facility_info_csv", return_value=fallback_result) as mock_csv:
            result = self.repo._facility_info_databricks("NONEXISTENT")
        mock_csv.assert_called_once_with("NONEXISTENT")
        self.assertEqual(result["facility_count"], 0)


# ---------------------------------------------------------------------------
# 19. _build_facility_result
# ---------------------------------------------------------------------------

class TestBuildFacilityResult(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def test_eastern_timezone_derived(self):
        rows = [{**_FAC_ROW, "location_lat": -34.0, "location_lng": 151.0}]
        result = self.repo._build_facility_result(rows)
        self.assertEqual(result["facilities"][0]["timezone_name"], "Australia/Eastern")

    def test_central_timezone_derived(self):
        rows = [{**_FAC_ROW, "location_lat": -31.0, "location_lng": 133.0}]
        result = self.repo._build_facility_result(rows)
        self.assertEqual(result["facilities"][0]["timezone_name"], "Australia/Central")

    def test_western_timezone_derived(self):
        rows = [{**_FAC_ROW, "location_lat": -28.0, "location_lng": 120.0}]
        result = self.repo._build_facility_result(rows)
        self.assertEqual(result["facilities"][0]["timezone_name"], "Australia/Western")

    def test_capacity_uses_maximum_mw(self):
        result = self.repo._build_facility_result([_FAC_ROW])
        # total_capacity_maximum_mw = 102.0 takes precedence
        self.assertAlmostEqual(result["facilities"][0]["total_capacity_mw"], 102.0)

    def test_empty_rows_returns_zero_count(self):
        result = self.repo._build_facility_result([])
        self.assertEqual(result["facility_count"], 0)


# ---------------------------------------------------------------------------
# 20. _derive_timezone_from_coordinates
# ---------------------------------------------------------------------------

class TestDeriveTimezoneFromCoordinates(unittest.TestCase):
    def test_eastern_australia(self):
        name, offset = TopicRepository._derive_timezone_from_coordinates(-34.0, 150.0)
        self.assertEqual(name, "Australia/Eastern")
        self.assertEqual(offset, "UTC+10:00")

    def test_central_australia(self):
        name, offset = TopicRepository._derive_timezone_from_coordinates(-31.0, 133.0)
        self.assertEqual(name, "Australia/Central")
        self.assertEqual(offset, "UTC+09:30")

    def test_western_australia(self):
        name, offset = TopicRepository._derive_timezone_from_coordinates(-28.0, 120.0)
        self.assertEqual(name, "Australia/Western")
        self.assertEqual(offset, "UTC+08:00")

    def test_out_of_scope_coordinates(self):
        name, offset = TopicRepository._derive_timezone_from_coordinates(40.7, -74.0)
        self.assertEqual(name, "UTC (approx)")
        self.assertIn("UTC", offset)


# ---------------------------------------------------------------------------
# 21. _format_utc_offset (static)
# ---------------------------------------------------------------------------

class TestFormatUtcOffset(unittest.TestCase):
    def test_positive_integer_hours(self):
        self.assertEqual(TopicRepository._format_utc_offset(10.0), "UTC+10:00")

    def test_negative_hours(self):
        self.assertEqual(TopicRepository._format_utc_offset(-5.0), "UTC-05:00")

    def test_fractional_half_hour(self):
        self.assertEqual(TopicRepository._format_utc_offset(9.5), "UTC+09:30")

    def test_zero_offset(self):
        self.assertEqual(TopicRepository._format_utc_offset(0.0), "UTC+00:00")


# ---------------------------------------------------------------------------
# 22. _collect_pipeline_alerts
# ---------------------------------------------------------------------------

class TestCollectPipelineAlerts(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo()

    def test_good_rows_produce_no_alerts(self):
        rows = [{"quality_flag": "GOOD", "quality_issues": "", "facility_name": "Alpha"}]
        alerts = self.repo._collect_pipeline_alerts(rows, [], [])
        self.assertEqual(alerts, [])

    def test_bad_flag_produces_alert(self):
        rows = [{"quality_flag": "BAD", "quality_issues": "stale_data", "facility_name": "Alpha"}]
        alerts = self.repo._collect_pipeline_alerts(rows, [], [])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["quality_flag"], "BAD")

    def test_alerts_capped_at_five(self):
        rows = [{"quality_flag": "BAD", "quality_issues": "err", "facility_name": f"F{i}"} for i in range(10)]
        alerts = self.repo._collect_pipeline_alerts(rows, [], [])
        self.assertLessEqual(len(alerts), 5)


if __name__ == "__main__":
    unittest.main()
