"""Unit tests for app.services.databricks_service — all external calls mocked."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(warehouse_id: str | None = "wh-abc123") -> MagicMock:
    s = MagicMock()
    s.databricks_host = "https://fake.azuredatabricks.net"
    s.databricks_token = "dapi-fake-token"
    s.databricks_warehouse_id = warehouse_id
    return s


def _make_client() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# get_databricks_client
# ---------------------------------------------------------------------------

class TestGetDatabricksClient(unittest.TestCase):
    def test_returns_workspace_client(self) -> None:
        mock_settings = _make_settings()
        mock_ws_client = MagicMock()

        with patch(
            "app.services.databricks_service.get_solar_chat_settings",
            return_value=mock_settings,
        ), patch(
            "app.services.databricks_service.WorkspaceClient",
            return_value=mock_ws_client,
        ) as MockWS:
            # Clear lru_cache so we get a fresh call
            import app.services.databricks_service as svc
            svc.get_databricks_client.cache_clear()
            result = svc.get_databricks_client()

            MockWS.assert_called_once_with(
                host=mock_settings.databricks_host,
                token=mock_settings.databricks_token,
            )
            self.assertIs(result, mock_ws_client)
            svc.get_databricks_client.cache_clear()


# ---------------------------------------------------------------------------
# get_job_info
# ---------------------------------------------------------------------------

class TestGetJobInfo(unittest.TestCase):
    def test_returns_job_dict(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_job = MagicMock()
        mock_job.as_dict.return_value = {"job_id": 42, "name": "test-job"}
        mock_client.jobs.get.return_value = mock_job

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.get_job_info(42)

        mock_client.jobs.get.assert_called_once_with(job_id=42)
        self.assertEqual(result, {"job_id": 42, "name": "test-job"})

    def test_propagates_sdk_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.get.side_effect = RuntimeError("SDK error")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                svc.get_job_info(99)


# ---------------------------------------------------------------------------
# get_job_runs
# ---------------------------------------------------------------------------

class TestGetJobRuns(unittest.TestCase):
    def test_returns_list_of_run_dicts(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        run1, run2 = MagicMock(), MagicMock()
        run1.as_dict.return_value = {"run_id": 1}
        run2.as_dict.return_value = {"run_id": 2}
        mock_client.jobs.list_runs.return_value = [run1, run2]

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.get_job_runs(10, limit=5)

        mock_client.jobs.list_runs.assert_called_once_with(job_id=10, limit=5)
        self.assertEqual(result, [{"run_id": 1}, {"run_id": 2}])

    def test_empty_run_list(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.list_runs.return_value = []

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.get_job_runs(10)

        self.assertEqual(result, [])

    def test_propagates_sdk_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.list_runs.side_effect = ConnectionError("network")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(ConnectionError):
                svc.get_job_runs(10)


# ---------------------------------------------------------------------------
# trigger_job_run
# ---------------------------------------------------------------------------

class TestTriggerJobRun(unittest.TestCase):
    def test_returns_run_id(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        run_response = MagicMock()
        run_response.run_id = 777
        mock_client.jobs.run_now.return_value = run_response

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.trigger_job_run(5)

        mock_client.jobs.run_now.assert_called_once_with(job_id=5)
        self.assertEqual(result, {"run_id": 777})

    def test_propagates_sdk_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.run_now.side_effect = PermissionError("not allowed")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(PermissionError):
                svc.trigger_job_run(5)


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------

class TestGetRun(unittest.TestCase):
    def test_returns_run_dict(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        run_obj = MagicMock()
        run_obj.as_dict.return_value = {"run_id": 100, "state": "SUCCESS"}
        mock_client.jobs.get_run.return_value = run_obj

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.get_run(100)

        mock_client.jobs.get_run.assert_called_once_with(run_id=100)
        self.assertEqual(result, {"run_id": 100, "state": "SUCCESS"})

    def test_propagates_sdk_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.get_run.side_effect = ValueError("bad run id")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(ValueError):
                svc.get_run(0)


# ---------------------------------------------------------------------------
# cancel_job_run
# ---------------------------------------------------------------------------

class TestCancelJobRun(unittest.TestCase):
    def test_returns_success_message(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.cancel_job_run(55)

        mock_client.jobs.cancel_run.assert_called_once_with(run_id=55)
        self.assertEqual(result, {"message": "Run 55 cancelled successfully"})

    def test_propagates_sdk_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_client.jobs.cancel_run.side_effect = RuntimeError("cancel failed")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                svc.cancel_job_run(55)


# ---------------------------------------------------------------------------
# update_job_schedule
# ---------------------------------------------------------------------------

class TestUpdateJobSchedule(unittest.TestCase):
    def _mock_client_with_schedule(
        self,
        tz: str = "UTC",
        pause: str | None = "UNPAUSED",
    ) -> MagicMock:
        mock_client = _make_client()

        existing_schedule = MagicMock()
        existing_schedule.timezone_id = tz
        existing_schedule.pause_status = MagicMock()
        existing_schedule.pause_status.value = pause

        job_obj = MagicMock()
        job_obj.settings = MagicMock()
        job_obj.settings.schedule = existing_schedule

        updated_schedule = MagicMock()
        updated_schedule.quartz_cron_expression = "0 0 * * *"
        updated_schedule.timezone_id = tz
        updated_schedule.pause_status = MagicMock()
        updated_schedule.pause_status.value = "UNPAUSED"

        updated_job = MagicMock()
        updated_job.settings = MagicMock()
        updated_job.settings.schedule = updated_schedule

        mock_client.jobs.get.side_effect = [job_obj, updated_job]
        return mock_client

    def test_happy_path_inherits_existing_schedule(self) -> None:
        import app.services.databricks_service as svc

        mock_client = self._mock_client_with_schedule(tz="Australia/Sydney", pause="UNPAUSED")

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.update_job_schedule(7, "0 0 * * *")

        self.assertEqual(result["job_id"], 7)
        self.assertEqual(result["message"], "Schedule updated successfully")
        self.assertIn("schedule", result)

    def test_explicit_timezone_and_pause_status(self) -> None:
        import app.services.databricks_service as svc

        mock_client = self._mock_client_with_schedule()

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.update_job_schedule(
                7,
                "0 12 * * *",
                timezone_id="US/Eastern",
                pause_status="PAUSED",
            )

        self.assertEqual(result["job_id"], 7)

    def test_invalid_pause_status_raises_value_error(self) -> None:
        import app.services.databricks_service as svc

        mock_client = self._mock_client_with_schedule()

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            with self.assertRaises(ValueError):
                svc.update_job_schedule(7, "0 0 * * *", pause_status="INVALID")

    def test_no_existing_schedule_uses_utc_default(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        job_obj = MagicMock()
        job_obj.settings = None

        updated_schedule = MagicMock()
        updated_schedule.quartz_cron_expression = "0 0 * * *"
        updated_schedule.timezone_id = "UTC"
        updated_schedule.pause_status = MagicMock()
        updated_schedule.pause_status.value = "UNPAUSED"
        updated_job = MagicMock()
        updated_job.settings = MagicMock()
        updated_job.settings.schedule = updated_schedule

        mock_client.jobs.get.side_effect = [job_obj, updated_job]

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.update_job_schedule(99, "0 0 * * *")

        self.assertEqual(result["schedule"]["timezone_id"], "UTC")

    def test_updated_job_has_no_settings_falls_back_to_resolved(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        existing_schedule = MagicMock()
        existing_schedule.timezone_id = "UTC"
        existing_schedule.pause_status = MagicMock()
        existing_schedule.pause_status.value = "UNPAUSED"

        job_obj = MagicMock()
        job_obj.settings = MagicMock()
        job_obj.settings.schedule = existing_schedule

        updated_job = MagicMock()
        updated_job.settings = None

        mock_client.jobs.get.side_effect = [job_obj, updated_job]

        with patch.object(svc, "get_databricks_client", return_value=mock_client):
            result = svc.update_job_schedule(3, "0 6 * * *")

        self.assertEqual(result["job_id"], 3)
        self.assertIn("schedule", result)


# ---------------------------------------------------------------------------
# execute_sql
# ---------------------------------------------------------------------------

class TestExecuteSql(unittest.TestCase):
    def _make_response(self, columns: list[str], rows: list[list]) -> MagicMock:
        col_mocks = [MagicMock(name=c) for c in columns]
        for cm, name in zip(col_mocks, columns):
            cm.name = name

        schema = MagicMock()
        schema.columns = col_mocks

        manifest = MagicMock()
        manifest.schema = schema

        result_obj = MagicMock()
        result_obj.data_array = rows

        response = MagicMock()
        response.manifest = manifest
        response.result = result_obj
        return response

    def test_returns_list_of_dicts(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_settings = _make_settings()
        response = self._make_response(["a", "b"], [["v1", "v2"], ["v3", "v4"]])
        mock_client.statement_execution.execute_statement.return_value = response

        with patch.object(svc, "get_databricks_client", return_value=mock_client), \
             patch.object(svc, "get_solar_chat_settings", return_value=mock_settings):
            result = svc.execute_sql("SELECT 1")

        self.assertEqual(result, [{"a": "v1", "b": "v2"}, {"a": "v3", "b": "v4"}])

    def test_raises_when_no_warehouse_id(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_settings = _make_settings(warehouse_id=None)

        with patch.object(svc, "get_databricks_client", return_value=mock_client), \
             patch.object(svc, "get_solar_chat_settings", return_value=mock_settings):
            with self.assertRaises(ValueError, msg="DATABRICKS_WAREHOUSE_ID is not set."):
                svc.execute_sql("SELECT 1")

    def test_returns_empty_when_no_result(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_settings = _make_settings()
        response = MagicMock()
        response.manifest = None
        response.result = None
        mock_client.statement_execution.execute_statement.return_value = response

        with patch.object(svc, "get_databricks_client", return_value=mock_client), \
             patch.object(svc, "get_solar_chat_settings", return_value=mock_settings):
            result = svc.execute_sql("SELECT 1")

        self.assertEqual(result, [])

    def test_returns_empty_when_response_none(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_settings = _make_settings()
        mock_client.statement_execution.execute_statement.return_value = None

        with patch.object(svc, "get_databricks_client", return_value=mock_client), \
             patch.object(svc, "get_solar_chat_settings", return_value=mock_settings):
            result = svc.execute_sql("SELECT 1")

        self.assertEqual(result, [])

    def test_empty_data_array_returns_empty(self) -> None:
        import app.services.databricks_service as svc

        mock_client = _make_client()
        mock_settings = _make_settings()
        response = self._make_response(["col1"], [])
        response.result.data_array = None
        mock_client.statement_execution.execute_statement.return_value = response

        with patch.object(svc, "get_databricks_client", return_value=mock_client), \
             patch.object(svc, "get_solar_chat_settings", return_value=mock_settings):
            result = svc.execute_sql("SELECT col1")

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# get_quality_summary_metrics
# ---------------------------------------------------------------------------

class TestGetQualitySummaryMetrics(unittest.TestCase):
    def test_with_data_computes_ratios(self) -> None:
        import app.services.databricks_service as svc

        rows = [{"valid": "800", "warning": "150", "invalid": "50"}]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_quality_summary_metrics()

        self.assertEqual(result["valid"], "800")
        self.assertEqual(result["warning"], "150")
        self.assertEqual(result["invalid"], "50")
        self.assertIn("of total", result["valid_ratio"])
        self.assertIn("of total", result["warning_ratio"])
        self.assertIn("of total", result["invalid_ratio"])

    def test_empty_results_returns_zeros(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_quality_summary_metrics()

        self.assertEqual(result["valid"], "0")
        self.assertEqual(result["warning"], "0")
        self.assertEqual(result["invalid"], "0")
        self.assertEqual(result["valid_ratio"], "0.0% of total")

    def test_none_values_treated_as_zero(self) -> None:
        import app.services.databricks_service as svc

        rows = [{"valid": None, "warning": None, "invalid": None}]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_quality_summary_metrics()

        self.assertEqual(result["valid"], "0")

    def test_large_numbers_formatted_with_comma(self) -> None:
        import app.services.databricks_service as svc

        rows = [{"valid": "1000000", "warning": "0", "invalid": "0"}]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_quality_summary_metrics()

        self.assertIn(",", result["valid"])


# ---------------------------------------------------------------------------
# get_facility_quality_scores
# ---------------------------------------------------------------------------

class TestGetFacilityQualityScores(unittest.TestCase):
    def test_computes_percentages(self) -> None:
        import app.services.databricks_service as svc

        rows = [
            {"facility": "FAC1", "total": "100", "valid": "80", "warning": "15", "invalid": "5"},
        ]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_facility_quality_scores()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["facility"], "FAC1")
        self.assertEqual(result[0]["valid"], "80.0%")
        self.assertEqual(result[0]["score"], "80")

    def test_skips_zero_total_rows(self) -> None:
        import app.services.databricks_service as svc

        rows = [{"facility": "FAC_EMPTY", "total": "0", "valid": "0", "warning": "0", "invalid": "0"}]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_facility_quality_scores()

        self.assertEqual(result, [])

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_facility_quality_scores()

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# get_recent_quality_issues
# ---------------------------------------------------------------------------

class TestGetRecentQualityIssues(unittest.TestCase):
    def test_maps_row_to_issue_dict(self) -> None:
        import app.services.databricks_service as svc

        rows = [
            {
                "time": "10:00 UTC",
                "facility": "pv.silver.energy_readings",
                "sensor": "null_rate",
                "issue": "null_rate_check",
                "affected": "12 records",
                "severity": "WARNING",
                "action": "Flagged",
            }
        ]
        with patch.object(svc, "execute_sql", return_value=rows):
            result = svc.get_recent_quality_issues()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["time"], "10:00 UTC")
        self.assertEqual(result[0]["severity"], "WARNING")

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_recent_quality_issues()

        self.assertEqual(result, [])

    def test_missing_keys_default_to_empty_string(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[{}]):
            result = svc.get_recent_quality_issues()

        self.assertEqual(result[0]["time"], "")
        self.assertEqual(result[0]["severity"], "")


# ---------------------------------------------------------------------------
# get_facility_heatmap_data
# ---------------------------------------------------------------------------

class TestGetFacilityHeatmapData(unittest.TestCase):
    def test_returns_raw_results(self) -> None:
        import app.services.databricks_service as svc

        expected = [{"date": "2024-01-01", "facility": "FAC1", "score": 95.0}]
        with patch.object(svc, "execute_sql", return_value=expected):
            result = svc.get_facility_heatmap_data()

        self.assertEqual(result, expected)

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_facility_heatmap_data()

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# get_daily_forecast
# ---------------------------------------------------------------------------

class TestGetDailyForecast(unittest.TestCase):
    def test_returns_raw_results(self) -> None:
        import app.services.databricks_service as svc

        expected = [{"date": "2024-06-01", "actual": 10.5, "predicted": 11.0}]
        with patch.object(svc, "execute_sql", return_value=expected):
            result = svc.get_daily_forecast()

        self.assertEqual(result, expected)

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_daily_forecast()

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# get_model_monitoring_metrics
# ---------------------------------------------------------------------------

class TestGetModelMonitoringMetrics(unittest.TestCase):
    def test_returns_raw_results(self) -> None:
        import app.services.databricks_service as svc

        expected = [{"date": "2024-06-01", "rmse": 1.2, "r2": 0.95}]
        with patch.object(svc, "execute_sql", return_value=expected):
            result = svc.get_model_monitoring_metrics()

        self.assertEqual(result, expected)

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_model_monitoring_metrics()

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# get_registry_models
# ---------------------------------------------------------------------------

class TestGetRegistryModels(unittest.TestCase):
    def test_returns_raw_results(self) -> None:
        import app.services.databricks_service as svc

        expected = [{"version": "v1", "algorithm": "GBT", "status": "Production"}]
        with patch.object(svc, "execute_sql", return_value=expected):
            result = svc.get_registry_models()

        self.assertEqual(result, expected)

    def test_empty_results(self) -> None:
        import app.services.databricks_service as svc

        with patch.object(svc, "execute_sql", return_value=[]):
            result = svc.get_registry_models()

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
