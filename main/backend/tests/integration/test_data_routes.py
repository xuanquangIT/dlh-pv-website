"""Integration tests for data pipeline, data quality, forecast and ML training routes.

Covers:
  Data Pipeline
  - GET  /data-pipeline/status
  - GET  /data-pipeline/jobs
  - GET  /data-pipeline/jobs/runs
  - POST /data-pipeline/jobs/run
  - GET  /data-pipeline/jobs/runs/{run_id}
  - POST /data-pipeline/jobs/runs/{run_id}/cancel
  - POST /data-pipeline/jobs/schedule

  Data Quality
  - GET  /data-quality/summary
  - GET  /data-quality/facility-scores
  - GET  /data-quality/recent-issues
  - GET  /data-quality/heatmap-data

  Forecast
  - GET  /forecast/summary-kpi
  - GET  /forecast/daily

  ML Training
  - GET  /ml-training/monitoring
"""
import os
import sys
import uuid
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-integration")
os.environ.setdefault("AUTH_COOKIE_NAME", "pv_access_token")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

from app.api.dependencies import get_current_user
from app.main import create_app

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000020")


def _make_user(role_id: str = "admin") -> SimpleNamespace:
    return SimpleNamespace(
        id=USER_ID,
        username="testuser",
        email="testuser@example.com",
        full_name="Test User",
        role_id=role_id,
        role=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class DataPipelineStatusTests(unittest.TestCase):
    """Tests for GET /data-pipeline/status."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_status_data_engineer_returns_200(self) -> None:
        self._set_role("data_engineer")
        response = self.client.get("/data-pipeline/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["module"], "data_pipeline")

    def test_status_system_role_returns_200(self) -> None:
        self._set_role("system")
        response = self.client.get("/data-pipeline/status")
        self.assertEqual(response.status_code, 200)

    def test_status_admin_role_forbidden(self) -> None:
        self._set_role("admin")
        response = self.client.get("/data-pipeline/status")
        self.assertEqual(response.status_code, 403)

    def test_status_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.get("/data-pipeline/status")
        self.assertEqual(response.status_code, 403)

    def test_status_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-pipeline/status")
        self.assertEqual(response.status_code, 403)

    def test_status_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/data-pipeline/status")
        self.assertIn(response.status_code, (401, 403))


class DataPipelineJobsTests(unittest.TestCase):
    """Tests for job-related pipeline routes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    # --- GET /jobs ---

    def test_get_jobs_admin_success(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.get_job_info", return_value={"job_id": 1}):
            response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], 1)

    def test_get_jobs_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch("app.api.data_pipeline.routes.get_job_info", return_value={"job_id": 2}):
            response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 200)

    def test_get_jobs_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 403)

    def test_get_jobs_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 403)

    def test_get_jobs_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 403)

    def test_get_jobs_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.get_job_info", side_effect=Exception("Databricks down")):
            response = self.client.get("/data-pipeline/jobs")
        self.assertEqual(response.status_code, 500)
        self.assertIn("Databricks down", response.json()["detail"])

    def test_get_jobs_custom_job_id(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.get_job_info", return_value={"job_id": 9999}) as mock:
            response = self.client.get("/data-pipeline/jobs?job_id=9999")
        self.assertEqual(response.status_code, 200)

    # --- GET /jobs/runs ---

    def test_get_runs_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.get_job_runs",
            return_value={"runs": [{"run_id": 1}]},
        ):
            response = self.client.get("/data-pipeline/jobs/runs")
        self.assertEqual(response.status_code, 200)

    def test_get_runs_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.data_pipeline.routes.get_job_runs",
            return_value={"runs": []},
        ):
            response = self.client.get("/data-pipeline/jobs/runs")
        self.assertEqual(response.status_code, 200)

    def test_get_runs_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-pipeline/jobs/runs")
        self.assertEqual(response.status_code, 403)

    def test_get_runs_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.get_job_runs",
            side_effect=Exception("Connection refused"),
        ):
            response = self.client.get("/data-pipeline/jobs/runs")
        self.assertEqual(response.status_code, 500)

    def test_get_runs_custom_limit(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.get_job_runs",
            return_value={"runs": []},
        ) as mock_fn:
            response = self.client.get("/data-pipeline/jobs/runs?limit=10")
        self.assertEqual(response.status_code, 200)

    # --- POST /jobs/run ---

    def test_run_job_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.trigger_job_run",
            return_value={"run_id": 777},
        ):
            response = self.client.post("/data-pipeline/jobs/run")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], 777)

    def test_run_job_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch("app.api.data_pipeline.routes.trigger_job_run", return_value={"run_id": 888}):
            response = self.client.post("/data-pipeline/jobs/run")
        self.assertEqual(response.status_code, 200)

    def test_run_job_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.post("/data-pipeline/jobs/run")
        self.assertEqual(response.status_code, 403)

    def test_run_job_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.trigger_job_run",
            side_effect=Exception("Trigger failed"),
        ):
            response = self.client.post("/data-pipeline/jobs/run")
        self.assertEqual(response.status_code, 500)

    # --- GET /jobs/runs/{run_id} ---

    def test_get_run_details_admin_success(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.get_run", return_value={"run_id": 42, "state": "SUCCESS"}):
            response = self.client.get("/data-pipeline/jobs/runs/42")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], 42)

    def test_get_run_details_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch("app.api.data_pipeline.routes.get_run", return_value={"run_id": 1}):
            response = self.client.get("/data-pipeline/jobs/runs/1")
        self.assertEqual(response.status_code, 200)

    def test_get_run_details_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-pipeline/jobs/runs/1")
        self.assertEqual(response.status_code, 403)

    def test_get_run_details_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.get_run", side_effect=Exception("Not found")):
            response = self.client.get("/data-pipeline/jobs/runs/999")
        self.assertEqual(response.status_code, 500)

    def test_get_run_details_invalid_run_id_returns_422(self) -> None:
        self._set_role("admin")
        response = self.client.get("/data-pipeline/jobs/runs/not-an-int")
        self.assertEqual(response.status_code, 422)

    # --- POST /jobs/runs/{run_id}/cancel ---

    def test_cancel_run_admin_success(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.cancel_job_run", return_value={"cancelled": True}):
            response = self.client.post("/data-pipeline/jobs/runs/42/cancel")
        self.assertEqual(response.status_code, 200)

    def test_cancel_run_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch("app.api.data_pipeline.routes.cancel_job_run", return_value={}):
            response = self.client.post("/data-pipeline/jobs/runs/1/cancel")
        self.assertEqual(response.status_code, 200)

    def test_cancel_run_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.post("/data-pipeline/jobs/runs/1/cancel")
        self.assertEqual(response.status_code, 403)

    def test_cancel_run_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.cancel_job_run",
            side_effect=Exception("Already completed"),
        ):
            response = self.client.post("/data-pipeline/jobs/runs/1/cancel")
        self.assertEqual(response.status_code, 500)


class DataPipelineScheduleTests(unittest.TestCase):
    """Tests for POST /data-pipeline/jobs/schedule."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    _VALID_PAYLOAD = {
        "quartz_cron_expression": "0 00 03 ? * 2",
        "timezone_id": "Asia/Ho_Chi_Minh",
        "pause_status": "UNPAUSED",
    }

    def test_update_schedule_admin_success(self) -> None:
        self._set_role("admin")
        with patch("app.api.data_pipeline.routes.update_job_schedule", return_value={"ok": True}):
            response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)

    def test_update_schedule_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch("app.api.data_pipeline.routes.update_job_schedule", return_value={"ok": True}):
            response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)

    def test_update_schedule_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 403)

    def test_update_schedule_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 403)

    def test_update_schedule_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 403)

    def test_update_schedule_invalid_pause_status_returns_422(self) -> None:
        self._set_role("data_engineer")
        bad_payload = {**self._VALID_PAYLOAD, "pause_status": "INVALID"}
        response = self.client.post("/data-pipeline/jobs/schedule", json=bad_payload)
        self.assertEqual(response.status_code, 422)

    def test_update_schedule_missing_cron_returns_422(self) -> None:
        self._set_role("data_engineer")
        bad_payload = {"timezone_id": "UTC", "pause_status": "PAUSED"}
        response = self.client.post("/data-pipeline/jobs/schedule", json=bad_payload)
        self.assertEqual(response.status_code, 422)

    def test_update_schedule_cron_too_short_returns_422(self) -> None:
        self._set_role("data_engineer")
        bad_payload = {**self._VALID_PAYLOAD, "quartz_cron_expression": "0 0"}
        response = self.client.post("/data-pipeline/jobs/schedule", json=bad_payload)
        self.assertEqual(response.status_code, 422)

    def test_update_schedule_service_value_error_returns_400(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.update_job_schedule",
            side_effect=ValueError("Invalid cron"),
        ):
            response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid cron", response.json()["detail"])

    def test_update_schedule_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_pipeline.routes.update_job_schedule",
            side_effect=Exception("Databricks error"),
        ):
            response = self.client.post("/data-pipeline/jobs/schedule", json=self._VALID_PAYLOAD)
        self.assertEqual(response.status_code, 500)

    def test_update_schedule_optional_fields_omitted(self) -> None:
        self._set_role("admin")
        minimal_payload = {"quartz_cron_expression": "0 00 03 ? * 2"}
        with patch("app.api.data_pipeline.routes.update_job_schedule", return_value={"ok": True}):
            response = self.client.post("/data-pipeline/jobs/schedule", json=minimal_payload)
        self.assertEqual(response.status_code, 200)

    def test_update_schedule_pause_only(self) -> None:
        self._set_role("admin")
        pause_payload = {
            "quartz_cron_expression": "0 00 03 ? * 2",
            "pause_status": "PAUSED",
        }
        with patch("app.api.data_pipeline.routes.update_job_schedule", return_value={"ok": True}):
            response = self.client.post("/data-pipeline/jobs/schedule", json=pause_payload)
        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

class DataQualitySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_summary_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_quality.routes.get_quality_summary_metrics",
            return_value={"total_checks": 100, "pass_rate": 0.97},
        ):
            response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 200)
        self.assertIn("total_checks", response.json())

    def test_summary_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.data_quality.routes.get_quality_summary_metrics",
            return_value={"total_checks": 80, "pass_rate": 0.95},
        ):
            response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 200)

    def test_summary_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 403)

    def test_summary_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 403)

    def test_summary_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 403)

    def test_summary_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_quality.routes.get_quality_summary_metrics",
            side_effect=Exception("Connection lost"),
        ):
            response = self.client.get("/data-quality/summary")
        self.assertEqual(response.status_code, 500)
        self.assertIn("quality summary", response.json()["detail"])

    def test_summary_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/data-quality/summary")
        self.assertIn(response.status_code, (401, 403))


class DataQualityFacilityScoresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_facility_scores_admin_success(self) -> None:
        self._set_role("admin")
        mock_data = [{"facility_id": "WRSF1", "score": 0.99}]
        with patch(
            "app.api.data_quality.routes.get_facility_quality_scores",
            return_value=mock_data,
        ):
            response = self.client.get("/data-quality/facility-scores")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertEqual(body[0]["facility_id"], "WRSF1")

    def test_facility_scores_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.data_quality.routes.get_facility_quality_scores",
            return_value=[],
        ):
            response = self.client.get("/data-quality/facility-scores")
        self.assertEqual(response.status_code, 200)

    def test_facility_scores_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.get("/data-quality/facility-scores")
        self.assertEqual(response.status_code, 403)

    def test_facility_scores_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_quality.routes.get_facility_quality_scores",
            side_effect=Exception("Query failed"),
        ):
            response = self.client.get("/data-quality/facility-scores")
        self.assertEqual(response.status_code, 500)
        self.assertIn("facility scores", response.json()["detail"])


class DataQualityRecentIssuesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_recent_issues_admin_success(self) -> None:
        self._set_role("admin")
        mock_issues = [{"issue": "null_rate", "table": "silver.energy_readings"}]
        with patch(
            "app.api.data_quality.routes.get_recent_quality_issues",
            return_value=mock_issues,
        ):
            response = self.client.get("/data-quality/recent-issues")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_recent_issues_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.data_quality.routes.get_recent_quality_issues",
            return_value=[],
        ):
            response = self.client.get("/data-quality/recent-issues")
        self.assertEqual(response.status_code, 200)

    def test_recent_issues_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-quality/recent-issues")
        self.assertEqual(response.status_code, 403)

    def test_recent_issues_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_quality.routes.get_recent_quality_issues",
            side_effect=Exception("Timeout"),
        ):
            response = self.client.get("/data-quality/recent-issues")
        self.assertEqual(response.status_code, 500)
        self.assertIn("recent issues", response.json()["detail"])


class DataQualityHeatmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_heatmap_admin_success(self) -> None:
        self._set_role("admin")
        mock_data = [{"facility_id": "WRSF1", "date": "2026-04-01", "score": 0.98}]
        with patch(
            "app.api.data_quality.routes.get_facility_heatmap_data",
            return_value=mock_data,
        ):
            response = self.client.get("/data-quality/heatmap-data")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)

    def test_heatmap_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.data_quality.routes.get_facility_heatmap_data",
            return_value=[],
        ):
            response = self.client.get("/data-quality/heatmap-data")
        self.assertEqual(response.status_code, 200)

    def test_heatmap_ml_engineer_forbidden(self) -> None:
        self._set_role("ml_engineer")
        response = self.client.get("/data-quality/heatmap-data")
        self.assertEqual(response.status_code, 403)

    def test_heatmap_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/data-quality/heatmap-data")
        self.assertEqual(response.status_code, 403)

    def test_heatmap_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.data_quality.routes.get_facility_heatmap_data",
            side_effect=Exception("Databricks timeout"),
        ):
            response = self.client.get("/data-quality/heatmap-data")
        self.assertEqual(response.status_code, 500)
        self.assertIn("heatmap data", response.json()["detail"])

    def test_heatmap_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/data-quality/heatmap-data")
        self.assertIn(response.status_code, (401, 403))


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

_MONITORING_METRICS = [
    {"date": "2026-04-20", "rmse": 0.5, "mae": 0.3, "r2": 0.92, "mape": 5.0, "skill_score": 0.8},
    {"date": "2026-04-21", "rmse": 0.4, "mae": 0.2, "r2": 0.95, "mape": 4.0, "skill_score": 0.85},
]

_DAILY_FORECAST = [
    {"date": "2026-04-22", "facility_id": "WRSF1", "predicted_kwh": 120.5},
    {"date": "2026-04-23", "facility_id": "WRSF1", "predicted_kwh": 130.0},
]


class ForecastSummaryKpiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_summary_kpi_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        # Should return last element (latest)
        self.assertEqual(body["date"], "2026-04-21")

    def test_summary_kpi_analyst_success(self) -> None:
        self._set_role("analyst")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 200)

    def test_summary_kpi_ml_engineer_success(self) -> None:
        self._set_role("ml_engineer")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 200)

    def test_summary_kpi_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 200)

    def test_summary_kpi_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 403)

    def test_summary_kpi_empty_metrics_returns_na(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            return_value=[],
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["rmse"], "N/A")

    def test_summary_kpi_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.forecast.routes.get_model_monitoring_metrics",
            side_effect=Exception("Databricks error"),
        ):
            response = self.client.get("/forecast/summary-kpi")
        self.assertEqual(response.status_code, 500)
        self.assertIn("forecast KPI", response.json()["detail"])

    def test_summary_kpi_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/forecast/summary-kpi")
        self.assertIn(response.status_code, (401, 403))


class ForecastDailyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_daily_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.forecast.routes.get_daily_forecast",
            return_value=_DAILY_FORECAST,
        ):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertEqual(len(body), 2)

    def test_daily_analyst_success(self) -> None:
        self._set_role("analyst")
        with patch(
            "app.api.forecast.routes.get_daily_forecast",
            return_value=_DAILY_FORECAST,
        ):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 200)

    def test_daily_ml_engineer_success(self) -> None:
        self._set_role("ml_engineer")
        with patch(
            "app.api.forecast.routes.get_daily_forecast",
            return_value=_DAILY_FORECAST,
        ):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 200)

    def test_daily_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.forecast.routes.get_daily_forecast",
            return_value=_DAILY_FORECAST,
        ):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 200)

    def test_daily_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 403)

    def test_daily_empty_returns_empty_list(self) -> None:
        self._set_role("admin")
        with patch("app.api.forecast.routes.get_daily_forecast", return_value=[]):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_daily_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.forecast.routes.get_daily_forecast",
            side_effect=Exception("No data"),
        ):
            response = self.client.get("/forecast/daily")
        self.assertEqual(response.status_code, 500)
        self.assertIn("daily forecast", response.json()["detail"])

    def test_daily_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/forecast/daily")
        self.assertIn(response.status_code, (401, 403))


# ---------------------------------------------------------------------------
# ML Training
# ---------------------------------------------------------------------------

class MlTrainingMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)

    def test_monitoring_admin_success(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertEqual(len(body), 2)
        self.assertEqual(body[0]["date"], "2026-04-20")

    def test_monitoring_ml_engineer_success(self) -> None:
        self._set_role("ml_engineer")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 200)

    def test_monitoring_data_engineer_success(self) -> None:
        self._set_role("data_engineer")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 200)

    def test_monitoring_analyst_forbidden(self) -> None:
        self._set_role("analyst")
        response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 403)

    def test_monitoring_system_forbidden(self) -> None:
        self._set_role("system")
        response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 403)

    def test_monitoring_empty_list_returns_200(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            return_value=[],
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_monitoring_service_error_returns_500(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            side_effect=Exception("Databricks unreachable"),
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 500)
        self.assertIn("ML monitoring", response.json()["detail"])

    def test_monitoring_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/ml-training/monitoring")
        self.assertIn(response.status_code, (401, 403))

    def test_monitoring_returns_list_with_expected_keys(self) -> None:
        self._set_role("admin")
        with patch(
            "app.api.ml_training.routes.get_model_monitoring_metrics",
            return_value=_MONITORING_METRICS,
        ):
            response = self.client.get("/ml-training/monitoring")
        self.assertEqual(response.status_code, 200)
        item = response.json()[0]
        for key in ("date", "rmse", "mae", "r2", "mape", "skill_score"):
            with self.subTest(key=key):
                self.assertIn(key, item)


# ---------------------------------------------------------------------------
# Cross-cutting: unauthenticated access on all data routes
# ---------------------------------------------------------------------------

class UnauthenticatedAccessTests(unittest.TestCase):
    """All data routes require authentication — spot-check each module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app, raise_server_exceptions=False)

    _ENDPOINTS = [
        ("GET", "/data-pipeline/status"),
        ("GET", "/data-pipeline/jobs"),
        ("GET", "/data-pipeline/jobs/runs"),
        ("POST", "/data-pipeline/jobs/run"),
        ("GET", "/data-quality/summary"),
        ("GET", "/data-quality/facility-scores"),
        ("GET", "/data-quality/recent-issues"),
        ("GET", "/data-quality/heatmap-data"),
        ("GET", "/forecast/summary-kpi"),
        ("GET", "/forecast/daily"),
        ("GET", "/ml-training/monitoring"),
    ]

    def test_all_endpoints_reject_unauthenticated(self) -> None:
        for method, path in self._ENDPOINTS:
            with self.subTest(method=method, path=path):
                if method == "GET":
                    response = self.client.get(path)
                else:
                    response = self.client.post(path)
                self.assertIn(
                    response.status_code,
                    (401, 403),
                    msg=f"{method} {path} should require auth, got {response.status_code}",
                )


if __name__ == "__main__":
    unittest.main()
