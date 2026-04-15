import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.dependencies import get_current_user
from app.main import create_app


class PermissionMatrixIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def _set_role(self, role_id: str) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            role_id=role_id,
            username="test.user",
            full_name="Test User",
            role=None,
        )

    def test_matrix_for_frontend_routes(self) -> None:
        route_matrix = {
            "/dashboard": {"admin", "data_engineer", "ml_engineer", "analyst"},
            "/pipeline": {"data_engineer"},
            "/quality": {"data_engineer"},
            "/training": {"ml_engineer"},
            "/registry": {"admin", "ml_engineer"},
            "/forecast": {"admin", "data_engineer", "ml_engineer", "analyst"},
            "/analytics": {"admin", "analyst"},
            "/solar-chat": {"admin", "data_engineer", "ml_engineer"},
        }

        roles = ["admin", "data_engineer", "ml_engineer", "analyst", "system"]
        for path, allowed_roles in route_matrix.items():
            for role in roles:
                with self.subTest(path=path, role=role):
                    self._set_role(role)
                    response = self.client.get(path, headers={"accept": "text/html"})
                    if role in allowed_roles:
                        self.assertEqual(response.status_code, 200)
                    else:
                        self.assertEqual(response.status_code, 403)

    def test_matrix_for_module_api_routes(self) -> None:
        api_matrix = {
            "/dashboard/summary": {"admin", "data_engineer", "ml_engineer", "analyst"},
            "/dashboard/embed-info": {"admin", "data_engineer", "ml_engineer", "analyst"},
            "/data-pipeline/status": {"data_engineer", "system"},
            "/data-quality/score": {"data_engineer"},
            "/ml-training/experiments": {"ml_engineer", "system"},
            "/model-registry/models": {"admin", "ml_engineer", "system"},
            "/forecast/next-72h": {"admin", "data_engineer", "ml_engineer", "analyst"},
            "/analytics/query-history": {"admin", "analyst"},
            "/solar-ai-chat/topics": {"admin", "data_engineer", "ml_engineer"},
        }

        roles = ["admin", "data_engineer", "ml_engineer", "analyst", "system"]
        for path, allowed_roles in api_matrix.items():
            for role in roles:
                with self.subTest(path=path, role=role):
                    self._set_role(role)
                    response = self.client.get(path)
                    if role in allowed_roles:
                        self.assertEqual(response.status_code, 200)
                    else:
                        self.assertEqual(response.status_code, 403)

    def test_data_pipeline_schedule_permissions(self) -> None:
        roles = ["admin", "data_engineer", "ml_engineer", "analyst", "system"]
        allowed_roles = {"admin", "data_engineer"}
        payload = {
            "quartz_cron_expression": "0 00 03 ? * 2",
            "timezone_id": "Asia/Ho_Chi_Minh",
            "pause_status": "UNPAUSED",
        }

        with patch("app.api.data_pipeline.routes.update_job_schedule", return_value={"ok": True}):
            for role in roles:
                with self.subTest(role=role):
                    self._set_role(role)
                    response = self.client.post("/data-pipeline/jobs/schedule", json=payload)
                    if role in allowed_roles:
                        self.assertEqual(response.status_code, 200)
                    else:
                        self.assertEqual(response.status_code, 403)

    def test_data_pipeline_schedule_payload_validation(self) -> None:
        self._set_role("data_engineer")
        invalid_payload = {
            "quartz_cron_expression": "0 00 03 ? * 2",
            "timezone_id": "Asia/Ho_Chi_Minh",
            "pause_status": "INVALID",
        }

        response = self.client.post("/data-pipeline/jobs/schedule", json=invalid_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
