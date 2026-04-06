import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app
from app.api.dependencies import get_current_user
from app.api.dashboard.routes import get_powerbi_service
from app.schemas.dashboard import EmbedTokenResponse


class _StubPowerBIService:
    def get_embed_info(self) -> EmbedTokenResponse:
        return EmbedTokenResponse(
            embed_token="test_embed_token",
            embed_url="https://app.powerbi.com/reportEmbed?reportId=test_report",
            report_id="test_report",
        )


def _get_stub_powerbi_service() -> _StubPowerBIService:
    return _StubPowerBIService()


class DashboardApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.dependency_overrides[get_powerbi_service] = _get_stub_powerbi_service
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def test_embed_info_requires_authenticated_user(self) -> None:
        response = self.client.get("/dashboard/embed-info")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_embed_info_rejects_unauthorized_role(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(role_id="system")
        response = self.client.get("/dashboard/embed-info")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"],
            "Role is not allowed to access dashboard embed information.",
        )

    def test_embed_info_allows_authorized_role(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(role_id="data_engineer")
        response = self.client.get("/dashboard/embed-info")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["embed_token"], "test_embed_token")
        self.assertEqual(body["report_id"], "test_report")


if __name__ == "__main__":
    unittest.main()
