import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Keep this test self-contained so it does not depend on local .env loading order.
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-integration")
os.environ.setdefault("AUTH_COOKIE_NAME", "pv_access_token")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

from app.api.dependencies import get_current_user
from app.main import create_app
from app.schemas.auth import Token
from app.services.auth_service import AuthService


def _stub_current_user() -> SimpleNamespace:
    return SimpleNamespace(
        username="admin",
        full_name="Admin User",
        role_id="admin",
        role=None,
    )


class AuthLoginFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = create_app()
        app.dependency_overrides[get_current_user] = _stub_current_user
        cls.client = TestClient(app)

    def test_login_sets_cookie_and_redirects_to_sanitized_next_path(self) -> None:
        with patch.object(
            AuthService,
            "authenticate_user",
            return_value=Token(access_token="stub.jwt.token"),
        ):
            response = self.client.post(
                "/auth/login",
                data={"username": "admin", "password": "admin123", "next": "//evil.com"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers.get("location"), "/dashboard")

        set_cookie_header = response.headers.get("set-cookie", "")
        self.assertIn("pv_access_token=stub.jwt.token", set_cookie_header)
        self.assertIn("HttpOnly", set_cookie_header)
        self.assertIn("Path=/", set_cookie_header)

    def test_dashboard_renders_and_common_js_does_not_force_login_redirect(self) -> None:
        dashboard_response = self.client.get("/dashboard", headers={"accept": "text/html"})
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn("common.js?v=20260406-authfix", dashboard_response.text)

        common_js_response = self.client.get("/static/js/platform_portal/common.js")
        self.assertEqual(common_js_response.status_code, 200)

        common_js = common_js_response.text
        self.assertNotIn('/login?next=', common_js)
        self.assertIn('window.location.assign("/auth/logout")', common_js)


if __name__ == "__main__":
    unittest.main()
