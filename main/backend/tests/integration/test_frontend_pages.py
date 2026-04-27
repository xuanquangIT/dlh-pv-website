import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.dependencies import get_current_user
from app.main import create_app


def _stub_current_user(role_id: str = "admin") -> SimpleNamespace:
    return SimpleNamespace(
        username="admin",
        full_name="Admin User",
        role_id=role_id,
        role=None,
        id="00000000-0000-0000-0000-000000000001",
    )


class FrontendPagesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.pop(get_current_user, None)

    def test_home_page_requires_auth_for_html_request(self) -> None:
        response = self.client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers.get("location"), "/login")

    def test_home_page_renders_eight_module_cards_for_authenticated_user(self) -> None:
        self.app.dependency_overrides[get_current_user] = _stub_current_user

        response = self.client.get("/", headers={"accept": "text/html"})
        self.assertEqual(response.status_code, 200)

        html = response.text
        self.assertGreaterEqual(html.count('class="module-card"'), 7)

    def test_solar_ai_chat_legacy_route_redirects_to_new_path(self) -> None:
        response = self.client.get("/solar-ai-chat", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers.get("location"), "/solar-chat")

    def test_login_page_sanitizes_next_path_and_shows_invalid_credentials_error(self) -> None:
        response = self.client.get("/login?next=//evil.com&error=invalid_credentials")
        self.assertEqual(response.status_code, 200)

        html = response.text
        self.assertIn('name="next" value="/dashboard"', html)
        self.assertIn("Incorrect username or password.", html)

    def test_login_page_keeps_safe_next_path(self) -> None:
        response = self.client.get("/login?next=/forecast")
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="next" value="/forecast"', response.text)

    def test_logout_page_renders(self) -> None:
        response = self.client.get("/logout")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Sign out from PV Lakehouse", response.text)

    def test_accounts_page_maps_analyst_to_data_analyst_chat_role(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _stub_current_user("analyst")
        response = self.client.get("/settings/accounts", headers={"accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        self.assertIn('window.PV_CHAT_ROLE = "data_analyst"', response.text)


if __name__ == "__main__":
    unittest.main()
