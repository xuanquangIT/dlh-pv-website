import os
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-integration")
os.environ.setdefault("AUTH_COOKIE_NAME", "pv_access_token")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

from app.api.dependencies import get_current_user
from app.main import create_app


def _stub_admin_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        username="admin",
        email="admin@example.com",
        full_name="Admin User",
        role_id="admin",
        role=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


class AuthAdminRoutesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.dependency_overrides[get_current_user] = _stub_admin_user
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.pop(get_current_user, None)

    def test_get_me_returns_current_user(self) -> None:
        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["username"], "admin")
        self.assertEqual(body["role_id"], "admin")

    def test_logout_get_and_post_redirect_to_login(self) -> None:
        get_response = self.client.get("/auth/logout", follow_redirects=False)
        self.assertEqual(get_response.status_code, 303)
        self.assertEqual(get_response.headers.get("location"), "/login")

        post_response = self.client.post("/auth/logout", follow_redirects=False)
        self.assertEqual(post_response.status_code, 303)
        self.assertEqual(post_response.headers.get("location"), "/login")

    def test_list_users_uses_service(self) -> None:
        user_payload = {
            "id": str(uuid.uuid4()),
            "username": "viewer",
            "email": "viewer@example.com",
            "full_name": "Viewer User",
            "role_id": "analyst",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch("app.api.auth.routes.AuthService.list_users", return_value=[user_payload]):
            response = self.client.get("/auth/users")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["username"], "viewer")

    def test_create_user_by_admin_uses_service(self) -> None:
        created_payload = {
            "id": str(uuid.uuid4()),
            "username": "dataeng",
            "email": "dataeng@example.com",
            "full_name": "Data Engineer",
            "role_id": "data_engineer",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch("app.api.auth.routes.AuthService.create_user_by_admin", return_value=created_payload):
            response = self.client.post(
                "/auth/users",
                json={
                    "username": "dataeng",
                    "email": "dataeng@example.com",
                    "full_name": "Data Engineer",
                    "role_id": "data_engineer",
                    "password": "secret123",
                    "is_active": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "dataeng")

    def test_register_user_uses_service(self) -> None:
        created_payload = {
            "id": str(uuid.uuid4()),
            "username": "mleng",
            "email": "mleng@example.com",
            "full_name": "ML Engineer",
            "role_id": "ml_engineer",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch("app.api.auth.routes.AuthService.create_user", return_value=created_payload):
            response = self.client.post(
                "/auth/register",
                json={
                    "username": "mleng",
                    "email": "mleng@example.com",
                    "full_name": "ML Engineer",
                    "role_id": "ml_engineer",
                    "password": "secret123",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role_id"], "ml_engineer")

    def test_update_user_status_uses_service(self) -> None:
        target_id = uuid.uuid4()
        updated_payload = {
            "id": str(target_id),
            "username": "analyst",
            "email": "analyst@example.com",
            "full_name": "Analyst User",
            "role_id": "analyst",
            "is_active": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with patch("app.api.auth.routes.AuthService.update_user_status", return_value=updated_payload):
            response = self.client.patch(f"/auth/users/{target_id}/status", json={"is_active": False})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_active"])

    def test_reset_user_password_returns_204(self) -> None:
        target_id = uuid.uuid4()
        with patch("app.api.auth.routes.AuthService.reset_user_password", return_value=None):
            response = self.client.patch(
                f"/auth/users/{target_id}/password",
                json={"new_password": "new-secure-password"},
            )

        self.assertEqual(response.status_code, 204)


if __name__ == "__main__":
    unittest.main()
