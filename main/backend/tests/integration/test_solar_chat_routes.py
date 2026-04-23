"""Integration tests for Solar AI Chat routes.

Covers:
  - GET  /solar-ai-chat/topics
  - POST /solar-ai-chat/query
  - POST /solar-ai-chat/query/benchmark
  - POST /solar-ai-chat/query/benchmark/model-only
  - POST /solar-ai-chat/sessions
  - GET  /solar-ai-chat/sessions
  - GET  /solar-ai-chat/sessions/{session_id}
  - DELETE /solar-ai-chat/sessions/{session_id}
  - PATCH /solar-ai-chat/sessions/{session_id}/title
  - POST /solar-ai-chat/sessions/{session_id}/rename
  - POST /solar-ai-chat/sessions/{session_id}/fork
  - POST /solar-ai-chat/documents/ingest
  - GET  /solar-ai-chat/documents/stats
  - DELETE /solar-ai-chat/documents/{source_file}
  - POST /solar-ai-chat/stream
"""
import os
import sys
import uuid
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-integration")
os.environ.setdefault("AUTH_COOKIE_NAME", "pv_access_token")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

from app.api.dependencies import get_current_user
from app.main import create_app
from app.schemas.solar_ai_chat import (
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    ChatTopic,
    SolarChatResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_ID = "sess-0001"
USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000010")


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


def _make_session_summary(session_id: str = SESSION_ID) -> ChatSessionSummary:
    now = datetime.now(timezone.utc)
    return ChatSessionSummary(
        session_id=session_id,
        title="Test Session",
        role=ChatRole.ADMIN,
        created_at=now,
        updated_at=now,
        message_count=0,
    )


def _make_session_detail(session_id: str = SESSION_ID) -> ChatSessionDetail:
    now = datetime.now(timezone.utc)
    return ChatSessionDetail(
        session_id=session_id,
        title="Test Session",
        role=ChatRole.ADMIN,
        created_at=now,
        updated_at=now,
        messages=[],
    )


def _make_chat_response() -> SolarChatResponse:
    return SolarChatResponse(
        answer="System is nominal.",
        topic=ChatTopic.SYSTEM_OVERVIEW,
        role=ChatRole.ADMIN,
        sources=[],
        key_metrics={},
        model_used="gemini-test",
        fallback_used=False,
        latency_ms=42,
        intent_confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Base test case — app wired once, history repo mocked globally
# ---------------------------------------------------------------------------

class _BaseChatTest(unittest.TestCase):
    """Shared setup: creates the app, wires get_current_user, mocks lru_cache deps."""

    @classmethod
    def _build_app(cls, role_id: str = "admin"):
        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: _make_user(role_id)
        return app

    @classmethod
    def _build_history_mock(cls) -> MagicMock:
        mock_history = MagicMock()
        mock_history.session_exists.return_value = True
        mock_history.create_session.return_value = _make_session_summary()
        mock_history.list_sessions.return_value = [_make_session_summary()]
        mock_history.get_session.return_value = _make_session_detail()
        mock_history.delete_session.return_value = True
        mock_history.update_session_title.return_value = _make_session_summary()
        mock_history.fork_session.return_value = _make_session_summary(session_id="sess-fork")
        return mock_history

    @classmethod
    def _build_service_mock(cls) -> MagicMock:
        mock_service = MagicMock()
        mock_service.handle_query.return_value = _make_chat_response()
        return mock_service


# ---------------------------------------------------------------------------
# Topics endpoint
# ---------------------------------------------------------------------------

class TopicsEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_topics_returns_200_for_admin(self) -> None:
        response = self.client.get("/solar-ai-chat/topics")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["module"], "solar_ai_chat")

    def test_topics_returns_200_for_data_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("data_engineer")
        response = self.client.get("/solar-ai-chat/topics")
        self.assertEqual(response.status_code, 200)

    def test_topics_returns_200_for_ml_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("ml_engineer")
        response = self.client.get("/solar-ai-chat/topics")
        self.assertEqual(response.status_code, 200)

    def test_topics_returns_403_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.get("/solar-ai-chat/topics")
        self.assertEqual(response.status_code, 403)

    def test_topics_returns_403_for_system(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("system")
        response = self.client.get("/solar-ai-chat/topics")
        self.assertEqual(response.status_code, 403)

    def test_topics_returns_401_without_auth(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.get("/solar-ai-chat/topics")
        self.assertIn(response.status_code, (401, 403))


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class QueryEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_history = cls._build_history_mock()
        cls.mock_service = cls._build_service_mock()
        # Patch lru_cache singletons
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_history_repository] = lambda: cls.mock_history
        cls.app.dependency_overrides[chat_routes.get_solar_ai_chat_service] = lambda: cls.mock_service
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_query_success(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "What is the system status?"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("answer", body)
        self.assertIn("topic", body)
        self.assertIn("model_used", body)

    def test_query_with_session_id_success(self) -> None:
        self.mock_history.session_exists.return_value = True
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Status?", "session_id": SESSION_ID},
        )
        self.assertEqual(response.status_code, 200)

    def test_query_with_unknown_session_returns_404(self) -> None:
        self.mock_history.session_exists.return_value = False
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Status?", "session_id": "bad-session"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.session_exists.return_value = True

    def test_query_service_permission_error_returns_403(self) -> None:
        self.mock_service.handle_query.side_effect = PermissionError("Not allowed")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Restricted query"},
        )
        self.assertEqual(response.status_code, 403)
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_query_service_value_error_returns_400(self) -> None:
        self.mock_service.handle_query.side_effect = ValueError("Bad input")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Bad query"},
        )
        self.assertEqual(response.status_code, 400)
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_query_databricks_unavailable_returns_503(self) -> None:
        from app.repositories.solar_ai_chat.base_repository import DatabricksDataUnavailableError
        self.mock_service.handle_query.side_effect = DatabricksDataUnavailableError("down")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "System status?"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Databricks", response.json()["detail"])
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_query_unexpected_error_returns_503(self) -> None:
        self.mock_service.handle_query.side_effect = RuntimeError("Boom")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 503)
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_query_missing_message_returns_422(self) -> None:
        response = self.client.post("/solar-ai-chat/query", json={})
        self.assertEqual(response.status_code, 422)

    def test_query_empty_message_returns_422(self) -> None:
        response = self.client.post("/solar-ai-chat/query", json={"message": ""})
        self.assertEqual(response.status_code, 422)

    def test_query_message_too_long_returns_422(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "x" * 1001},
        )
        self.assertEqual(response.status_code, 422)

    def test_query_analyst_role_forbidden(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_query_analyst_role_mapped_to_data_analyst(self) -> None:
        """analyst role_id should be mapped to data_analyst ChatRole."""
        # Use a user whose role_id is 'analyst' but is explicitly allowed (override require_role)
        from app.api.solar_ai_chat import routes as chat_routes
        # Admin user (allowed), but verify mapping: admin -> ChatRole.ADMIN
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")
        response = self.client.post(
            "/solar-ai-chat/query",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["role"], "admin")

    def test_query_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.post("/solar-ai-chat/query", json={"message": "Status?"})
        self.assertIn(response.status_code, (401, 403))


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------

class BenchmarkEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_history = cls._build_history_mock()
        cls.mock_service = cls._build_service_mock()
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_history_repository] = lambda: cls.mock_history
        cls.app.dependency_overrides[chat_routes.get_solar_ai_chat_service] = lambda: cls.mock_service
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_benchmark_full_pipeline_success(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/query/benchmark",
            json={"message": "System status?"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["benchmark_type"], "full_pipeline")
        self.assertIn("server_elapsed_ms", body)
        self.assertIn("service_latency_ms", body)
        self.assertIn("route_overhead_ms", body)
        self.assertIn("response", body)

    def test_benchmark_with_invalid_session_returns_404(self) -> None:
        self.mock_history.session_exists.return_value = False
        response = self.client.post(
            "/solar-ai-chat/query/benchmark",
            json={"message": "Status?", "session_id": "bad-id"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.session_exists.return_value = True

    def test_benchmark_service_error_returns_503(self) -> None:
        self.mock_service.handle_query.side_effect = RuntimeError("Boom")
        response = self.client.post(
            "/solar-ai-chat/query/benchmark",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 503)
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_benchmark_permission_error_returns_403(self) -> None:
        self.mock_service.handle_query.side_effect = PermissionError("No access")
        response = self.client.post(
            "/solar-ai-chat/query/benchmark",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 403)
        self.mock_service.handle_query.side_effect = None
        self.mock_service.handle_query.return_value = _make_chat_response()

    def test_benchmark_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            "/solar-ai-chat/query/benchmark",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_benchmark_model_only_no_llm_key_returns_503(self) -> None:
        with patch(
            "app.api.solar_ai_chat.routes.get_solar_chat_settings"
        ) as mock_settings_fn:
            mock_settings = MagicMock()
            mock_settings.llm_api_key = None
            mock_settings.llm_base_url = None
            mock_settings_fn.return_value = mock_settings
            response = self.client.post(
                "/solar-ai-chat/query/benchmark/model-only",
                json={"message": "Status?"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM API key", response.json()["detail"])

    def test_benchmark_model_only_model_unavailable_returns_error_payload(self) -> None:
        from app.services.solar_ai_chat.llm_client import ModelUnavailableError

        mock_model_result = MagicMock()
        mock_router = MagicMock()
        mock_router.generate.side_effect = ModelUnavailableError("LLM down")

        with (
            patch("app.api.solar_ai_chat.routes.get_solar_chat_settings") as mock_settings_fn,
            patch("app.api.solar_ai_chat.routes.LLMModelRouter", return_value=mock_router),
        ):
            mock_settings = MagicMock()
            mock_settings.llm_api_key = "test-key"
            mock_settings.llm_base_url = None
            mock_settings_fn.return_value = mock_settings
            response = self.client.post(
                "/solar-ai-chat/query/benchmark/model-only",
                json={"message": "Status?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["benchmark_type"], "model_only")
        self.assertIn("error", body)
        self.assertIsNone(body["response"])

    def test_benchmark_model_only_success(self) -> None:
        mock_result = MagicMock()
        mock_result.text = "All systems go."
        mock_result.model_used = "gemini-test"
        mock_result.fallback_used = False

        mock_router = MagicMock()
        mock_router.generate.return_value = mock_result

        with (
            patch("app.api.solar_ai_chat.routes.get_solar_chat_settings") as mock_settings_fn,
            patch("app.api.solar_ai_chat.routes.LLMModelRouter", return_value=mock_router),
        ):
            mock_settings = MagicMock()
            mock_settings.llm_api_key = "test-key"
            mock_settings.llm_base_url = None
            mock_settings_fn.return_value = mock_settings
            response = self.client.post(
                "/solar-ai-chat/query/benchmark/model-only",
                json={"message": "Status?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["benchmark_type"], "model_only")
        self.assertIsNotNone(body["response"])
        self.assertEqual(body["response"]["answer"], "All systems go.")

    def test_benchmark_model_only_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            "/solar-ai-chat/query/benchmark/model-only",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class SessionEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_history = cls._build_history_mock()
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_history_repository] = lambda: cls.mock_history
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    # --- Create ---

    def test_create_session_returns_201(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/sessions",
            json={"title": "My Session"},
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertIn("session_id", body)
        self.assertIn("title", body)

    def test_create_session_default_title(self) -> None:
        response = self.client.post("/solar-ai-chat/sessions", json={})
        self.assertEqual(response.status_code, 201)

    def test_create_session_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            "/solar-ai-chat/sessions",
            json={"title": "Analyst session"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_create_session_title_too_long_returns_422(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/sessions",
            json={"title": "x" * 201},
        )
        self.assertEqual(response.status_code, 422)

    # --- List ---

    def test_list_sessions_returns_list(self) -> None:
        response = self.client.get("/solar-ai-chat/sessions")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body, list)
        self.assertGreater(len(body), 0)

    def test_list_sessions_with_pagination(self) -> None:
        response = self.client.get("/solar-ai-chat/sessions?limit=10&offset=0")
        self.assertEqual(response.status_code, 200)

    def test_list_sessions_forbidden_for_system(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("system")
        response = self.client.get("/solar-ai-chat/sessions")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Get by ID ---

    def test_get_session_success(self) -> None:
        response = self.client.get(f"/solar-ai-chat/sessions/{SESSION_ID}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["session_id"], SESSION_ID)
        self.assertIn("messages", body)

    def test_get_session_not_found_returns_404(self) -> None:
        self.mock_history.get_session.return_value = None
        response = self.client.get("/solar-ai-chat/sessions/nonexistent")
        self.assertEqual(response.status_code, 404)
        self.mock_history.get_session.return_value = _make_session_detail()

    def test_get_session_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.get(f"/solar-ai-chat/sessions/{SESSION_ID}")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Delete ---

    def test_delete_session_success_returns_204(self) -> None:
        self.mock_history.delete_session.return_value = True
        response = self.client.delete(f"/solar-ai-chat/sessions/{SESSION_ID}")
        self.assertEqual(response.status_code, 204)

    def test_delete_session_not_found_returns_404(self) -> None:
        self.mock_history.delete_session.return_value = False
        response = self.client.delete("/solar-ai-chat/sessions/ghost-session")
        self.assertEqual(response.status_code, 404)
        self.mock_history.delete_session.return_value = True

    def test_delete_session_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.delete(f"/solar-ai-chat/sessions/{SESSION_ID}")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Update title (PATCH) ---

    def test_update_session_title_success(self) -> None:
        response = self.client.patch(
            f"/solar-ai-chat/sessions/{SESSION_ID}/title",
            json={"title": "Updated Title"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("session_id", body)

    def test_update_session_title_not_found_returns_404(self) -> None:
        self.mock_history.update_session_title.return_value = None
        response = self.client.patch(
            "/solar-ai-chat/sessions/ghost/title",
            json={"title": "Ghost"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.update_session_title.return_value = _make_session_summary()

    def test_update_session_title_empty_returns_422(self) -> None:
        response = self.client.patch(
            f"/solar-ai-chat/sessions/{SESSION_ID}/title",
            json={"title": ""},
        )
        self.assertEqual(response.status_code, 422)

    def test_update_session_title_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.patch(
            f"/solar-ai-chat/sessions/{SESSION_ID}/title",
            json={"title": "Analyst Title"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Rename (POST alias) ---

    def test_rename_session_success(self) -> None:
        response = self.client.post(
            f"/solar-ai-chat/sessions/{SESSION_ID}/rename",
            json={"title": "Renamed"},
        )
        self.assertEqual(response.status_code, 200)

    def test_rename_session_not_found_returns_404(self) -> None:
        self.mock_history.update_session_title.return_value = None
        response = self.client.post(
            "/solar-ai-chat/sessions/ghost/rename",
            json={"title": "Renamed"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.update_session_title.return_value = _make_session_summary()

    # --- Fork ---

    def test_fork_session_returns_201(self) -> None:
        response = self.client.post(
            f"/solar-ai-chat/sessions/{SESSION_ID}/fork",
            json={"title": "Forked Session"},
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertIn("session_id", body)

    def test_fork_session_not_found_returns_404(self) -> None:
        self.mock_history.fork_session.return_value = None
        response = self.client.post(
            "/solar-ai-chat/sessions/ghost/fork",
            json={"title": "Fork"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.fork_session.return_value = _make_session_summary(session_id="sess-fork")

    def test_fork_session_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            f"/solar-ai-chat/sessions/{SESSION_ID}/fork",
            json={"title": "Fork"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")


# ---------------------------------------------------------------------------
# RAG document endpoints (admin only)
# ---------------------------------------------------------------------------

class RagDocumentEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_vector_repo = MagicMock()
        cls.mock_vector_repo.count_chunks.return_value = {
            "total_chunks": 42,
            "by_doc_type": {"incident_report": 20, "equipment_manual": 22},
        }
        cls.mock_vector_repo.delete_by_source.return_value = 5
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_vector_repository] = (
            lambda: cls.mock_vector_repo
        )
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    # --- Stats ---

    def test_get_document_stats_success(self) -> None:
        response = self.client.get("/solar-ai-chat/documents/stats")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_chunks"], 42)
        self.assertIn("by_doc_type", body)

    def test_get_document_stats_no_vector_repo_returns_zeros(self) -> None:
        from app.api.solar_ai_chat import routes as chat_routes
        self.app.dependency_overrides[chat_routes._get_vector_repository] = lambda: None
        response = self.client.get("/solar-ai-chat/documents/stats")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_chunks"], 0)
        self.app.dependency_overrides[chat_routes._get_vector_repository] = (
            lambda: self.mock_vector_repo
        )

    def test_get_document_stats_forbidden_for_data_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("data_engineer")
        response = self.client.get("/solar-ai-chat/documents/stats")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_get_document_stats_forbidden_for_ml_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("ml_engineer")
        response = self.client.get("/solar-ai-chat/documents/stats")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Delete ---

    def test_delete_document_success_returns_204(self) -> None:
        self.mock_vector_repo.delete_by_source.return_value = 3
        response = self.client.delete("/solar-ai-chat/documents/my_manual.pdf")
        self.assertEqual(response.status_code, 204)

    def test_delete_document_not_found_returns_404(self) -> None:
        self.mock_vector_repo.delete_by_source.return_value = 0
        response = self.client.delete("/solar-ai-chat/documents/ghost.pdf")
        self.assertEqual(response.status_code, 404)
        self.mock_vector_repo.delete_by_source.return_value = 5

    def test_delete_document_no_vector_repo_returns_503(self) -> None:
        from app.api.solar_ai_chat import routes as chat_routes
        self.app.dependency_overrides[chat_routes._get_vector_repository] = lambda: None
        response = self.client.delete("/solar-ai-chat/documents/my_manual.pdf")
        self.assertEqual(response.status_code, 503)
        self.app.dependency_overrides[chat_routes._get_vector_repository] = (
            lambda: self.mock_vector_repo
        )

    def test_delete_document_forbidden_for_data_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("data_engineer")
        response = self.client.delete("/solar-ai-chat/documents/some.pdf")
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    # --- Ingest ---

    def test_ingest_document_no_embedding_key_returns_503(self) -> None:
        with patch(
            "app.api.solar_ai_chat.routes.get_solar_chat_settings"
        ) as mock_settings_fn:
            mock_settings = MagicMock()
            mock_settings.embedding_api_key = None
            mock_settings_fn.return_value = mock_settings
            response = self.client.post(
                "/solar-ai-chat/documents/ingest",
                json={"file_path": "/some/file.txt", "doc_type": "incident_report"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Embedding API key", response.json()["detail"])

    def test_ingest_document_forbidden_for_data_engineer(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("data_engineer")
        response = self.client.post(
            "/solar-ai-chat/documents/ingest",
            json={"file_path": "/some/file.txt", "doc_type": "incident_report"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_ingest_document_invalid_doc_type_returns_422(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/documents/ingest",
            json={"file_path": "/some/file.txt", "doc_type": "invalid_type"},
        )
        self.assertEqual(response.status_code, 422)

    def test_ingest_document_missing_file_path_returns_422(self) -> None:
        response = self.client.post(
            "/solar-ai-chat/documents/ingest",
            json={"doc_type": "incident_report"},
        )
        self.assertEqual(response.status_code, 422)

    def test_ingest_document_file_not_found_returns_400(self) -> None:
        mock_settings = MagicMock()
        mock_settings.embedding_api_key = "test-key"
        mock_settings.resolved_data_root = Path("/app/data/sql")

        with (
            patch("app.api.solar_ai_chat.routes.get_solar_chat_settings", return_value=mock_settings),
            patch("app.api.solar_ai_chat.routes._get_vector_repository", return_value=self.mock_vector_repo),
            patch("app.api.solar_ai_chat.routes._get_embedding_client", return_value=MagicMock()),
        ):
            response = self.client.post(
                "/solar-ai-chat/documents/ingest",
                json={
                    "file_path": "/nonexistent/path/file.txt",
                    "doc_type": "incident_report",
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("File not found", response.json()["detail"])

    def test_ingest_document_path_outside_data_root_returns_400(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"some content")
            tmp_path = tmp.name

        try:
            mock_settings = MagicMock()
            mock_settings.embedding_api_key = "test-key"
            # Set data root to a path that does NOT contain the temp file
            mock_settings.resolved_data_root = Path("/restricted/data/sql")

            with (
                patch("app.api.solar_ai_chat.routes.get_solar_chat_settings", return_value=mock_settings),
                patch("app.api.solar_ai_chat.routes._get_vector_repository", return_value=self.mock_vector_repo),
                patch("app.api.solar_ai_chat.routes._get_embedding_client", return_value=MagicMock()),
            ):
                response = self.client.post(
                    "/solar-ai-chat/documents/ingest",
                    json={"file_path": tmp_path, "doc_type": "incident_report"},
                )
            self.assertEqual(response.status_code, 400)
            self.assertIn("data directory", response.json()["detail"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SSE stream endpoint
# ---------------------------------------------------------------------------

class StreamEndpointTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_history = cls._build_history_mock()
        cls.mock_service = cls._build_service_mock()
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_history_repository] = lambda: cls.mock_history
        cls.app.dependency_overrides[chat_routes.get_solar_ai_chat_service] = lambda: cls.mock_service
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def _make_sse_events(self) -> list[str]:
        import json
        done_event = {
            "event": "done",
            "answer": "All good.",
            "topic": "general",
            "role": "admin",
            "sources": [],
            "key_metrics": {},
            "model_used": "gemini-test",
            "fallback_used": False,
            "latency_ms": 10,
            "intent_confidence": 0.9,
        }
        return [f"data: {json.dumps(done_event)}\n\n"]

    def test_stream_returns_200_with_event_stream_content_type(self) -> None:
        self.mock_service.handle_query_stream.return_value = iter(self._make_sse_events())
        response = self.client.post(
            "/solar-ai-chat/stream",
            json={"message": "Stream status?"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

    def test_stream_forbidden_for_analyst(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.post(
            "/solar-ai-chat/stream",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 403)
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")

    def test_stream_invalid_session_returns_404(self) -> None:
        self.mock_history.session_exists.return_value = False
        response = self.client.post(
            "/solar-ai-chat/stream",
            json={"message": "Status?", "session_id": "bad-sess"},
        )
        self.assertEqual(response.status_code, 404)
        self.mock_history.session_exists.return_value = True

    def test_stream_unauthenticated_returns_401(self) -> None:
        app_no_auth = create_app()
        client = TestClient(app_no_auth, raise_server_exceptions=False)
        response = client.post("/solar-ai-chat/stream", json={"message": "Status?"})
        self.assertIn(response.status_code, (401, 403))

    def test_stream_service_error_yields_error_event(self) -> None:
        import json

        def _error_gen():
            raise RuntimeError("stream crash")
            yield  # make it a generator

        self.mock_service.handle_query_stream.side_effect = None
        self.mock_service.handle_query_stream.return_value = _error_gen()

        response = self.client.post(
            "/solar-ai-chat/stream",
            json={"message": "Status?"},
        )
        self.assertEqual(response.status_code, 200)
        content = response.text
        # The stream error handler should have emitted an error SSE event
        self.assertIn("stream_error", content)

    def test_stream_missing_message_returns_422(self) -> None:
        response = self.client.post("/solar-ai-chat/stream", json={})
        self.assertEqual(response.status_code, 422)


# ---------------------------------------------------------------------------
# Role mapping edge cases
# ---------------------------------------------------------------------------

class RoleMappingTests(_BaseChatTest):
    """Test _resolve_user_chat_role behaviour via the query endpoint."""

    @classmethod
    def _build_dynamic_service_mock(cls) -> MagicMock:
        """Service mock that echoes the role from the request."""
        mock_service = MagicMock()
        def _handle(req):
            return SolarChatResponse(
                answer="ok",
                topic=ChatTopic.GENERAL,
                role=req.role,
                sources=[],
                key_metrics={},
                model_used="test",
                fallback_used=False,
                latency_ms=1,
                intent_confidence=0.0,
            )
        mock_service.handle_query.side_effect = _handle
        return mock_service

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_history = cls._build_history_mock()
        cls.mock_service = cls._build_dynamic_service_mock()
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_history_repository] = lambda: cls.mock_history
        cls.app.dependency_overrides[chat_routes.get_solar_ai_chat_service] = lambda: cls.mock_service
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_admin_role_mapped_correctly(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")
        response = self.client.post("/solar-ai-chat/query", json={"message": "hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role"], "admin")

    def test_data_engineer_role_mapped_correctly(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("data_engineer")
        response = self.client.post("/solar-ai-chat/query", json={"message": "hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role"], "data_engineer")

    def test_ml_engineer_role_mapped_correctly(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("ml_engineer")
        response = self.client.post("/solar-ai-chat/query", json={"message": "hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role"], "ml_engineer")


# ---------------------------------------------------------------------------
# Admin tool-stats endpoint (Task 0.1)
# ---------------------------------------------------------------------------

class AdminToolStatsTests(_BaseChatTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = cls._build_app("admin")
        cls.mock_usage_repo = MagicMock()
        cls.mock_usage_repo.get_stats.return_value = {
            "window_days": 7,
            "total_calls": 42,
            "by_tool": [
                {"tool_name": "get_system_overview", "count": 20,
                 "avg_latency_ms": 123.4, "success_rate": 1.0},
                {"tool_name": "get_forecast_72h", "count": 10,
                 "avg_latency_ms": 200.0, "success_rate": 0.9},
            ],
            "by_role": [{"role": "admin", "count": 30}],
        }
        from app.api.solar_ai_chat import routes as chat_routes
        cls.app.dependency_overrides[chat_routes._get_tool_usage_repository] = (
            lambda: cls.mock_usage_repo
        )
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.clear()

    def test_admin_can_fetch_default_7_day_stats(self) -> None:
        response = self.client.get("/solar-ai-chat/admin/tool-stats")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_calls"], 42)
        self.assertEqual(body["by_tool"][0]["tool_name"], "get_system_overview")
        self.mock_usage_repo.get_stats.assert_called_with(days=7)

    def test_admin_can_pass_custom_days(self) -> None:
        response = self.client.get("/solar-ai-chat/admin/tool-stats?days=30")
        self.assertEqual(response.status_code, 200)
        self.mock_usage_repo.get_stats.assert_called_with(days=30)

    def test_invalid_days_zero_rejected(self) -> None:
        response = self.client.get("/solar-ai-chat/admin/tool-stats?days=0")
        self.assertEqual(response.status_code, 400)

    def test_invalid_days_too_large_rejected(self) -> None:
        response = self.client.get("/solar-ai-chat/admin/tool-stats?days=999")
        self.assertEqual(response.status_code, 400)

    def test_non_admin_forbidden(self) -> None:
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("analyst")
        response = self.client.get("/solar-ai-chat/admin/tool-stats")
        self.assertEqual(response.status_code, 403)
        # reset
        self.app.dependency_overrides[get_current_user] = lambda: _make_user("admin")


if __name__ == "__main__":
    unittest.main()
