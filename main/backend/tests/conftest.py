"""Shared pytest fixtures and stubs for the PV Lakehouse website test suite."""
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Bcrypt stub — prevents real bcrypt from being required in CI
# ---------------------------------------------------------------------------

def _install_bcrypt_stub() -> None:
    if "bcrypt" in sys.modules:
        return

    bcrypt_stub = types.ModuleType("bcrypt")
    bcrypt_stub.__version__ = "4.1.3"
    bcrypt_stub.__about__ = types.SimpleNamespace(__version__="4.1.3")
    bcrypt_stub.gensalt = lambda rounds=12: b"stub-salt"
    bcrypt_stub.hashpw = lambda password, salt: password + b"." + salt
    bcrypt_stub.checkpw = lambda password, hashed: True
    bcrypt_stub.kdf = lambda password, salt, desired_key_bytes, rounds, ignore_few_rounds=False: b"k" * desired_key_bytes
    sys.modules["bcrypt"] = bcrypt_stub


try:
    import bcrypt
except Exception:
    _install_bcrypt_stub()


# ---------------------------------------------------------------------------
# Environment overrides — prevent .env file from being required
# ---------------------------------------------------------------------------

TEST_SECRET_KEY = "test-secret-key-for-pytest-only-not-production"
TEST_DB_URL = "postgresql://test:test@localhost/test_db"

_ENV_OVERRIDES = {
    "AUTH_SECRET_KEY": TEST_SECRET_KEY,
    "DATABASE_URL": TEST_DB_URL,
    "DATABRICKS_HOST": "https://test.databricks.com",
    "DATABRICKS_TOKEN": "test-token",
    "DATABRICKS_SQL_HTTP_PATH": "/sql/1.0/warehouses/test",
    "UC_CATALOG": "pv",
    "UC_SILVER_SCHEMA": "silver",
    "UC_GOLD_SCHEMA": "gold",
    "UC_APP_CATALOG": "pv",
    "UC_APP_SCHEMA": "app",
    "SOLAR_CHAT_LLM_API_FORMAT": "gemini",
    "SOLAR_CHAT_LLM_API_KEY": "test-llm-key",
    "SOLAR_CHAT_PRIMARY_MODEL": "gemini-2.0-flash",
    "SOLAR_CHAT_FALLBACK_MODEL": "gemini-1.5-flash",
    "SOLAR_CHAT_EMBEDDING_MODEL": "text-embedding-004",
    "SOLAR_CHAT_HISTORY_BACKEND": "postgres",
}


@pytest.fixture(autouse=True, scope="session")
def override_settings_env():
    """Override environment variables so settings load without a real .env file."""
    with patch.dict("os.environ", _ENV_OVERRIDES, clear=False):
        yield


# ---------------------------------------------------------------------------
# AuthUser factory
# ---------------------------------------------------------------------------

def make_auth_user(
    role_id: str = "analyst",
    user_id: str | None = None,
    username: str = "testuser",
    email: str = "test@example.com",
    is_active: bool = True,
) -> MagicMock:
    """Return a MagicMock that behaves like an AuthUser ORM object."""
    user = MagicMock()
    user.id = uuid.UUID(user_id) if user_id else uuid.uuid4()
    user.username = username
    user.email = email
    user.full_name = "Test User"
    user.is_active = is_active
    user.role_id = role_id
    user.created_at = datetime.now(timezone.utc)
    user.role = MagicMock()
    user.role.name = role_id
    return user


@pytest.fixture
def analyst_user():
    return make_auth_user(role_id="analyst")


@pytest.fixture
def admin_user():
    return make_auth_user(role_id="admin", username="adminuser", email="admin@example.com")


@pytest.fixture
def ml_engineer_user():
    return make_auth_user(role_id="ml_engineer", username="mluser", email="ml@example.com")


@pytest.fixture
def data_engineer_user():
    return make_auth_user(role_id="data_engineer", username="deuser", email="de@example.com")


# ---------------------------------------------------------------------------
# FastAPI TestClient with dependency overrides
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Create the FastAPI app for testing, with DB/static files stubbed out."""
    with patch("fastapi.staticfiles.StaticFiles.__init__", return_value=None), \
         patch("fastapi.staticfiles.StaticFiles.__call__"), \
         patch("sqlalchemy.create_engine", return_value=MagicMock()), \
         patch("sqlalchemy.orm.sessionmaker", return_value=MagicMock()):
        from app.main import create_app as _create_app
        return _create_app()


@pytest.fixture
def client(app, analyst_user):
    """TestClient with analyst user injected as current user."""
    from app.api.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: analyst_user
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def admin_client(app, admin_user):
    """TestClient with admin user injected."""
    from app.api.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: admin_user
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client(app):
    """TestClient with no user override — tests 401 enforcement."""
    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Mock Databricks cursor / connection
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dbx_cursor():
    """A MagicMock that mimics a Databricks SQL cursor."""
    cursor = MagicMock()
    cursor.description = []
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    return cursor


@pytest.fixture
def mock_dbx_connection(mock_dbx_cursor):
    """A MagicMock connection that returns mock_dbx_cursor."""
    conn = MagicMock()
    conn.cursor.return_value = mock_dbx_cursor
    return conn


# ---------------------------------------------------------------------------
# Mock SQLAlchemy session
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_session():
    """A MagicMock SQLAlchemy session."""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.filter_by.return_value = session
    session.first.return_value = None
    session.all.return_value = []
    session.add.return_value = None
    session.commit.return_value = None
    session.delete.return_value = None
    session.refresh.return_value = None
    return session


# ---------------------------------------------------------------------------
# Reusable sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_energy_metrics():
    return {
        "all_facilities": [
            {"facility": "WRSF1", "energy_mwh": 150.5, "capacity_factor_pct": 22.3},
            {"facility": "AVLSF", "energy_mwh": 120.0, "capacity_factor_pct": 18.1},
            {"facility": "BOMENSF", "energy_mwh": 90.2, "capacity_factor_pct": 15.7},
        ],
        "top_facilities": [{"facility": "WRSF1", "energy_mwh": 150.5}],
        "bottom_facilities": [{"facility": "BOMENSF", "energy_mwh": 90.2}],
        "facility_count": 8,
        "window_days": 7,
    }


@pytest.fixture
def sample_daily_metrics():
    return [
        {"date": "2026-04-15", "facility": "WRSF1", "energy_mwh": 145.2, "capacity_factor_pct": 21.5},
        {"date": "2026-04-16", "facility": "WRSF1", "energy_mwh": 132.8, "capacity_factor_pct": 19.7},
        {"date": "2026-04-17", "facility": "WRSF1", "energy_mwh": 158.1, "capacity_factor_pct": 23.4},
        {"date": "2026-04-18", "facility": "WRSF1", "energy_mwh": 140.0, "capacity_factor_pct": 20.8},
        {"date": "2026-04-19", "facility": "WRSF1", "energy_mwh": 155.5, "capacity_factor_pct": 23.1},
    ]


@pytest.fixture
def sample_forecast_metrics():
    return {
        "daily_forecast": [
            {"date": "2026-04-22", "expected_mwh": 1200.0, "confidence_interval": {"low": 1100.0, "high": 1300.0}},
            {"date": "2026-04-23", "expected_mwh": 1150.0, "confidence_interval": {"low": 1050.0, "high": 1250.0}},
            {"date": "2026-04-24", "expected_mwh": 1250.0, "confidence_interval": {"low": 1150.0, "high": 1350.0}},
        ]
    }
