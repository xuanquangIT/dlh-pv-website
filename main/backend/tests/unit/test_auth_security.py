"""Comprehensive unit tests for auth security helpers and AuthService.

Covers:
  - security.py: verify_password, get_password_hash, create_access_token,
    decode_access_token
  - services/auth/service.py: AuthService.authenticate_user, create_user,
    create_user_by_admin, list_users, update_user_status, reset_user_password,
    _raise_duplicate_user_error
  - repositories/auth/user_repository.py: DuplicateUserError, UserRepository
    helper methods (_as_auth_user, get_by_username, get_by_email, get_by_id,
    create, list_users, update_active_status, update_password)

Strategy:
  - JWT tests use real jose.jwt logic with a deterministic test secret.
  - Password tests use the bcrypt stub from conftest (checkpw always True).
  - DB calls are mocked via patch("...SessionLocal").
  - All FastAPI HTTPException cases are asserted for correct status codes.
"""
from __future__ import annotations

import os
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path & env setup — must happen before any project imports
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Provide a deterministic secret so JWT tests are self-contained.
os.environ.setdefault("AUTH_SECRET_KEY", "test-super-secret-key-for-unit-tests-only")
os.environ.setdefault("AUTH_COOKIE_NAME", "pv_access_token")
os.environ.setdefault("AUTH_COOKIE_SECURE", "false")

# bcrypt stub (mirrors conftest) — must be installed before security import
if "bcrypt" not in sys.modules:
    bcrypt_stub = types.ModuleType("bcrypt")
    bcrypt_stub.gensalt = lambda rounds=12: b"stub-salt"
    bcrypt_stub.hashpw = lambda password, salt: password + b"." + salt
    # Always return True; we test wrong-password via AuthService mock, not bcrypt itself.
    bcrypt_stub.checkpw = lambda password, hashed: True
    sys.modules["bcrypt"] = bcrypt_stub

# Stub databricks so repository imports work (only if real package not installed)
try:
    import databricks  # noqa: F401
except ImportError:
    _db_pkg = types.ModuleType("databricks")
    _db_pkg.__path__ = []  # mark as a package so submodules can be imported
    _db_sql = types.ModuleType("databricks.sql")
    _db_sql.connect = MagicMock()
    sys.modules.setdefault("databricks", _db_pkg)
    sys.modules.setdefault("databricks.sql", _db_sql)

from fastapi import HTTPException  # noqa: E402

from app.core.security import (  # noqa: E402
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from app.core.settings import AuthSettings  # noqa: E402
from app.repositories.auth.user_repository import (  # noqa: E402
    DuplicateUserError,
    UserRepository,
)
from app.schemas.auth import AdminUserCreate, LoginRequest, UserCreate  # noqa: E402
from app.services.auth.service import AuthService  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_TEST_SECRET = "test-super-secret-key-for-unit-tests-only"
_TEST_ALGORITHM = "HS256"
_USER_UUID = uuid.uuid4()
_USER_UUID_STR = str(_USER_UUID)


def _make_auth_settings(**overrides) -> AuthSettings:
    kwargs = dict(
        secret_key=_TEST_SECRET,
        algorithm=_TEST_ALGORITHM,
        access_token_expire_minutes=30,
        cookie_name="pv_access_token",
        cookie_secure=False,
    )
    kwargs.update(overrides)
    return AuthSettings.model_construct(**kwargs)


def _fake_user(**kwargs) -> SimpleNamespace:
    defaults = dict(
        id=_USER_UUID,
        username="testuser",
        email="test@example.com",
        hashed_password="hashed.stub-salt",
        full_name="Test User",
        is_active=True,
        role_id="analyst",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        role=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _cm(obj):
    """Wrap obj in a context-manager mock."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=obj)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ===========================================================================
# 1. verify_password
# ===========================================================================

class TestVerifyPassword:
    """bcrypt stub's checkpw always returns True, so we focus on the ValueError
    guard and the basic call path."""

    def test_returns_true_for_correct_password(self):
        # Hash a password then verify it matches
        hashed = get_password_hash("correct-password")
        assert verify_password("correct-password", hashed) is True

    def test_returns_false_on_bcrypt_value_error(self):
        """If bcrypt raises ValueError (e.g. invalid hash), returns False."""
        import bcrypt as _bcrypt
        with patch.object(_bcrypt, "checkpw", side_effect=ValueError("invalid hash")):
            result = verify_password("plain", "badhash")
        assert result is False

    def test_encodes_password_before_passing_to_bcrypt(self):
        """Ensures the bytes-level encoding step is executed."""
        import bcrypt as _bcrypt
        calls = []
        original = _bcrypt.checkpw

        def _recording_checkpw(pw, hashed):
            calls.append(pw)
            return True

        with patch.object(_bcrypt, "checkpw", side_effect=_recording_checkpw):
            verify_password("mypassword", "somehash")

        assert calls[0] == b"mypassword"


# ===========================================================================
# 2. get_password_hash
# ===========================================================================

class TestGetPasswordHash:
    def test_returns_string(self):
        h = get_password_hash("secret")
        assert isinstance(h, str)

    def test_output_is_bcrypt_format(self):
        h = get_password_hash("secret")
        # Real bcrypt always starts with $2b$ and is 60 chars
        assert h.startswith("$2") and len(h) == 60

    def test_different_passwords_produce_different_hashes(self):
        h1 = get_password_hash("password1")
        h2 = get_password_hash("password2")
        assert h1 != h2


# ===========================================================================
# 3. create_access_token / decode_access_token (real jose JWT)
# ===========================================================================

class TestJwtTokenCreation:
    def _patch_settings(self, **kwargs):
        settings = _make_auth_settings(**kwargs)
        return patch("app.core.security.get_auth_settings", return_value=settings)

    def test_creates_decodable_token(self):
        with self._patch_settings():
            token = create_access_token("user123")
        with self._patch_settings():
            payload = decode_access_token(token)
        assert payload["sub"] == "user123"

    def test_uuid_subject_round_trips(self):
        uid = str(uuid.uuid4())
        with self._patch_settings():
            token = create_access_token(uid)
        with self._patch_settings():
            payload = decode_access_token(token)
        assert payload["sub"] == uid

    def test_custom_expires_delta(self):
        with self._patch_settings():
            token = create_access_token("user", expires_delta=timedelta(minutes=5))
        with self._patch_settings():
            payload = decode_access_token(token)
        assert "exp" in payload

    def test_expired_token_raises(self):
        from jose import ExpiredSignatureError
        with self._patch_settings():
            token = create_access_token("user", expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception):  # jose raises JWTError / ExpiredSignatureError
            with self._patch_settings():
                decode_access_token(token)

    def test_tampered_token_raises(self):
        with self._patch_settings():
            token = create_access_token("user")
        tampered = token[:-4] + "XXXX"
        with pytest.raises(Exception):
            with self._patch_settings():
                decode_access_token(tampered)

    def test_wrong_secret_raises(self):
        with self._patch_settings(secret_key="original-secret"):
            token = create_access_token("user")
        with pytest.raises(Exception):
            with self._patch_settings(secret_key="wrong-secret"):
                decode_access_token(token)

    def test_default_expiry_uses_settings(self):
        """Without explicit delta, expiry is derived from access_token_expire_minutes."""
        with self._patch_settings(access_token_expire_minutes=60):
            token = create_access_token("user")
        with self._patch_settings(access_token_expire_minutes=60):
            payload = decode_access_token(token)
        now = datetime.now(timezone.utc)
        # exp should be ~60 minutes from now — allow ±5s skew
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        delta = (exp - now).total_seconds()
        assert 3550 <= delta <= 3650

    def test_empty_subject_still_encodes(self):
        with self._patch_settings():
            token = create_access_token("")
        with self._patch_settings():
            payload = decode_access_token(token)
        assert payload["sub"] == ""

    def test_completely_invalid_token_raises(self):
        with pytest.raises(Exception):
            with patch("app.core.security.get_auth_settings", return_value=_make_auth_settings()):
                decode_access_token("this.is.not.a.jwt")


# ===========================================================================
# 4. DuplicateUserError
# ===========================================================================

class TestDuplicateUserError:
    def test_field_attribute_set(self):
        err = DuplicateUserError("username")
        assert err.field == "username"

    def test_message_includes_field(self):
        err = DuplicateUserError("email")
        assert "email" in str(err)

    def test_is_exception_subclass(self):
        assert issubclass(DuplicateUserError, Exception)


# ===========================================================================
# 5. UserRepository._as_auth_user
# ===========================================================================

class TestAsAuthUser:
    def test_converts_dict_to_namespace(self):
        row = {
            "id": str(_USER_UUID),
            "username": "alice",
            "email": "alice@example.com",
            "hashed_password": "hash",
            "full_name": "Alice",
            "is_active": True,
            "role_id": "admin",
            "created_at": datetime(2026, 1, 1),
        }
        user = UserRepository._as_auth_user(row)
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert isinstance(user.id, uuid.UUID)

    def test_is_active_coerced_to_bool(self):
        row = {
            "id": str(_USER_UUID),
            "username": "bob",
            "email": "b@b.com",
            "hashed_password": "x",
            "is_active": 1,  # integer
            "role_id": "analyst",
        }
        user = UserRepository._as_auth_user(row)
        assert user.is_active is True

    def test_missing_full_name_is_none(self):
        row = {
            "id": str(_USER_UUID),
            "username": "bob",
            "email": "b@b.com",
            "hashed_password": "x",
            "is_active": True,
            "role_id": "analyst",
        }
        user = UserRepository._as_auth_user(row)
        assert user.full_name is None

    def test_missing_created_at_gets_default(self):
        row = {
            "id": str(_USER_UUID),
            "username": "bob",
            "email": "b@b.com",
            "hashed_password": "x",
            "is_active": True,
            "role_id": "analyst",
        }
        user = UserRepository._as_auth_user(row)
        assert isinstance(user.created_at, datetime)


# ===========================================================================
# 6. UserRepository.get_by_username
# ===========================================================================

class TestGetByUsername:
    def _db_mock_with_user(self, orm_user):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = orm_user
        return db

    def _make_orm_user(self):
        u = MagicMock()
        u.id = _USER_UUID
        u.username = "alice"
        u.email = "alice@example.com"
        u.hashed_password = "hash"
        u.full_name = "Alice"
        u.is_active = True
        u.role_id = "admin"
        u.created_at = datetime(2026, 1, 1)
        return u

    def test_returns_user_when_found(self):
        orm_user = self._make_orm_user()
        repo = UserRepository()
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(self._db_mock_with_user(orm_user)),
        ):
            user = repo.get_by_username("alice")
        assert user is not None
        assert user.username == "alice"

    def test_returns_none_when_not_found(self):
        repo = UserRepository()
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(self._db_mock_with_user(None)),
        ):
            user = repo.get_by_username("ghost")
        assert user is None


# ===========================================================================
# 7. UserRepository.get_by_email
# ===========================================================================

class TestGetByEmail:
    def _orm_user(self):
        u = MagicMock()
        u.id = _USER_UUID
        u.username = "alice"
        u.email = "alice@example.com"
        u.hashed_password = "hash"
        u.full_name = "Alice"
        u.is_active = True
        u.role_id = "admin"
        u.created_at = datetime(2026, 1, 1)
        return u

    def test_returns_user_when_found(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = self._orm_user()
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            user = repo.get_by_email("alice@example.com")
        assert user is not None

    def test_returns_none_when_not_found(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            user = repo.get_by_email("noone@example.com")
        assert user is None


# ===========================================================================
# 8. UserRepository.get_by_id
# ===========================================================================

class TestGetById:
    def test_returns_user_when_found(self):
        repo = UserRepository()
        orm_user = MagicMock()
        orm_user.id = _USER_UUID
        orm_user.username = "alice"
        orm_user.email = "a@a.com"
        orm_user.hashed_password = "h"
        orm_user.full_name = None
        orm_user.is_active = True
        orm_user.role_id = "admin"
        orm_user.created_at = datetime(2026, 1, 1)
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = orm_user
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            user = repo.get_by_id(_USER_UUID)
        assert user is not None
        assert user.id == _USER_UUID

    def test_returns_none_when_not_found(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            user = repo.get_by_id(uuid.uuid4())
        assert user is None


# ===========================================================================
# 9. UserRepository.create
# ===========================================================================

class TestUserRepositoryCreate:
    def test_raises_duplicate_on_existing_username(self):
        repo = UserRepository()
        existing = _fake_user()
        with patch.object(repo, "get_by_username", return_value=existing):
            with pytest.raises(DuplicateUserError) as exc_info:
                repo.create(UserCreate(
                    username="testuser",
                    email="new@example.com",
                    password="password123",
                    role_id="analyst",
                ))
        assert exc_info.value.field == "username"

    def test_raises_duplicate_on_existing_email(self):
        repo = UserRepository()
        existing = _fake_user()
        with patch.object(repo, "get_by_username", return_value=None), \
             patch.object(repo, "get_by_email", return_value=existing):
            with pytest.raises(DuplicateUserError) as exc_info:
                repo.create(UserCreate(
                    username="newuser",
                    email="test@example.com",
                    password="password123",
                    role_id="analyst",
                ))
        assert exc_info.value.field == "email"

    def test_creates_user_and_returns_it(self):
        repo = UserRepository()
        new_user = _fake_user(username="newuser", email="new@example.com")
        db_mock = MagicMock()

        with patch.object(repo, "get_by_username", return_value=None), \
             patch.object(repo, "get_by_email", return_value=None), \
             patch.object(repo, "get_by_id", return_value=new_user), \
             patch(
                 "app.repositories.auth.user_repository.SessionLocal",
                 return_value=_cm(db_mock),
             ), \
             patch("app.repositories.auth.user_repository.get_password_hash",
                   return_value="hashed"):
            created = repo.create(UserCreate(
                username="newuser",
                email="new@example.com",
                password="password123",
                role_id="analyst",
            ))

        assert created.username == "newuser"
        db_mock.add.assert_called_once()
        db_mock.commit.assert_called_once()

    def test_raises_runtime_error_when_re_read_fails(self):
        repo = UserRepository()
        db_mock = MagicMock()
        with patch.object(repo, "get_by_username", return_value=None), \
             patch.object(repo, "get_by_email", return_value=None), \
             patch.object(repo, "get_by_id", return_value=None), \
             patch(
                 "app.repositories.auth.user_repository.SessionLocal",
                 return_value=_cm(db_mock),
             ), \
             patch("app.repositories.auth.user_repository.get_password_hash",
                   return_value="hashed"):
            with pytest.raises(RuntimeError, match="could not re-read"):
                repo.create(UserCreate(
                    username="newuser2",
                    email="new2@example.com",
                    password="password123",
                    role_id="analyst",
                ))


# ===========================================================================
# 10. UserRepository.update_active_status
# ===========================================================================

class TestUpdateActiveStatus:
    def test_updates_and_returns_user(self):
        repo = UserRepository()
        orm_user = MagicMock()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = orm_user
        updated_user = _fake_user(is_active=False)

        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch.object(repo, "get_by_id", return_value=updated_user):
            result = repo.update_active_status(_fake_user(), False)

        assert result.is_active is False
        db_mock.commit.assert_called_once()

    def test_raises_when_user_not_found(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            with pytest.raises(ValueError, match="not found"):
                repo.update_active_status(_fake_user(), True)

    def test_raises_runtime_error_when_re_read_fails(self):
        repo = UserRepository()
        orm_user = MagicMock()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = orm_user
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch.object(repo, "get_by_id", return_value=None):
            with pytest.raises(RuntimeError, match="could not re-read"):
                repo.update_active_status(_fake_user(), True)


# ===========================================================================
# 11. UserRepository.update_password
# ===========================================================================

class TestUpdatePassword:
    def test_updates_hashed_password(self):
        repo = UserRepository()
        orm_user = MagicMock()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = orm_user

        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ), patch("app.repositories.auth.user_repository.get_password_hash",
                 return_value="new-hash"):
            repo.update_password(_fake_user(), "newpassword")

        assert orm_user.hashed_password == "new-hash"
        db_mock.commit.assert_called_once()

    def test_raises_when_user_not_found(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.filter.return_value.first.return_value = None
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            with pytest.raises(ValueError, match="not found"):
                repo.update_password(_fake_user(), "anypassword")


# ===========================================================================
# 12. UserRepository.list_users
# ===========================================================================

class TestListUsers:
    def test_returns_list_of_users(self):
        repo = UserRepository()
        orm_users = []
        for i in range(3):
            u = MagicMock()
            u.id = uuid.uuid4()
            u.username = f"user{i}"
            u.email = f"user{i}@example.com"
            u.hashed_password = "hash"
            u.full_name = None
            u.is_active = True
            u.role_id = "analyst"
            u.created_at = datetime(2026, 1, 1)
            orm_users.append(u)

        db_mock = MagicMock()
        db_mock.query.return_value.order_by.return_value.all.return_value = orm_users

        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            users = repo.list_users()

        assert len(users) == 3

    def test_empty_database_returns_empty_list(self):
        repo = UserRepository()
        db_mock = MagicMock()
        db_mock.query.return_value.order_by.return_value.all.return_value = []
        with patch(
            "app.repositories.auth.user_repository.SessionLocal",
            return_value=_cm(db_mock),
        ):
            users = repo.list_users()
        assert users == []


# ===========================================================================
# 13. AuthService.authenticate_user
# ===========================================================================

class TestAuthServiceAuthenticateUser:
    def _make_service(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        return service

    def test_raises_401_when_user_not_found(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            service.authenticate_user(LoginRequest(username="nobody", password="pw"))
        assert exc_info.value.status_code == 401

    def test_raises_401_when_password_wrong(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = _fake_user()
        with patch("app.services.auth.service.verify_password", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                service.authenticate_user(LoginRequest(username="testuser", password="wrong"))
        assert exc_info.value.status_code == 401
        assert "Incorrect" in exc_info.value.detail

    def test_raises_403_when_user_inactive(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = _fake_user(is_active=False)
        with patch("app.services.auth.service.verify_password", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                service.authenticate_user(LoginRequest(username="testuser", password="pw"))
        assert exc_info.value.status_code == 403
        assert "Inactive" in exc_info.value.detail

    def test_returns_token_on_success(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = _fake_user()
        with patch("app.services.auth.service.verify_password", return_value=True), \
             patch("app.services.auth.service.create_access_token",
                   return_value="mock.jwt.token"):
            token = service.authenticate_user(LoginRequest(username="testuser", password="pw"))
        assert token.access_token == "mock.jwt.token"
        assert token.token_type == "bearer"

    def test_access_token_created_with_user_id_as_subject(self):
        service = self._make_service()
        user = _fake_user()
        service.repo.get_by_username.return_value = user
        captured = {}
        with patch("app.services.auth.service.verify_password", return_value=True), \
             patch("app.services.auth.service.create_access_token",
                   side_effect=lambda subject: (captured.update({"sub": subject}) or "tok")):
            service.authenticate_user(LoginRequest(username="testuser", password="pw"))
        assert captured["sub"] == user.id


# ===========================================================================
# 14. AuthService.create_user
# ===========================================================================

class TestAuthServiceCreateUser:
    def _make_service(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        return service

    def _user_create(self, username="newuser", email="new@example.com"):
        return UserCreate(
            username=username,
            email=email,
            password="password123",
            role_id="analyst",
        )

    def test_raises_409_on_existing_username(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = _fake_user()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user(self._user_create())
        assert exc_info.value.status_code == 409
        assert "Username" in exc_info.value.detail

    def test_raises_409_on_existing_email(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = _fake_user()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user(self._user_create())
        assert exc_info.value.status_code == 409
        assert "Email" in exc_info.value.detail

    def test_raises_400_on_invalid_role(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = None
        service.repo.get_role_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            service.create_user(self._user_create())
        assert exc_info.value.status_code == 400
        assert "role_id" in exc_info.value.detail

    def test_returns_user_read_on_success(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = None
        service.repo.get_role_by_id.return_value = SimpleNamespace(id="analyst", name="Analyst")
        new_user = _fake_user(username="newuser", email="new@example.com")
        service.repo.create.return_value = new_user
        result = service.create_user(self._user_create())
        assert result.username == "newuser"

    def test_propagates_duplicate_user_error_as_409(self):
        from app.repositories.auth.user_repository import DuplicateUserError
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = None
        service.repo.get_role_by_id.return_value = SimpleNamespace(id="analyst", name="Analyst")
        service.repo.create.side_effect = DuplicateUserError("username")
        with pytest.raises(HTTPException) as exc_info:
            service.create_user(self._user_create())
        assert exc_info.value.status_code == 409


# ===========================================================================
# 15. AuthService.create_user_by_admin
# ===========================================================================

class TestAuthServiceCreateUserByAdmin:
    def _make_service(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        return service

    def _admin_create(self):
        return AdminUserCreate(
            username="adminuser",
            email="admin@example.com",
            password="adminpass1",
            role_id="admin",
            is_active=True,
        )

    def test_raises_409_on_existing_username(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = _fake_user()
        with pytest.raises(HTTPException) as exc_info:
            service.create_user_by_admin(self._admin_create())
        assert exc_info.value.status_code == 409

    def test_raises_400_on_invalid_role(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = None
        service.repo.get_role_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            service.create_user_by_admin(self._admin_create())
        assert exc_info.value.status_code == 400

    def test_returns_user_read_on_success(self):
        service = self._make_service()
        service.repo.get_by_username.return_value = None
        service.repo.get_by_email.return_value = None
        service.repo.get_role_by_id.return_value = SimpleNamespace(id="admin", name="Admin")
        new_user = _fake_user(username="adminuser", email="admin@example.com", role_id="admin")
        service.repo.create.return_value = new_user
        result = service.create_user_by_admin(self._admin_create())
        assert result.username == "adminuser"


# ===========================================================================
# 16. AuthService.list_users
# ===========================================================================

class TestAuthServiceListUsers:
    def test_returns_list_of_user_read(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        users = [_fake_user(username=f"user{i}", email=f"u{i}@a.com") for i in range(3)]
        service.repo.list_users.return_value = users
        result = service.list_users()
        assert len(result) == 3

    def test_empty_returns_empty_list(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        service.repo.list_users.return_value = []
        result = service.list_users()
        assert result == []


# ===========================================================================
# 17. AuthService.update_user_status
# ===========================================================================

class TestAuthServiceUpdateUserStatus:
    def _make_service(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        return service

    def test_raises_404_when_user_not_found(self):
        service = self._make_service()
        service.repo.get_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            service.update_user_status(
                user_id=_USER_UUID,
                is_active=False,
                current_user=_fake_user(),
            )
        assert exc_info.value.status_code == 404

    def test_raises_400_when_self_deactivation(self):
        service = self._make_service()
        current = _fake_user()
        service.repo.get_by_id.return_value = current  # same user
        with pytest.raises(HTTPException) as exc_info:
            service.update_user_status(
                user_id=current.id,
                is_active=False,
                current_user=current,
            )
        assert exc_info.value.status_code == 400

    def test_returns_updated_user_on_success(self):
        service = self._make_service()
        target = _fake_user(id=uuid.uuid4())  # different UUID
        current = _fake_user()
        service.repo.get_by_id.return_value = target
        updated = _fake_user(is_active=True, id=target.id, username=target.username, email=target.email)
        service.repo.update_active_status.return_value = updated
        result = service.update_user_status(
            user_id=target.id,
            is_active=True,
            current_user=current,
        )
        assert result.is_active is True


# ===========================================================================
# 18. AuthService.reset_user_password
# ===========================================================================

class TestAuthServiceResetUserPassword:
    def _make_service(self):
        service = AuthService.__new__(AuthService)
        service.repo = MagicMock()
        return service

    def test_raises_400_when_password_too_short(self):
        service = self._make_service()
        with pytest.raises(HTTPException) as exc_info:
            service.reset_user_password(user_id=_USER_UUID, new_password="short")
        assert exc_info.value.status_code == 400
        assert "8 characters" in exc_info.value.detail

    def test_raises_404_when_user_not_found(self):
        service = self._make_service()
        service.repo.get_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            service.reset_user_password(user_id=_USER_UUID, new_password="longpassword")
        assert exc_info.value.status_code == 404

    def test_calls_update_password_on_success(self):
        service = self._make_service()
        user = _fake_user()
        service.repo.get_by_id.return_value = user
        service.reset_user_password(user_id=_USER_UUID, new_password="validpassword")
        service.repo.update_password.assert_called_once_with(user, "validpassword")

    def test_exactly_8_chars_is_allowed(self):
        service = self._make_service()
        user = _fake_user()
        service.repo.get_by_id.return_value = user
        service.reset_user_password(user_id=_USER_UUID, new_password="12345678")
        service.repo.update_password.assert_called_once()


# ===========================================================================
# 19. AuthService._raise_duplicate_user_error
# ===========================================================================

class TestRaiseDuplicateUserError:
    def test_username_maps_to_correct_message(self):
        err = DuplicateUserError("username")
        with pytest.raises(HTTPException) as exc_info:
            AuthService._raise_duplicate_user_error(err)
        assert exc_info.value.status_code == 409
        assert "Username" in exc_info.value.detail

    def test_email_maps_to_correct_message(self):
        err = DuplicateUserError("email")
        with pytest.raises(HTTPException) as exc_info:
            AuthService._raise_duplicate_user_error(err)
        assert exc_info.value.status_code == 409
        assert "Email" in exc_info.value.detail

    def test_unknown_field_uses_fallback_message(self):
        err = DuplicateUserError("phone")
        with pytest.raises(HTTPException) as exc_info:
            AuthService._raise_duplicate_user_error(err)
        assert exc_info.value.status_code == 409
        assert "already exists" in exc_info.value.detail
