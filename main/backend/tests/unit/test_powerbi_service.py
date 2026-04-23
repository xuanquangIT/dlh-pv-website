"""Unit tests for app.services.dashboard.powerbi_service — all external calls mocked."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import httpx
from app.schemas.dashboard import EmbedTokenResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_settings(
    tenant_id: str = "tenant-123",
    client_id: str = "client-abc",
    client_secret: str = "secret-xyz",
    workspace_id: str = "ws-001",
    report_id: str = "rep-002",
) -> MagicMock:
    s = MagicMock()
    s.tenant_id = tenant_id
    s.client_id = client_id
    s.client_secret = client_secret
    s.workspace_id = workspace_id
    s.report_id = report_id
    return s


def _empty_settings() -> MagicMock:
    s = MagicMock()
    s.tenant_id = None
    s.client_id = None
    s.client_secret = None
    s.workspace_id = None
    s.report_id = None
    return s


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestPowerBIServiceConstructor(unittest.TestCase):
    def test_authority_url_built_from_tenant_id(self) -> None:
        from app.services.dashboard.powerbi_service import PowerBIService

        mock_settings = _full_settings()
        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=mock_settings,
        ):
            svc = PowerBIService()

        self.assertEqual(
            svc.authority_url,
            f"https://login.microsoftonline.com/{mock_settings.tenant_id}",
        )

    def test_scope_is_powerbi_api(self) -> None:
        from app.services.dashboard.powerbi_service import PowerBIService

        mock_settings = _full_settings()
        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=mock_settings,
        ):
            svc = PowerBIService()

        self.assertEqual(svc.scope, ["https://analysis.windows.net/powerbi/api/.default"])


# ---------------------------------------------------------------------------
# _get_access_token
# ---------------------------------------------------------------------------

class TestGetAccessToken(unittest.TestCase):
    def _make_service(self, settings: MagicMock) -> object:
        from app.services.dashboard.powerbi_service import PowerBIService

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=settings,
        ):
            return PowerBIService()

    def test_returns_mock_token_when_credentials_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_ACCESS_TOKEN

        svc = self._make_service(_empty_settings())
        result = svc._get_access_token()
        self.assertEqual(result, MOCK_ACCESS_TOKEN)

    def test_returns_mock_token_when_only_client_id_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_ACCESS_TOKEN

        s = _full_settings()
        s.client_id = None
        svc = self._make_service(s)
        result = svc._get_access_token()
        self.assertEqual(result, MOCK_ACCESS_TOKEN)

    def test_returns_mock_token_when_only_client_secret_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_ACCESS_TOKEN

        s = _full_settings()
        s.client_secret = None
        svc = self._make_service(s)
        result = svc._get_access_token()
        self.assertEqual(result, MOCK_ACCESS_TOKEN)

    def test_returns_mock_token_when_only_tenant_id_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_ACCESS_TOKEN

        s = _full_settings()
        s.tenant_id = None
        svc = self._make_service(s)
        result = svc._get_access_token()
        self.assertEqual(result, MOCK_ACCESS_TOKEN)

    def test_returns_cached_token_when_silent_succeeds(self) -> None:
        svc = self._make_service(_full_settings())

        mock_app = MagicMock()
        mock_app.acquire_token_silent.return_value = {"access_token": "cached-token-abc"}

        with patch(
            "app.services.dashboard.powerbi_service.ConfidentialClientApplication",
            return_value=mock_app,
        ):
            result = svc._get_access_token()

        self.assertEqual(result, "cached-token-abc")
        mock_app.acquire_token_for_client.assert_not_called()

    def test_fetches_token_when_silent_returns_none(self) -> None:
        svc = self._make_service(_full_settings())

        mock_app = MagicMock()
        mock_app.acquire_token_silent.return_value = None
        mock_app.acquire_token_for_client.return_value = {"access_token": "new-token-xyz"}

        with patch(
            "app.services.dashboard.powerbi_service.ConfidentialClientApplication",
            return_value=mock_app,
        ):
            result = svc._get_access_token()

        self.assertEqual(result, "new-token-xyz")
        mock_app.acquire_token_for_client.assert_called_once()

    def test_returns_none_when_token_fetch_fails(self) -> None:
        svc = self._make_service(_full_settings())

        mock_app = MagicMock()
        mock_app.acquire_token_silent.return_value = None
        mock_app.acquire_token_for_client.return_value = {
            "error": "invalid_client",
            "error_description": "Bad credentials",
        }

        with patch(
            "app.services.dashboard.powerbi_service.ConfidentialClientApplication",
            return_value=mock_app,
        ):
            result = svc._get_access_token()

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# get_embed_info — no workspace/report configured
# ---------------------------------------------------------------------------

class TestGetEmbedInfoNoCredentials(unittest.TestCase):
    def _make_service(self, settings: MagicMock) -> object:
        from app.services.dashboard.powerbi_service import PowerBIService

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=settings,
        ):
            return PowerBIService()

    def test_returns_mock_embed_info_when_workspace_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_EMBED_TOKEN

        s = _full_settings()
        s.workspace_id = None
        svc = self._make_service(s)

        result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, MOCK_EMBED_TOKEN)
        self.assertEqual(result.report_id, "mock_report_id")

    def test_returns_mock_embed_info_when_report_missing(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_EMBED_TOKEN

        s = _full_settings()
        s.report_id = None
        svc = self._make_service(s)

        result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, MOCK_EMBED_TOKEN)

    def test_returns_mock_embed_info_when_access_token_is_mock(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_ACCESS_TOKEN, MOCK_EMBED_TOKEN

        s = _full_settings()
        svc = self._make_service(s)

        # Code checks `access_token == MOCK_ACCESS_TOKEN` to detect mock mode;
        # patching with MOCK_ACCESS_TOKEN triggers the mock embed-info branch.
        with patch.object(svc, "_get_access_token", return_value=MOCK_ACCESS_TOKEN):
            result = svc.get_embed_info()

        self.assertEqual(result.embed_token, MOCK_EMBED_TOKEN)

    def test_returns_mock_embed_info_when_access_token_is_none(self) -> None:
        from app.services.dashboard.powerbi_service import MOCK_EMBED_TOKEN

        s = _full_settings()
        svc = self._make_service(s)

        with patch.object(svc, "_get_access_token", return_value=None):
            result = svc.get_embed_info()

        self.assertEqual(result.embed_token, MOCK_EMBED_TOKEN)
        self.assertEqual(result.report_id, s.report_id)


# ---------------------------------------------------------------------------
# get_embed_info — full happy path
# ---------------------------------------------------------------------------

class TestGetEmbedInfoHappyPath(unittest.TestCase):
    def _make_service(self) -> object:
        from app.services.dashboard.powerbi_service import PowerBIService

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=_full_settings(),
        ):
            return PowerBIService()

    def _build_http_client_mock(
        self,
        embed_url: str = "https://app.powerbi.com/embed?reportId=rep-002",
        embed_token: str = "real-embed-token",
    ) -> MagicMock:
        report_resp = MagicMock()
        report_resp.raise_for_status = MagicMock()
        report_resp.json.return_value = {"embedUrl": embed_url}

        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {"token": embed_token}

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = report_resp
        mock_client_instance.post.return_value = token_resp
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        return mock_client_instance

    def test_full_happy_path_returns_real_tokens(self) -> None:
        svc = self._make_service()

        mock_client_instance = self._build_http_client_mock()

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, "real-embed-token")
        self.assertEqual(result.embed_url, "https://app.powerbi.com/embed?reportId=rep-002")
        self.assertEqual(result.report_id, "rep-002")

    def test_embed_url_missing_from_report_response(self) -> None:
        svc = self._make_service()

        mock_client_instance = self._build_http_client_mock(embed_url="")

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertEqual(result.embed_url, "")

    def test_token_missing_from_token_response(self) -> None:
        svc = self._make_service()

        report_resp = MagicMock()
        report_resp.raise_for_status = MagicMock()
        report_resp.json.return_value = {"embedUrl": "https://example.com/embed"}

        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {}  # no "token" key

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = report_resp
        mock_client_instance.post.return_value = token_resp
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertEqual(result.embed_token, "")


# ---------------------------------------------------------------------------
# get_embed_info — error paths
# ---------------------------------------------------------------------------

class TestGetEmbedInfoErrorPaths(unittest.TestCase):
    def _make_service(self) -> object:
        from app.services.dashboard.powerbi_service import PowerBIService

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=_full_settings(),
        ):
            return PowerBIService()

    def test_http_status_error_returns_fallback(self) -> None:
        svc = self._make_service()

        mock_response = MagicMock()
        mock_response.text = "Unauthorized"
        mock_response.status_code = 401
        http_err = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = http_err
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, "error_fetching_token")
        self.assertEqual(result.report_id, "rep-002")

    def test_request_error_returns_fallback(self) -> None:
        svc = self._make_service()

        network_err = httpx.RequestError("Connection refused", request=MagicMock())

        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = network_err
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, "error_fetching_token")

    def test_value_error_from_invalid_json_returns_fallback(self) -> None:
        svc = self._make_service()

        report_resp = MagicMock()
        report_resp.raise_for_status = MagicMock()
        report_resp.json.side_effect = ValueError("Invalid JSON")

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = report_resp
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertIsInstance(result, EmbedTokenResponse)
        self.assertEqual(result.embed_token, "error_fetching_token")

    def test_http_5xx_on_token_endpoint_returns_fallback(self) -> None:
        svc = self._make_service()

        report_resp = MagicMock()
        report_resp.raise_for_status = MagicMock()
        report_resp.json.return_value = {"embedUrl": "https://example.com/embed"}

        mock_token_response = MagicMock()
        mock_token_response.text = "Internal Server Error"
        mock_token_response.status_code = 500
        token_http_err = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=mock_token_response,
        )
        token_resp = MagicMock()
        token_resp.raise_for_status.side_effect = token_http_err

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = report_resp
        mock_client_instance.post.return_value = token_resp
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)

        with patch.object(svc, "_get_access_token", return_value="real-access-token"), \
             patch("app.services.dashboard.powerbi_service.httpx.Client", return_value=mock_client_instance):
            result = svc.get_embed_info()

        self.assertEqual(result.embed_token, "error_fetching_token")


# ---------------------------------------------------------------------------
# get_powerbi_service factory
# ---------------------------------------------------------------------------

class TestGetPowerBIServiceFactory(unittest.TestCase):
    def test_returns_powerbi_service_instance(self) -> None:
        from app.services.dashboard.powerbi_service import PowerBIService, get_powerbi_service

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=_empty_settings(),
        ):
            result = get_powerbi_service()

        self.assertIsInstance(result, PowerBIService)

    def test_each_call_returns_new_instance(self) -> None:
        from app.services.dashboard.powerbi_service import get_powerbi_service

        with patch(
            "app.services.dashboard.powerbi_service.get_powerbi_settings",
            return_value=_empty_settings(),
        ):
            svc1 = get_powerbi_service()
            svc2 = get_powerbi_service()

        self.assertIsNot(svc1, svc2)


if __name__ == "__main__":
    unittest.main()
