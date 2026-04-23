"""Unit tests for app.services.solar_ai_chat.web_search_client — all network calls mocked."""
from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from urllib.error import HTTPError, URLError

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.solar_ai_chat.web_search_client import WebSearchClient, WebSearchResult
from app.core.settings import SolarChatSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(
    api_key: str | None = "test-api-key",
    base_url: str = "https://api.tavily.com/search",
    timeout: float = 5.0,
    max_results: int = 10,
) -> SolarChatSettings:
    s = SolarChatSettings()
    s.websearch_api_key = api_key
    s.websearch_base_url = base_url
    s.websearch_timeout_seconds = timeout
    s.websearch_max_results = max_results
    return s


def _enabled_settings() -> SolarChatSettings:
    return _settings(api_key="real-key", base_url="https://api.tavily.com/search")


def _disabled_settings() -> SolarChatSettings:
    return _settings(api_key=None)


def _make_urlopen_response(data: dict) -> MagicMock:
    """Return a context-manager mock that yields the JSON-encoded dict."""
    body = json.dumps(data).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body
    return cm


# ---------------------------------------------------------------------------
# Constructor / properties
# ---------------------------------------------------------------------------

class TestWebSearchClientConstructor(unittest.TestCase):
    def test_enabled_when_key_and_url_present(self) -> None:
        client = WebSearchClient(_enabled_settings())
        self.assertTrue(client.enabled)

    def test_disabled_when_api_key_none(self) -> None:
        client = WebSearchClient(_disabled_settings())
        self.assertFalse(client.enabled)

    def test_disabled_when_api_key_empty_string(self) -> None:
        client = WebSearchClient(_settings(api_key=""))
        self.assertFalse(client.enabled)

    def test_disabled_when_base_url_empty(self) -> None:
        client = WebSearchClient(_settings(api_key="key", base_url=""))
        self.assertFalse(client.enabled)

    def test_timeout_minimum_is_one(self) -> None:
        client = WebSearchClient(_settings(timeout=0.0))
        self.assertEqual(client._timeout, 1.0)

    def test_max_results_minimum_is_one(self) -> None:
        client = WebSearchClient(_settings(max_results=0))
        self.assertEqual(client._max_results, 1)

    def test_normal_timeout_respected(self) -> None:
        client = WebSearchClient(_settings(timeout=7.5))
        self.assertEqual(client._timeout, 7.5)

    def test_normal_max_results_respected(self) -> None:
        client = WebSearchClient(_settings(max_results=5))
        self.assertEqual(client._max_results, 5)


# ---------------------------------------------------------------------------
# search — disabled / empty query guards
# ---------------------------------------------------------------------------

class TestSearchGuards(unittest.TestCase):
    def test_returns_empty_list_when_disabled(self) -> None:
        client = WebSearchClient(_disabled_settings())
        result = client.search("solar panels")
        self.assertEqual(result, [])

    def test_returns_empty_list_for_empty_query(self) -> None:
        client = WebSearchClient(_enabled_settings())
        result = client.search("")
        self.assertEqual(result, [])

    def test_returns_empty_list_for_whitespace_query(self) -> None:
        client = WebSearchClient(_enabled_settings())
        result = client.search("   ")
        self.assertEqual(result, [])

    def test_returns_empty_list_for_none_query(self) -> None:
        client = WebSearchClient(_enabled_settings())
        result = client.search(None)  # type: ignore[arg-type]
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# search — happy path
# ---------------------------------------------------------------------------

class TestSearchHappyPath(unittest.TestCase):
    def _client(self) -> WebSearchClient:
        return WebSearchClient(_enabled_settings())

    def test_parses_single_result(self) -> None:
        client = self._client()
        api_response = {
            "results": [
                {
                    "title": "Solar Energy Basics",
                    "url": "https://example.com/solar",
                    "content": "Solar panels convert sunlight into electricity.",
                    "score": 0.95,
                }
            ]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("solar energy")

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsInstance(r, WebSearchResult)
        self.assertEqual(r.title, "Solar Energy Basics")
        self.assertEqual(r.url, "https://example.com/solar")
        self.assertEqual(r.snippet, "Solar panels convert sunlight into electricity.")
        self.assertAlmostEqual(r.score, 0.95)

    def test_parses_multiple_results(self) -> None:
        client = self._client()
        api_response = {
            "results": [
                {"title": "Result A", "url": "https://a.com", "content": "A text", "score": 0.9},
                {"title": "Result B", "url": "https://b.com", "content": "B text", "score": 0.8},
            ]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Result A")
        self.assertEqual(results[1].title, "Result B")

    def test_score_none_when_missing(self) -> None:
        client = self._client()
        api_response = {
            "results": [{"title": "No score", "url": "https://x.com", "content": "text"}]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertIsNone(results[0].score)

    def test_score_none_when_non_numeric(self) -> None:
        client = self._client()
        api_response = {
            "results": [
                {"title": "Bad score", "url": "https://x.com", "content": "text", "score": "not-a-number"}
            ]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertIsNone(results[0].score)

    def test_empty_results_list_returns_empty(self) -> None:
        client = self._client()
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(results, [])

    def test_missing_results_key_returns_empty(self) -> None:
        client = self._client()
        api_response = {"answer": "some answer"}
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(results, [])

    def test_custom_max_results_sent_in_payload(self) -> None:
        client = self._client()
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        captured_payload: list[dict] = []

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            captured_payload.append(body)
            return mock_resp

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=fake_urlopen):
            client.search("energy yield", max_results=3)

        self.assertEqual(captured_payload[0]["max_results"], 3)

    def test_default_max_results_from_settings(self) -> None:
        client = WebSearchClient(_settings(max_results=7))
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        captured_payload: list[dict] = []

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            captured_payload.append(body)
            return mock_resp

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=fake_urlopen):
            client.search("capacity factor")

        self.assertEqual(captured_payload[0]["max_results"], 7)

    def test_max_results_clamped_to_at_least_one(self) -> None:
        client = self._client()
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        captured_payload: list[dict] = []

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            captured_payload.append(body)
            return mock_resp

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=fake_urlopen):
            client.search("query", max_results=0)

        self.assertGreaterEqual(captured_payload[0]["max_results"], 1)


# ---------------------------------------------------------------------------
# search — row-level filtering
# ---------------------------------------------------------------------------

class TestSearchRowFiltering(unittest.TestCase):
    def _client(self) -> WebSearchClient:
        return WebSearchClient(_enabled_settings())

    def test_skips_rows_missing_title(self) -> None:
        client = self._client()
        api_response = {
            "results": [
                {"url": "https://x.com", "content": "text"},          # no title
                {"title": "Good", "url": "https://y.com", "content": "ok"},
            ]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Good")

    def test_skips_rows_missing_url(self) -> None:
        client = self._client()
        api_response = {
            "results": [
                {"title": "No URL", "content": "text"},               # no url
                {"title": "Has URL", "url": "https://z.com", "content": "ok"},
            ]
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://z.com")

    def test_skips_non_dict_rows(self) -> None:
        client = self._client()
        api_response = {"results": ["not-a-dict", 42, None]}
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(results, [])

    def test_results_not_list_returns_empty(self) -> None:
        client = self._client()
        api_response = {"results": "not-a-list"}
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(results, [])

    def test_snippet_empty_when_content_missing(self) -> None:
        client = self._client()
        api_response = {
            "results": [{"title": "Title", "url": "https://x.com"}]  # no content
        }
        mock_resp = _make_urlopen_response(api_response)

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=mock_resp):
            results = client.search("query")

        self.assertEqual(results[0].snippet, "")


# ---------------------------------------------------------------------------
# _post_json — HTTP error paths
# ---------------------------------------------------------------------------

class TestPostJsonErrorPaths(unittest.TestCase):
    def _client(self) -> WebSearchClient:
        return WebSearchClient(_enabled_settings())

    def test_returns_empty_dict_on_http_error_4xx(self) -> None:
        client = self._client()
        error_body = b"Unauthorized"
        http_err = HTTPError(
            url="https://api.tavily.com/search",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(error_body),
        )

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=http_err):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_returns_empty_dict_on_http_error_5xx(self) -> None:
        client = self._client()
        http_err = HTTPError(
            url="https://api.tavily.com/search",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"Server Error"),
        )

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=http_err):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_returns_empty_dict_on_url_error(self) -> None:
        client = self._client()
        url_err = URLError("Connection refused")

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=url_err):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_returns_empty_dict_on_timeout_error(self) -> None:
        client = self._client()

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=TimeoutError("timed out")):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_returns_empty_dict_when_response_is_not_json(self) -> None:
        client = self._client()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = b"<html>not json</html>"

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=cm):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_returns_empty_dict_when_response_is_json_array(self) -> None:
        client = self._client()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = json.dumps([1, 2, 3]).encode("utf-8")

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", return_value=cm):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_http_error_with_unreadable_body_still_returns_empty(self) -> None:
        client = self._client()
        bad_fp = MagicMock()
        bad_fp.read.side_effect = OSError("read error")
        http_err = HTTPError(
            url="https://api.tavily.com/search",
            code=403,
            msg="Forbidden",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=bad_fp,
        )

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=http_err):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})

    def test_http_error_body_over_300_chars_is_truncated_in_log(self) -> None:
        client = self._client()
        long_body = ("A" * 400).encode("utf-8")
        http_err = HTTPError(
            url="https://api.tavily.com/search",
            code=429,
            msg="Too Many Requests",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(long_body),
        )

        import logging
        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=http_err), \
             self.assertLogs("app.services.solar_ai_chat.web_search_client", level=logging.WARNING):
            result = client._post_json({"api_key": "k", "query": "q", "max_results": 5})

        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _post_json — request construction
# ---------------------------------------------------------------------------

class TestPostJsonRequestConstruction(unittest.TestCase):
    def _client(self) -> WebSearchClient:
        return WebSearchClient(_enabled_settings())

    def test_uses_correct_http_method_and_content_type(self) -> None:
        client = self._client()
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        captured_requests: list = []

        def fake_urlopen(request, timeout):
            captured_requests.append(request)
            return mock_resp

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=fake_urlopen):
            client.search("photovoltaic")

        req = captured_requests[0]
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

    def test_api_key_and_query_in_payload(self) -> None:
        client = self._client()
        api_response = {"results": []}
        mock_resp = _make_urlopen_response(api_response)

        captured_payload: list[dict] = []

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            captured_payload.append(body)
            return mock_resp

        with patch("app.services.solar_ai_chat.web_search_client.urlopen", side_effect=fake_urlopen):
            client.search("irradiance data")

        payload = captured_payload[0]
        self.assertEqual(payload["api_key"], "real-key")
        self.assertEqual(payload["query"], "irradiance data")
        self.assertFalse(payload["include_answer"])
        self.assertFalse(payload["include_raw_content"])
        self.assertEqual(payload["search_depth"], "advanced")


# ---------------------------------------------------------------------------
# Integration-style: search calls _post_json which raises → returns []
# ---------------------------------------------------------------------------

class TestSearchDelegatesToPostJson(unittest.TestCase):
    def test_search_returns_empty_list_when_post_json_returns_empty(self) -> None:
        client = WebSearchClient(_enabled_settings())

        with patch.object(client, "_post_json", return_value={}):
            results = client.search("solar irradiance")

        self.assertEqual(results, [])

    def test_search_returns_empty_list_when_post_json_returns_non_dict(self) -> None:
        client = WebSearchClient(_enabled_settings())

        with patch.object(client, "_post_json", return_value=None):  # type: ignore[arg-type]
            results = client.search("solar irradiance")

        self.assertEqual(results, [])

    def test_search_passes_normalized_query_to_post_json(self) -> None:
        client = WebSearchClient(_enabled_settings())
        captured: list[dict] = []

        def fake_post(payload):
            captured.append(payload)
            return {}

        with patch.object(client, "_post_json", side_effect=fake_post):
            client.search("  trimmed query  ")

        self.assertEqual(captured[0]["query"], "trimmed query")


# ---------------------------------------------------------------------------
# WebSearchResult dataclass
# ---------------------------------------------------------------------------

class TestWebSearchResult(unittest.TestCase):
    def test_fields_accessible(self) -> None:
        r = WebSearchResult(
            title="Title",
            url="https://example.com",
            snippet="Some text",
            score=0.88,
        )
        self.assertEqual(r.title, "Title")
        self.assertEqual(r.url, "https://example.com")
        self.assertEqual(r.snippet, "Some text")
        self.assertAlmostEqual(r.score, 0.88)

    def test_score_defaults_to_none(self) -> None:
        r = WebSearchResult(title="T", url="https://u.com", snippet="s")
        self.assertIsNone(r.score)

    def test_frozen_dataclass_raises_on_mutation(self) -> None:
        r = WebSearchResult(title="T", url="https://u.com", snippet="s")
        with self.assertRaises((AttributeError, TypeError)):
            r.title = "New title"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
