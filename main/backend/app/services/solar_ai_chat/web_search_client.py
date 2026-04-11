"""External web-search client for concept grounding in Solar AI Chat."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.settings import SolarChatSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    score: float | None = None


class WebSearchClient:
    """Small Tavily-compatible web-search client.

    The client is intentionally defensive: request failures return an empty
    result set so chat responses can safely fall back to deterministic paths.
    """

    def __init__(self, settings: SolarChatSettings) -> None:
        self._api_key = str(settings.websearch_api_key or "").strip()
        self._base_url = str(settings.websearch_base_url or "").strip()
        self._timeout = max(1.0, float(settings.websearch_timeout_seconds))
        self._max_results = max(1, int(settings.websearch_max_results))

    @property
    def enabled(self) -> bool:
        return bool(self._api_key and self._base_url)

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        if not self.enabled:
            return []

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        limit = self._max_results if max_results is None else max(1, int(max_results))
        payload: dict[str, object] = {
            "api_key": self._api_key,
            "query": normalized_query,
            "max_results": limit,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        response = self._post_json(payload)
        rows = response.get("results", []) if isinstance(response, dict) else []
        if not isinstance(rows, list):
            return []

        results: list[WebSearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            snippet = str(row.get("content") or "").strip()
            if not title or not url:
                continue

            raw_score = row.get("score")
            score: float | None
            if raw_score is None:
                score = None
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None

            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                )
            )

        return results

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            url=self._base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "pv-lakehouse-solar-ai-chat/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as request_error:
            body_preview = ""
            try:
                body_preview = request_error.read().decode("utf-8", errors="replace")
            except Exception:
                body_preview = ""
            trimmed = body_preview.strip()
            if len(trimmed) > 300:
                trimmed = trimmed[:300] + "..."
            logger.warning(
                "web_search_http_error code=%s reason=%s body=%s",
                request_error.code,
                request_error.reason,
                trimmed,
            )
            return {}
        except (URLError, TimeoutError) as request_error:
            logger.warning("web_search_request_error error=%s", request_error)
            return {}

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("web_search_parse_error body_is_not_json")
            return {}

        if not isinstance(parsed, dict):
            return {}
        return parsed
