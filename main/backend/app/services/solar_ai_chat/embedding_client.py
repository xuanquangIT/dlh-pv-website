import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSIONS = 3072

# HTTP status codes that indicate the *current key* is at fault and a different
# key may succeed. 5xx and connection errors are not key-specific so don't fail
# over (would just burn the fallback's quota for nothing).
_KEY_SCOPED_HTTP_STATUSES = frozenset({401, 403, 429})


class EmbeddingUnavailableError(RuntimeError):
    pass


class _KeyScopedEmbeddingError(EmbeddingUnavailableError):
    """Raised when the *current* key failed in a way another key might fix
    (auth revoked, per-key rate limit, per-key quota). Carrier for the
    fail-over loop — never leaks past ``_post``."""


class GeminiEmbeddingClient:
    """Client for Gemini text-embedding API with primary/fallback key fail-over.

    Pass ``api_key`` for single-key use, or ``api_keys`` (ordered: primary
    first, fallback second) to enable fail-over on key-scoped errors
    (401/403/429 + RESOURCE_EXHAUSTED). Once a fallback succeeds, it becomes
    the active key for subsequent calls until *it* fails too.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "text-embedding-004",
        timeout: float = 10.0,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        api_keys: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        keys: list[str] = []
        if api_keys:
            for k in api_keys:
                if k and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        if api_key and api_key.strip() and api_key.strip() not in keys:
            keys.insert(0, api_key.strip())
        if not keys:
            raise ValueError("GeminiEmbeddingClient requires at least one API key.")

        self._api_keys: list[str] = keys
        self._active_index: int = 0
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            raise EmbeddingUnavailableError("Embedding input text is empty after trimming.")

        endpoint = f"{self._base_url}/models/{self._model}:embedContent"
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": normalized_text}]},
        }
        result = self._post_with_failover(endpoint, payload)
        values = result.get("embedding", {}).get("values", [])
        return self._validate_vector(values)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        normalized_texts: list[str] = []
        for index, text in enumerate(texts):
            normalized_text = " ".join(str(text or "").split()).strip()
            if not normalized_text:
                raise EmbeddingUnavailableError(
                    f"Embedding input text at index {index} is empty after trimming."
                )
            normalized_texts.append(normalized_text)

        endpoint = f"{self._base_url}/models/{self._model}:batchEmbedContents"
        requests_body = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in normalized_texts
        ]
        payload = {"requests": requests_body}
        try:
            result = self._post_with_failover(endpoint, payload)
        except EmbeddingUnavailableError as exc:
            if "location is not supported" in str(exc).lower():
                raise
            # Some embedding endpoints reject batch mode. Fall back to sequential calls.
            return [self.embed_text(text) for text in normalized_texts]

        embeddings_list = result.get("embeddings", [])
        if not isinstance(embeddings_list, list) or len(embeddings_list) != len(normalized_texts):
            raise EmbeddingUnavailableError(
                f"Expected {len(normalized_texts)} embeddings, got {len(embeddings_list)}."
            )
        return [self._validate_vector(emb.get("values", [])) for emb in embeddings_list]

    def _validate_vector(self, values: list) -> list[float]:
        """Validate vector dimensions and return it. Raises EmbeddingUnavailableError on mismatch."""
        if not isinstance(values, list) or len(values) != self._dimensions:
            raise EmbeddingUnavailableError(
                f"Expected {self._dimensions}-dim vector, got {len(values)}."
            )
        return values

    def _post_with_failover(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Try active key first; on key-scoped failure, advance to the next key
        and retry. Stops once every key has been tried."""
        last_error: EmbeddingUnavailableError | None = None
        for attempt in range(len(self._api_keys)):
            key_index = (self._active_index + attempt) % len(self._api_keys)
            try:
                result = self._post(url, payload, self._api_keys[key_index])
                if key_index != self._active_index:
                    logger.warning(
                        "embedding_key_failover_succeeded from_index=%d to_index=%d",
                        self._active_index, key_index,
                    )
                    self._active_index = key_index
                return result
            except _KeyScopedEmbeddingError as exc:
                last_error = exc
                if len(self._api_keys) > 1 and attempt + 1 < len(self._api_keys):
                    logger.warning(
                        "embedding_key_failed key_index=%d will_retry_with_next reason=%s",
                        key_index, str(exc)[:200],
                    )
                continue
        assert last_error is not None
        raise last_error

    def _post(self, url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict):
                    raise EmbeddingUnavailableError("Unexpected response format.")
                return body
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive read for non-standard HTTPError objects
                error_body = ""

            if "User location is not supported for the API use" in error_body:
                logger.warning("Embedding API location is not supported for this key/project.")
                raise EmbeddingUnavailableError(
                    "Embedding API location is not supported for this key/project."
                ) from exc

            logger.error("Embedding API HTTP error: %s %s", exc.code, exc.reason)

            # Decide whether the key itself is at fault (worth a fail-over) or
            # the request/server is at fault (fail-over would just waste the
            # other key's quota).
            is_key_scoped = (
                exc.code in _KEY_SCOPED_HTTP_STATUSES
                or "RESOURCE_EXHAUSTED" in error_body
                or "PERMISSION_DENIED" in error_body
            )
            err_cls = _KeyScopedEmbeddingError if is_key_scoped else EmbeddingUnavailableError

            if error_body:
                raise err_cls(
                    f"Embedding API error: {exc.code} {exc.reason} | body: {error_body[:300]}"
                ) from exc
            raise err_cls(
                f"Embedding API error: {exc.code} {exc.reason}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            logger.error("Embedding API connection error: %s", exc)
            raise EmbeddingUnavailableError(
                f"Embedding API unreachable: {exc}"
            ) from exc
