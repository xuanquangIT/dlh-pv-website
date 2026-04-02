import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSIONS = 3072


class EmbeddingUnavailableError(RuntimeError):
    pass


class GeminiEmbeddingClient:
    """Client for Gemini text-embedding-004 API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "text-embedding-004",
        timeout: float = 10.0,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        # C4: API key sent as header, never in URL
        endpoint = f"{self._base_url}/models/{self._model}:embedContent"
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
        }
        result = self._post(endpoint, payload)
        values = result.get("embedding", {}).get("values", [])
        return self._validate_vector(values)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # C4: API key sent as header, never in URL
        endpoint = f"{self._base_url}/models/{self._model}:batchEmbedContents"
        requests_body = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
        payload = {"requests": requests_body}
        result = self._post(endpoint, payload)

        embeddings_list = result.get("embeddings", [])
        if not isinstance(embeddings_list, list) or len(embeddings_list) != len(texts):
            raise EmbeddingUnavailableError(
                f"Expected {len(texts)} embeddings, got {len(embeddings_list)}."
            )
        return [self._validate_vector(emb.get("values", [])) for emb in embeddings_list]

    def _validate_vector(self, values: list) -> list[float]:
        """Validate vector dimensions and return it. Raises EmbeddingUnavailableError on mismatch."""
        if not isinstance(values, list) or len(values) != self._dimensions:
            raise EmbeddingUnavailableError(
                f"Expected {self._dimensions}-dim vector, got {len(values)}."
            )
        return values

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
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
            logger.error("Embedding API HTTP error: %s %s", exc.code, exc.reason)
            raise EmbeddingUnavailableError(
                f"Embedding API error: {exc.code} {exc.reason}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            logger.error("Embedding API connection error: %s", exc)
            raise EmbeddingUnavailableError(
                f"Embedding API unreachable: {exc}"
            ) from exc
