import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSIONS = 768


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
        endpoint = (
            f"{self._base_url}/models/{self._model}:embedContent"
            f"?key={self._api_key}"
        )
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
        }
        result = self._post(endpoint, payload)
        embedding = result.get("embedding", {})
        values = embedding.get("values", [])
        if not isinstance(values, list) or len(values) != self._dimensions:
            raise EmbeddingUnavailableError(
                f"Expected {self._dimensions}-dim vector, got {len(values)}."
            )
        return values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        endpoint = (
            f"{self._base_url}/models/{self._model}:batchEmbedContents"
            f"?key={self._api_key}"
        )
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

        vectors: list[list[float]] = []
        for emb in embeddings_list:
            values = emb.get("values", [])
            if not isinstance(values, list) or len(values) != self._dimensions:
                raise EmbeddingUnavailableError(
                    f"Expected {self._dimensions}-dim vector in batch, got {len(values)}."
                )
            vectors.append(values)
        return vectors

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
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
