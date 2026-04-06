import json
import unittest
from unittest.mock import MagicMock, patch

from app.services.solar_ai_chat.embedding_client import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    EmbeddingUnavailableError,
    GeminiEmbeddingClient,
)


class EmbedTextTests(unittest.TestCase):

    def _make_client(self) -> GeminiEmbeddingClient:
        return GeminiEmbeddingClient(
            api_key="test-key",
            base_url="https://example.com/v1beta",
            model="text-embedding-004",
        )

    def _mock_response(self, body: dict) -> MagicMock:
        resp = MagicMock()
        resp.read.return_value = json.dumps(body).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_text_returns_768_dim_vector(self, mock_urlopen: MagicMock) -> None:
        vector = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_urlopen.return_value = self._mock_response(
            {"embedding": {"values": vector}},
        )
        client = self._make_client()
        result = client.embed_text("hello world")
        self.assertEqual(len(result), DEFAULT_EMBEDDING_DIMENSIONS)
        self.assertAlmostEqual(result[0], 0.1)

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_text_raises_on_wrong_dimensions(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_response(
            {"embedding": {"values": [0.1, 0.2]}},
        )
        client = self._make_client()
        with self.assertRaises(EmbeddingUnavailableError):
            client.embed_text("hello")

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_text_raises_on_empty_embedding(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_response({"embedding": {}})
        client = self._make_client()
        with self.assertRaises(EmbeddingUnavailableError):
            client.embed_text("hello")

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_batch_returns_multiple_vectors(self, mock_urlopen: MagicMock) -> None:
        vector = [0.5] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_urlopen.return_value = self._mock_response(
            {"embeddings": [{"values": vector}, {"values": vector}]},
        )
        client = self._make_client()
        result = client.embed_batch(["text 1", "text 2"])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), DEFAULT_EMBEDDING_DIMENSIONS)

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_batch_empty_input_returns_empty(self, mock_urlopen: MagicMock) -> None:
        client = self._make_client()
        result = client.embed_batch([])
        self.assertEqual(result, [])
        mock_urlopen.assert_not_called()

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_batch_raises_on_count_mismatch(self, mock_urlopen: MagicMock) -> None:
        vector = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_urlopen.return_value = self._mock_response(
            {"embeddings": [{"values": vector}]},
        )
        client = self._make_client()
        with self.assertRaises(EmbeddingUnavailableError):
            client.embed_batch(["a", "b", "c"])

    @patch("app.services.solar_ai_chat.embedding_client.urlopen")
    def test_embed_text_builds_correct_url(self, mock_urlopen: MagicMock) -> None:
        vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_urlopen.return_value = self._mock_response(
            {"embedding": {"values": vector}},
        )
        client = self._make_client()
        client.embed_text("test")
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertIn("text-embedding-004:embedContent", request_obj.full_url)
        self.assertIn("key=test-key", request_obj.full_url)


if __name__ == "__main__":
    unittest.main()
