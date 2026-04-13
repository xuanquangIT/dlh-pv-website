import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.schemas.solar_ai_chat.rag import RagChunk
from app.services.solar_ai_chat.embedding_client import DEFAULT_EMBEDDING_DIMENSIONS
from app.services.solar_ai_chat.rag_ingestion_service import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RagIngestionService,
)


class ChunkSplittingTests(unittest.TestCase):

    def _make_service(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> RagIngestionService:
        return RagIngestionService(
            vector_repo=MagicMock(),
            embedding_client=MagicMock(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def test_short_text_produces_single_chunk(self) -> None:
        svc = self._make_service(chunk_size=100, chunk_overlap=20)
        chunks = svc._split_into_chunks("Hello world")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_long_text_produces_overlapping_chunks(self) -> None:
        text = "A" * 200
        svc = self._make_service(chunk_size=100, chunk_overlap=20)
        chunks = svc._split_into_chunks(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 100)

    def test_empty_text_produces_no_chunks(self) -> None:
        svc = self._make_service()
        chunks = svc._split_into_chunks("   ")
        self.assertEqual(len(chunks), 0)

    def test_overlap_creates_shared_content(self) -> None:
        text = "ABCDEFGHIJ" * 10
        svc = self._make_service(chunk_size=20, chunk_overlap=5)
        chunks = svc._split_into_chunks(text)
        self.assertGreater(len(chunks), 1)
        first_end = chunks[0][-5:]
        second_start = chunks[1][:5]
        self.assertEqual(first_end, second_start)


class IngestDocumentTests(unittest.TestCase):

    def test_ingest_calls_embed_and_upsert(self, ) -> None:
        mock_vector = MagicMock()
        mock_vector.upsert_chunks.return_value = 2

        mock_embed = MagicMock()
        vector = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_embed.embed_batch.return_value = [vector, vector]

        svc = RagIngestionService(
            vector_repo=mock_vector,
            embedding_client=mock_embed,
            chunk_size=50,
            chunk_overlap=10,
        )

        tmp_file = Path(__file__).parent / "_test_rag_doc.txt"
        tmp_file.write_text("A" * 100, encoding="utf-8")
        try:
            result = svc.ingest_document(tmp_file, doc_type="incident_report")
            self.assertEqual(result, 2)
            mock_embed.embed_batch.assert_called_once()
            mock_vector.upsert_chunks.assert_called_once()
            chunks_arg = mock_vector.upsert_chunks.call_args[0][0]
            self.assertEqual(len(chunks_arg), 2)
            self.assertIsInstance(chunks_arg[0], RagChunk)
            self.assertEqual(chunks_arg[0].doc_type, "incident_report")
        finally:
            tmp_file.unlink(missing_ok=True)

    def test_ingest_empty_file_returns_zero(self) -> None:
        svc = RagIngestionService(
            vector_repo=MagicMock(),
            embedding_client=MagicMock(),
        )
        tmp_file = Path(__file__).parent / "_test_rag_empty.txt"
        tmp_file.write_text("   ", encoding="utf-8")
        try:
            result = svc.ingest_document(tmp_file, doc_type="model_changelog")
            self.assertEqual(result, 0)
        finally:
            tmp_file.unlink(missing_ok=True)

    def test_delete_document_delegates_to_vector_repo(self) -> None:
        mock_vector = MagicMock()
        mock_vector.delete_by_source.return_value = 5
        svc = RagIngestionService(
            vector_repo=mock_vector,
            embedding_client=MagicMock(),
        )
        result = svc.delete_document("report.txt")
        self.assertEqual(result, 5)
        mock_vector.delete_by_source.assert_called_once_with("report.txt")


class VectorRepositoryMockTests(unittest.TestCase):

    def test_search_documents_unavailable_returns_message(self) -> None:
        from app.services.solar_ai_chat.tool_executor import ToolExecutor
        from app.schemas.solar_ai_chat.enums import ChatRole

        repo = MagicMock()
        executor = ToolExecutor(repository=repo, vector_repo=None, embedding_client=None)
        metrics, sources = executor.execute(
            "search_documents", {"query": "hello"}, ChatRole.ADMIN,
        )
        self.assertIn("not configured", metrics["message"])
        self.assertEqual(sources[0]["data_source"], "unavailable")

    def test_search_documents_with_mocked_vector(self) -> None:
        from app.services.solar_ai_chat.tool_executor import ToolExecutor
        from app.schemas.solar_ai_chat.enums import ChatRole
        from app.schemas.solar_ai_chat.rag import RetrievedChunk

        mock_vector = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_text.return_value = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_vector.search_similar.return_value = [
            RetrievedChunk(
                content="Sample chunk text",
                source_file="report.txt",
                doc_type="incident_report",
                similarity_score=0.92,
            ),
        ]

        repo = MagicMock()
        executor = ToolExecutor(
            repository=repo,
            vector_repo=mock_vector,
            embedding_client=mock_embed,
        )
        metrics, sources = executor.execute(
            "search_documents", {"query": "su co"}, ChatRole.ADMIN,
        )
        self.assertEqual(metrics["total_results"], 1)
        self.assertEqual(metrics["chunks"][0]["source_file"], "report.txt")
        self.assertEqual(sources[0]["data_source"], "pgvector")

    def test_search_documents_no_results(self) -> None:
        from app.services.solar_ai_chat.tool_executor import ToolExecutor
        from app.schemas.solar_ai_chat.enums import ChatRole

        mock_vector = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_text.return_value = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS
        mock_vector.search_similar.return_value = []

        repo = MagicMock()
        executor = ToolExecutor(
            repository=repo,
            vector_repo=mock_vector,
            embedding_client=mock_embed,
        )
        metrics, sources = executor.execute(
            "search_documents", {"query": "xyz"}, ChatRole.ADMIN,
        )
        self.assertIn("Khong tim thay", metrics["message"])

    def test_search_documents_empty_query_skips_embedding_call(self) -> None:
        from app.services.solar_ai_chat.tool_executor import ToolExecutor
        from app.schemas.solar_ai_chat.enums import ChatRole

        mock_vector = MagicMock()
        mock_embed = MagicMock()

        repo = MagicMock()
        executor = ToolExecutor(
            repository=repo,
            vector_repo=mock_vector,
            embedding_client=mock_embed,
        )
        metrics, sources = executor.execute(
            "search_documents", {}, ChatRole.ADMIN,
        )

        self.assertIn("Empty search query", metrics["message"])
        self.assertEqual(metrics["chunks"], [])
        self.assertEqual(metrics["total_results"], 0)
        self.assertEqual(sources[0]["data_source"], "pgvector")
        mock_embed.embed_text.assert_not_called()
        mock_vector.search_similar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
