import unittest
from unittest.mock import MagicMock, patch

from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat.rag import RagChunk


class VectorRepositoryUnitTests(unittest.TestCase):
    """Tests VectorRepository methods using a mocked psycopg2 connection."""

    def _make_repo(self) -> VectorRepository:
        settings = MagicMock()
        settings.pg_host = "localhost"
        settings.pg_port = 5432
        settings.pg_database = "test_db"
        settings.pg_user = "test_user"
        settings.pg_password = "test_pass"
        return VectorRepository(settings=settings)

    def test_upsert_empty_list_returns_zero(self) -> None:
        repo = self._make_repo()
        result = repo.upsert_chunks([])
        self.assertEqual(result, 0)

    def test_dsn_format(self) -> None:
        repo = self._make_repo()
        self.assertIn("host=localhost", repo._dsn)
        self.assertIn("port=5432", repo._dsn)
        self.assertIn("dbname=test_db", repo._dsn)

    @patch("app.repositories.solar_ai_chat.vector_repository.psycopg2")
    def test_upsert_calls_execute_batch(self, mock_psycopg2: MagicMock) -> None:
        repo = self._make_repo()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn
        repo._connect = MagicMock(return_value=mock_conn)

        chunks = [
            RagChunk(
                doc_type="incident_report",
                source_file="test.txt",
                chunk_index=0,
                content="test content",
                embedding=[0.1] * 768,
            ),
        ]
        result = repo.upsert_chunks(chunks)
        self.assertEqual(result, 1)
        mock_conn.commit.assert_called_once()

    def test_delete_by_source_commits(self) -> None:
        repo = self._make_repo()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connect = MagicMock(return_value=mock_conn)

        result = repo.delete_by_source("test.txt")
        self.assertEqual(result, 3)
        mock_conn.commit.assert_called_once()

    def test_search_similar_with_doc_type(self) -> None:
        repo = self._make_repo()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "content": "found chunk",
                "source_file": "report.txt",
                "doc_type": "incident_report",
                "score": 0.85,
            },
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connect = MagicMock(return_value=mock_conn)

        results = repo.search_similar(
            query_embedding=[0.1] * 768,
            top_k=5,
            doc_type="incident_report",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "found chunk")
        self.assertAlmostEqual(results[0].similarity_score, 0.85)

    def test_search_similar_without_doc_type(self) -> None:
        repo = self._make_repo()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connect = MagicMock(return_value=mock_conn)

        results = repo.search_similar(query_embedding=[0.0] * 768, top_k=3)
        self.assertEqual(len(results), 0)

    def test_count_chunks_returns_aggregated_stats(self) -> None:
        repo = self._make_repo()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"doc_type": "incident_report", "cnt": 10},
            {"doc_type": "model_changelog", "cnt": 5},
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        repo._connect = MagicMock(return_value=mock_conn)

        stats = repo.count_chunks()
        self.assertEqual(stats["total_chunks"], 15)
        self.assertEqual(stats["by_doc_type"]["incident_report"], 10)


if __name__ == "__main__":
    unittest.main()
