import logging
from typing import Any

import psycopg2
import psycopg2.extras

from app.core.settings import SolarChatSettings
from app.schemas.solar_ai_chat.rag import RagChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorRepository:
    """PostgreSQL-backed vector store using pgvector for similarity search."""

    def __init__(self, settings: SolarChatSettings) -> None:
        self._dsn = (
            f"host={settings.pg_host} port={settings.pg_port} "
            f"dbname={settings.pg_database} "
            f"user={settings.pg_user} password={settings.pg_password}"
        )

    def _connect(self) -> "psycopg2.extensions.connection":
        return psycopg2.connect(self._dsn)

    def upsert_chunks(self, chunks: list[RagChunk]) -> int:
        if not chunks:
            return 0

        sql = """
            INSERT INTO rag_documents (doc_type, source_file, chunk_index, content, embedding)
            VALUES (%(doc_type)s, %(source_file)s, %(chunk_index)s, %(content)s, %(embedding)s)
            ON CONFLICT (source_file, chunk_index)
            DO UPDATE SET content = EXCLUDED.content,
                          embedding = EXCLUDED.embedding,
                          doc_type = EXCLUDED.doc_type,
                          created_at = now()
        """
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            rows.append({
                "doc_type": chunk.doc_type,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": str(chunk.embedding),
            })

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows)
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        if doc_type:
            sql = """
                SELECT content, source_file, doc_type,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM rag_documents
                WHERE doc_type = %(dtype)s
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(k)s
            """
            params = {"emb": str(query_embedding), "dtype": doc_type, "k": top_k}
        else:
            sql = """
                SELECT content, source_file, doc_type,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM rag_documents
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(k)s
            """
            params = {"emb": str(query_embedding), "k": top_k}

        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            RetrievedChunk(
                content=row["content"],
                source_file=row["source_file"],
                doc_type=row["doc_type"],
                similarity_score=float(row["score"]),
            )
            for row in rows
        ]

    def delete_by_source(self, source_file: str) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_documents WHERE source_file = %s",
                    (source_file,),
                )
                deleted = cur.rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()

    def count_chunks(self, doc_type: str | None = None) -> dict[str, int]:
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT doc_type, COUNT(*) AS cnt FROM rag_documents GROUP BY doc_type"
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        by_type = {row["doc_type"]: int(row["cnt"]) for row in rows}
        total = sum(by_type.values())

        if doc_type:
            return {"total_chunks": by_type.get(doc_type, 0), "by_doc_type": {doc_type: by_type.get(doc_type, 0)}}

        return {"total_chunks": total, "by_doc_type": by_type}
