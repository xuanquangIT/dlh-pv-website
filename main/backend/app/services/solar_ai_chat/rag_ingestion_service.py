import logging
from pathlib import Path

from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat.rag import RagChunk
from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
EMBED_BATCH_SIZE = 20


class RagIngestionService:
    """Ingests text documents into pgvector for RAG retrieval."""

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_client: GeminiEmbeddingClient,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._vector_repo = vector_repo
        self._embedding_client = embedding_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest_document(self, file_path: Path, doc_type: str) -> int:
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            logger.warning("Empty document: %s", file_path.name)
            return 0

        raw_chunks = self._split_into_chunks(text)
        logger.info(
            "Chunked %s into %d pieces (size=%d, overlap=%d)",
            file_path.name, len(raw_chunks), self._chunk_size, self._chunk_overlap,
        )

        all_rag_chunks: list[RagChunk] = []
        for batch_start in range(0, len(raw_chunks), EMBED_BATCH_SIZE):
            batch_texts = raw_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            embeddings = self._embedding_client.embed_batch(batch_texts)

            for i, (content, embedding) in enumerate(zip(batch_texts, embeddings)):
                all_rag_chunks.append(RagChunk(
                    doc_type=doc_type,
                    source_file=file_path.name,
                    chunk_index=batch_start + i,
                    content=content,
                    embedding=embedding,
                ))

        inserted = self._vector_repo.upsert_chunks(all_rag_chunks)
        logger.info("Ingested %d chunks from %s", inserted, file_path.name)
        return inserted

    def delete_document(self, source_file: str) -> int:
        deleted = self._vector_repo.delete_by_source(source_file)
        logger.info("Deleted %d chunks for %s", deleted, source_file)
        return deleted

    def _split_into_chunks(self, text: str) -> list[str]:
        chunks: list[str] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks
