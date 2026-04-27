from pydantic import BaseModel, Field


class RagChunk(BaseModel):
    doc_type: str
    source_file: str
    chunk_index: int
    content: str
    embedding: list[float] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    content: str
    source_file: str
    doc_type: str
    similarity_score: float


class RagSearchResult(BaseModel):
    chunks: list[RetrievedChunk]
    query_text: str


class IngestDocumentRequest(BaseModel):
    file_path: str
    doc_type: str = Field(
        pattern=r"^(incident_report|equipment_manual|model_changelog)$",
    )


class IngestDocumentResponse(BaseModel):
    source_file: str
    doc_type: str
    chunks_ingested: int


class RagStatsResponse(BaseModel):
    total_chunks: int
    by_doc_type: dict[str, int]
