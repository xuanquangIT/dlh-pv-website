from pathlib import Path
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import require_role
from app.db.database import AuthUser, SessionLocal
from app.core.settings import get_solar_chat_settings
from app.repositories.solar_ai_chat.history_repository import ChatHistoryRepository
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.repositories.solar_ai_chat.postgres_history_repository import PostgresChatHistoryRepository
from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat import (
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    CreateSessionRequest,
    ForkSessionRequest,
    IngestDocumentRequest,
    IngestDocumentResponse,
    RagStatsResponse,
    SolarChatRequest,
    SolarChatResponse,
    UpdateSessionTitleRequest,
)
from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient
from app.services.solar_ai_chat.gemini_client import GeminiModelRouter
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.intent_service import VietnameseIntentService
from app.services.solar_ai_chat.rag_ingestion_service import RagIngestionService

router = APIRouter(prefix="/solar-ai-chat", tags=["Solar AI Chat"])


def _resolve_user_chat_role(current_user: AuthUser) -> ChatRole:
    try:
        if current_user.role_id == "analyst":
            return ChatRole.DATA_ANALYST
        return ChatRole(current_user.role_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role is not allowed to access Solar AI Chat.",
        ) from exc


# ------------------------------------------------------------------
# Singleton dependency factories  (H1: cached, not recreated per request)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_history_repository() -> ChatHistoryRepository | PostgresChatHistoryRepository:
    settings = get_solar_chat_settings()
    backend = settings.history_backend.strip().lower()

    if backend == "postgres":
        return PostgresChatHistoryRepository(session_factory=SessionLocal)

    storage_dir = settings.resolved_data_root.parent / "chat_history"
    return ChatHistoryRepository(storage_dir=storage_dir)


@lru_cache(maxsize=1)
def _get_vector_repository() -> VectorRepository | None:
    settings = get_solar_chat_settings()
    if not settings.pg_host:
        return None
    return VectorRepository(settings=settings)


@lru_cache(maxsize=1)
def _get_embedding_client() -> GeminiEmbeddingClient | None:
    settings = get_solar_chat_settings()
    if not settings.gemini_api_key:
        return None
    return GeminiEmbeddingClient(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


@lru_cache(maxsize=1)
def get_solar_ai_chat_service() -> SolarAIChatService:
    settings = get_solar_chat_settings()

    model_router: GeminiModelRouter | None = None
    if settings.gemini_api_key:
        model_router = GeminiModelRouter(settings=settings)

    embedding_client = _get_embedding_client()
    intent_svc = VietnameseIntentService(embedding_client=embedding_client)
    intent_svc.initialize_semantic_router()

    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=intent_svc,
        model_router=model_router,
        history_repository=_get_history_repository(),
        vector_repo=_get_vector_repository(),
        embedding_client=embedding_client,
    )


# ------------------------------------------------------------------
# Chat query
# ------------------------------------------------------------------

@router.get("/topics")
def get_solar_ai_chat_topics(
    _: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
) -> dict[str, str]:
    return {
        "module": "solar_ai_chat",
        "message": "Solar AI Chat topics endpoint is ready.",
    }


@router.post("/query", response_model=SolarChatResponse)
def query_solar_ai_chat(
    request: SolarChatRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    service: SolarAIChatService = Depends(get_solar_ai_chat_service),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> SolarChatResponse:
    effective_role = _resolve_user_chat_role(current_user)
    if request.session_id:
        owned_session = history.get_session(
            session_id=request.session_id,
            owner_user_id=str(current_user.id),
        )
        if owned_session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    scoped_request = request.model_copy(update={"role": effective_role})
    try:
        return service.handle_query(scoped_request)
    except PermissionError as permission_error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(permission_error),
        ) from permission_error
    except ValueError as value_error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(value_error),
        ) from value_error
    except Exception as unexpected_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Solar AI Chat is temporarily unavailable. Please retry shortly.",
        ) from unexpected_error


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------

@router.post("/sessions", response_model=ChatSessionSummary, status_code=201)
def create_session(
    request: CreateSessionRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    effective_role = _resolve_user_chat_role(current_user)
    return history.create_session(
        role=effective_role,
        title=request.title,
        owner_user_id=str(current_user.id),
    )


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> list[ChatSessionSummary]:
    return history.list_sessions(owner_user_id=str(current_user.id))


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(
    session_id: str,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionDetail:
    session = history.get_session(session_id=session_id, owner_user_id=str(current_user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> None:
    if not history.delete_session(session_id=session_id, owner_user_id=str(current_user.id)):
        raise HTTPException(status_code=404, detail="Session not found.")


@router.patch("/sessions/{session_id}/title", response_model=ChatSessionSummary)
def update_session_title(
    session_id: str,
    request: UpdateSessionTitleRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    updated = history.update_session_title(
        session_id=session_id,
        title=request.title,
        owner_user_id=str(current_user.id),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return updated


@router.post("/sessions/{session_id}/rename", response_model=ChatSessionSummary)
def rename_session(
    session_id: str,
    request: UpdateSessionTitleRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    updated = history.update_session_title(
        session_id=session_id,
        title=request.title,
        owner_user_id=str(current_user.id),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return updated


@router.post(
    "/sessions/{session_id}/fork",
    response_model=ChatSessionSummary,
    status_code=201,
)
def fork_session(
    session_id: str,
    request: ForkSessionRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: ChatHistoryRepository | PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    effective_role = _resolve_user_chat_role(current_user)
    result = history.fork_session(
        source_session_id=session_id,
        new_title=request.title,
        new_role=effective_role,
        owner_user_id=str(current_user.id),
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source session not found.")
    return result


# ------------------------------------------------------------------
# RAG document ingestion (Admin only)
# ------------------------------------------------------------------

@router.post(
    "/documents/ingest",
    response_model=IngestDocumentResponse,
    status_code=201,
)
def ingest_document(
    request: IngestDocumentRequest,
    _: AuthUser = Depends(require_role(["admin"])),
) -> IngestDocumentResponse:
    settings = get_solar_chat_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Embedding API key is not configured.",
        )

    vector_repo = _get_vector_repository()
    embedding_client = _get_embedding_client()
    if not vector_repo or not embedding_client:
        raise HTTPException(
            status_code=503,
            detail="RAG infrastructure is not available.",
        )

    # M3: Restrict ingestion path to within the configured data root
    data_root = settings.resolved_data_root.parent
    file_path = Path(request.file_path).resolve()
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )
    try:
        file_path.relative_to(data_root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="File path must be within the configured data directory.",
        )

    ingestion_service = RagIngestionService(
        vector_repo=vector_repo,
        embedding_client=embedding_client,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    chunks_count = ingestion_service.ingest_document(
        file_path=file_path,
        doc_type=request.doc_type,
    )
    return IngestDocumentResponse(
        source_file=file_path.name,
        doc_type=request.doc_type,
        chunks_ingested=chunks_count,
    )


@router.get("/documents/stats", response_model=RagStatsResponse)
def get_document_stats(
    _: AuthUser = Depends(require_role(["admin"])),
    vector_repo: VectorRepository | None = Depends(_get_vector_repository),
) -> RagStatsResponse:
    if not vector_repo:
        return RagStatsResponse(total_chunks=0, by_doc_type={})
    stats = vector_repo.count_chunks()
    return RagStatsResponse(
        total_chunks=stats["total_chunks"],
        by_doc_type=stats["by_doc_type"],
    )


@router.delete("/documents/{source_file}", status_code=204)
def delete_document(
    source_file: str,
    _: AuthUser = Depends(require_role(["admin"])),
    vector_repo: VectorRepository | None = Depends(_get_vector_repository),
) -> None:
    if not vector_repo:
        raise HTTPException(
            status_code=503,
            detail="RAG infrastructure is not available.",
        )
    deleted = vector_repo.delete_by_source(source_file)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for '{source_file}'.",
        )
