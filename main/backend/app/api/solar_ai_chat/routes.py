from pathlib import Path
from functools import lru_cache
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import require_role
from app.db.database import AuthUser
from app.core.settings import get_solar_chat_settings
from app.repositories.solar_ai_chat.postgres_history_repository import PostgresChatHistoryRepository
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.repositories.solar_ai_chat.base_repository import DatabricksDataUnavailableError
from app.repositories.solar_ai_chat.tool_usage_repository import ToolUsageRepository
from app.repositories.solar_ai_chat.vector_repository import VectorRepository
from app.schemas.solar_ai_chat import (
    ChatRole,
    ChatSessionDetail,
    ChatSessionSummary,
    CreateSessionRequest,
    ForkSessionRequest,
    IngestDocumentRequest,
    IngestDocumentResponse,
    LLMProfileList,
    LLMProfileSummary,
    RagStatsResponse,
    SolarChatRequest,
    SolarChatResponse,
    UpdateSessionTitleRequest,
)
from app.services.solar_ai_chat.embedding_client import GeminiEmbeddingClient
from app.services.solar_ai_chat.llm_client import LLMModelRouter, ModelUnavailableError
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.intent_service import VietnameseIntentService
from app.services.solar_ai_chat.model_profile_service import (
    PICKER_AUTH_ROLES,
    get_default_profile_id,
    list_profiles_for_role,
    resolve_profile,
    settings_with_profile_override,
)
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
def _get_history_repository() -> PostgresChatHistoryRepository:
    return PostgresChatHistoryRepository()


@lru_cache(maxsize=1)
def _get_vector_repository() -> VectorRepository | None:
    settings = get_solar_chat_settings()
    if not settings.pg_host:
        return None
    # RAG (search_documents) is opt-in: exposing it changes the agent's tool
    # palette and can drift synthesis on queries that have nothing to do with
    # documents. Set SOLAR_CHAT_RAG_ENABLED=1 when docs are actually ingested.
    import os
    if os.environ.get("SOLAR_CHAT_RAG_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    return VectorRepository(settings=settings)


@lru_cache(maxsize=1)
def _get_embedding_client() -> GeminiEmbeddingClient | None:
    settings = get_solar_chat_settings()
    keys = settings.embedding_api_keys
    if not keys:
        return None
    return GeminiEmbeddingClient(
        api_keys=keys,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


@lru_cache(maxsize=1)
def _get_tool_usage_repository() -> ToolUsageRepository:
    """Singleton telemetry repository. Safe to construct even when the DB is
    offline; each call is internally try/except-guarded."""
    return ToolUsageRepository()


@lru_cache(maxsize=1)
def _get_shared_intent_service() -> VietnameseIntentService:
    """Intent service is expensive to bootstrap (embedding API calls for the
    semantic router). Share one instance across default + per-profile services."""
    settings = get_solar_chat_settings()
    intent_svc = VietnameseIntentService(
        embedding_client=_get_embedding_client(),
        semantic_enabled=settings.intent_semantic_enabled,
        semantic_min_confidence=settings.intent_semantic_min_confidence,
        semantic_keyword_score_threshold=settings.intent_keyword_fastpath_score,
    )
    intent_svc.initialize_semantic_router()
    return intent_svc


def _build_chat_service(settings, model_router: LLMModelRouter | None) -> SolarAIChatService:
    """Construct a SolarAIChatService with a specific router + settings copy.

    Used both for the cached default service and for per-request profile
    overrides. All other dependencies (intent, repositories, embeddings) are
    shared singletons so the override path stays cheap."""
    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=_get_shared_intent_service(),
        model_router=model_router,
        history_repository=_get_history_repository(),
        vector_repo=_get_vector_repository(),
        embedding_client=_get_embedding_client(),
        tool_usage_logger=_get_tool_usage_repository(),
        planner_enabled=settings.planner_enabled,
        orchestrator_enabled=settings.orchestrator_enabled,
        verifier_enabled=settings.verifier_enabled,
        hybrid_retrieval_enabled=settings.hybrid_retrieval_enabled,
        max_tool_steps=settings.max_tool_steps,
        planner_max_output_tokens=settings.llm_planner_max_output_tokens,
        synthesis_max_output_tokens=settings.llm_synthesis_max_output_tokens,
        verifier_max_output_tokens=settings.llm_verifier_max_output_tokens,
        deep_planner_enabled=settings.planner_enabled,
    )


@lru_cache(maxsize=1)
def get_solar_ai_chat_service() -> SolarAIChatService:
    """Build the cached server-default chat service.

    Priority order for the default LLM connection:
      1. Profile flagged ``SOLAR_CHAT_PROFILE_<N>_DEFAULT=true`` —
         lets operators pick any provider as default without duplicating
         credentials in the legacy ``SOLAR_CHAT_LLM_*`` keys.
      2. Legacy ``SOLAR_CHAT_LLM_*`` env vars (still honoured for
         backwards compatibility).
      3. No router (chat falls back to scope-guard / direct synthesis).
    """
    base_settings = get_solar_chat_settings()
    default_profile_id = get_default_profile_id()
    default_profile = (
        resolve_profile(default_profile_id, "admin")
        if default_profile_id else None
    )
    if default_profile is not None:
        # Use the default profile's connection (provider, base_url, key,
        # primary/fallback model) — this becomes the server-wide default
        # for every request that doesn't carry a per-request override.
        effective_settings = settings_with_profile_override(
            base_settings, default_profile, model_name=None,
        )
    else:
        effective_settings = base_settings
    model_router: LLMModelRouter | None = None
    if effective_settings.llm_api_key or effective_settings.llm_base_url:
        model_router = LLMModelRouter(settings=effective_settings)
    return _build_chat_service(effective_settings, model_router)


def _resolve_chat_service_for_request(
    request: SolarChatRequest, current_user: AuthUser
) -> SolarAIChatService:
    """Pick the cached default service, or build a per-request override
    service when the caller is admin/ml_engineer and selected a valid
    (profile, model) pair.

    Unknown profile IDs raise 400 so the UI shows a clear error rather
    than silently degrading to default. An unknown model name within a
    valid profile silently falls back to the profile's primary model."""
    profile_id = (request.model_profile_id or "").strip()
    if not profile_id:
        return get_solar_ai_chat_service()

    if current_user.role_id not in PICKER_AUTH_ROLES:
        # Silent fallback — non-picker roles never see the dropdown anyway.
        return get_solar_ai_chat_service()

    profile = resolve_profile(profile_id, current_user.role_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown or unavailable LLM profile: '{profile_id}'.",
        )

    model_name = (request.model_name or "").strip() or None
    overridden_settings = settings_with_profile_override(
        get_solar_chat_settings(), profile, model_name=model_name,
    )
    router = LLMModelRouter(settings=overridden_settings)
    return _build_chat_service(overridden_settings, router)


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


@router.get("/llm-profiles", response_model=LLMProfileList)
def list_llm_profiles(
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
) -> LLMProfileList:
    """Return the LLM provider profiles the current user may pick from.

    Each profile bundles a single provider connection with its list of
    selectable models. Non-picker roles (data_engineer, analyst-mapped) get
    an empty list so the frontend hides the dropdown entirely."""
    profiles = list_profiles_for_role(current_user.role_id)
    default_id = get_default_profile_id()
    default_model_name: str | None = None
    for p in profiles:
        if p.id == default_id:
            default_model_name = p.primary_model
            break
    return LLMProfileList(
        profiles=[
            LLMProfileSummary(
                id=p.id,
                label=p.label,
                provider=p.provider,
                primary_model=p.primary_model,
                models=list(p.models),
                fallback_model=p.fallback_model,
                is_default=(p.id == default_id),
            )
            for p in profiles
        ],
        default_profile_id=default_id,
        default_model_name=default_model_name,
    )


@router.post("/query", response_model=SolarChatResponse)
def query_solar_ai_chat(
    request: SolarChatRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> SolarChatResponse:
    service = _resolve_chat_service_for_request(request, current_user)
    effective_role = _resolve_user_chat_role(current_user)
    if request.session_id:
        owned_session = history.session_exists(
            session_id=request.session_id,
            owner_user_id=str(current_user.id),
        )
        if not owned_session:
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
    except DatabricksDataUnavailableError as databricks_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Databricks is temporarily unavailable. "
                "No fallback data was used. Please retry shortly."
            ),
        ) from databricks_error
    except Exception as unexpected_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Solar AI Chat is temporarily unavailable. Please retry shortly.",
        ) from unexpected_error


@router.post("/query/benchmark")
def benchmark_solar_ai_chat_query(
    request: SolarChatRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> dict[str, object]:
    service = _resolve_chat_service_for_request(request, current_user)
    endpoint_started = time.perf_counter()
    effective_role = _resolve_user_chat_role(current_user)
    if request.session_id:
        owned_session = history.session_exists(
            session_id=request.session_id,
            owner_user_id=str(current_user.id),
        )
        if not owned_session:
            raise HTTPException(status_code=404, detail="Session not found.")

    scoped_request = request.model_copy(update={"role": effective_role})
    try:
        response = service.handle_query(scoped_request)
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
    except DatabricksDataUnavailableError as databricks_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Databricks is temporarily unavailable. "
                "No fallback data was used. Please retry shortly."
            ),
        ) from databricks_error
    except Exception as unexpected_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Solar AI Chat is temporarily unavailable. Please retry shortly.",
        ) from unexpected_error

    endpoint_elapsed_ms = int((time.perf_counter() - endpoint_started) * 1000)
    service_latency_ms = int(response.latency_ms)
    return {
        "benchmark_type": "full_pipeline",
        "server_elapsed_ms": endpoint_elapsed_ms,
        "service_latency_ms": service_latency_ms,
        "route_overhead_ms": max(0, endpoint_elapsed_ms - service_latency_ms),
        "response": response.model_dump(),
    }


@router.post("/query/benchmark/model-only")
def benchmark_solar_ai_chat_model_only(
    request: SolarChatRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
) -> dict[str, object]:
    endpoint_started = time.perf_counter()
    effective_role = _resolve_user_chat_role(current_user)

    settings = get_solar_chat_settings()
    if not settings.llm_api_key and not settings.llm_base_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM API key is not configured.",
        )

    model_router = LLMModelRouter(settings=settings)
    prompt = (
        "You are Solar AI. Answer concisely and clearly in English (max 8 sentences).\n"
        f"User Role: {effective_role.value}\n"
        f"Question: {request.message}\n"
        "Do not call tools, do not retrieve additional data."
    )

    model_started = time.perf_counter()
    try:
        model_result = model_router.generate(prompt)
    except ModelUnavailableError as model_error:
        model_elapsed_ms = int((time.perf_counter() - model_started) * 1000)
        endpoint_elapsed_ms = int((time.perf_counter() - endpoint_started) * 1000)
        return {
            "benchmark_type": "model_only",
            "server_elapsed_ms": endpoint_elapsed_ms,
            "model_generation_ms": model_elapsed_ms,
            "route_overhead_ms": max(0, endpoint_elapsed_ms - model_elapsed_ms),
            "error": str(model_error),
            "response": None,
        }

    model_elapsed_ms = int((time.perf_counter() - model_started) * 1000)
    endpoint_elapsed_ms = int((time.perf_counter() - endpoint_started) * 1000)

    return {
        "benchmark_type": "model_only",
        "server_elapsed_ms": endpoint_elapsed_ms,
        "model_generation_ms": model_elapsed_ms,
        "route_overhead_ms": max(0, endpoint_elapsed_ms - model_elapsed_ms),
        "response": {
            "answer": model_result.text,
            "model_used": model_result.model_used,
            "fallback_used": model_result.fallback_used,
            "role": effective_role.value,
        },
    }


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------

@router.post("/sessions", response_model=ChatSessionSummary, status_code=201)
def create_session(
    request: CreateSessionRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    effective_role = _resolve_user_chat_role(current_user)
    return history.create_session(
        role=effective_role,
        title=request.title,
        owner_user_id=str(current_user.id),
    )


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    limit: int = 50,
    offset: int = 0,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> list[ChatSessionSummary]:
    return history.list_sessions(owner_user_id=str(current_user.id), limit=limit, offset=offset)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(
    session_id: str,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionDetail:
    session = history.get_session(session_id=session_id, owner_user_id=str(current_user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
) -> None:
    if not history.delete_session(session_id=session_id, owner_user_id=str(current_user.id)):
        raise HTTPException(status_code=404, detail="Session not found.")


@router.patch("/sessions/{session_id}/title", response_model=ChatSessionSummary)
def update_session_title(
    session_id: str,
    request: UpdateSessionTitleRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
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
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
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
    history: PostgresChatHistoryRepository = Depends(_get_history_repository),
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
    if not settings.embedding_api_keys:
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


# ------------------------------------------------------------------
# Admin: tool usage telemetry
# ------------------------------------------------------------------

@router.get("/admin/tool-stats")
def get_tool_usage_stats(
    days: int = 7,
    _: AuthUser = Depends(require_role(["admin"])),
    usage_repo: ToolUsageRepository = Depends(_get_tool_usage_repository),
) -> dict[str, object]:
    """Return aggregated chat_tool_usage rows for the last ``days`` days.
    Admin-only — the raw table contains per-user activity."""
    if days < 1 or days > 365:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`days` must be between 1 and 365.",
        )
    return usage_repo.get_stats(days=days)
