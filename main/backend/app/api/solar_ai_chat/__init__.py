from fastapi import APIRouter, Depends, HTTPException, status

from app.core.settings import get_solar_chat_settings
from app.repositories.solar_ai_chat.history_repository import ChatHistoryRepository
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat import (
    ChatSessionDetail,
    ChatSessionSummary,
    CreateSessionRequest,
    ForkSessionRequest,
    SolarChatRequest,
    SolarChatResponse,
)
from app.services.solar_ai_chat.gemini_client import GeminiModelRouter
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.intent_service import VietnameseIntentService

router = APIRouter(prefix="/solar-ai-chat", tags=["Solar AI Chat"])


@router.get("/topics")
def get_solar_ai_chat_topics() -> dict[str, str]:
    return {
        "module": "solar_ai_chat",
        "message": "Solar AI Chat topics endpoint is ready.",
    }


def _get_history_repository() -> ChatHistoryRepository:
    settings = get_solar_chat_settings()
    storage_dir = settings.resolved_data_root.parent / "chat_history"
    return ChatHistoryRepository(storage_dir=storage_dir)


def get_solar_ai_chat_service() -> SolarAIChatService:
    settings = get_solar_chat_settings()

    model_router: GeminiModelRouter | None = None
    if settings.gemini_api_key:
        model_router = GeminiModelRouter(settings=settings)

    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=VietnameseIntentService(),
        model_router=model_router,
        history_repository=_get_history_repository(),
    )


@router.post("/query", response_model=SolarChatResponse)
def query_solar_ai_chat(
    request: SolarChatRequest,
    service: SolarAIChatService = Depends(get_solar_ai_chat_service),
) -> SolarChatResponse:
    try:
        return service.handle_query(request)
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


@router.post("/sessions", response_model=ChatSessionSummary, status_code=201)
def create_session(
    request: CreateSessionRequest,
    history: ChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    return history.create_session(role=request.role, title=request.title)


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    history: ChatHistoryRepository = Depends(_get_history_repository),
) -> list[ChatSessionSummary]:
    return history.list_sessions()


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(
    session_id: str,
    history: ChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionDetail:
    session = history.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    history: ChatHistoryRepository = Depends(_get_history_repository),
) -> None:
    if not history.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


@router.post("/sessions/{session_id}/fork", response_model=ChatSessionSummary, status_code=201)
def fork_session(
    session_id: str,
    request: ForkSessionRequest,
    history: ChatHistoryRepository = Depends(_get_history_repository),
) -> ChatSessionSummary:
    result = history.fork_session(
        source_session_id=session_id,
        new_title=request.title,
        new_role=request.role,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Source session not found.")
    return result
