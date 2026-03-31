from fastapi import APIRouter, Depends, HTTPException, status

from app.core.settings import get_solar_chat_settings
from app.repositories.solar_chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat import SolarChatRequest, SolarChatResponse
from app.services.gemini_client import GeminiModelRouter
from app.services.solar_ai_chat_service import SolarAIChatService
from app.services.solar_chat_intent_service import VietnameseIntentService

router = APIRouter(prefix="/solar-ai-chat", tags=["Solar AI Chat"])


@router.get("/topics")
def get_solar_ai_chat_topics() -> dict[str, str]:
    return {
        "module": "solar_ai_chat",
        "message": "Solar AI Chat topics endpoint is ready.",
    }


def get_solar_ai_chat_service() -> SolarAIChatService:
    settings = get_solar_chat_settings()

    model_router: GeminiModelRouter | None = None
    if settings.gemini_api_key:
        model_router = GeminiModelRouter(settings=settings)

    return SolarAIChatService(
        repository=SolarChatRepository(settings=settings),
        intent_service=VietnameseIntentService(),
        model_router=model_router,
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
