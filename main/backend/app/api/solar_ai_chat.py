from fastapi import APIRouter

router = APIRouter(prefix="/solar-ai-chat", tags=["Solar AI Chat"])


@router.get("/topics")
def get_solar_ai_chat_topics() -> dict[str, str]:
    return {
        "module": "solar_ai_chat",
        "message": "Solar AI Chat API placeholder is ready.",
    }
