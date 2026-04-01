from typing import Any

from pydantic import BaseModel, Field

from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic


class SourceMetadata(BaseModel):
    layer: str = Field(description="Data layer name. Accepted values: Silver or Gold.")
    dataset: str = Field(description="Dataset name used to answer the query.")


class SolarChatRequest(BaseModel):
    message: str = Field(min_length=3, max_length=1000)
    role: ChatRole
    session_id: str | None = Field(default=None, max_length=100)


class SolarChatResponse(BaseModel):
    answer: str
    topic: ChatTopic
    role: ChatRole
    sources: list[SourceMetadata]
    key_metrics: dict[str, Any]
    model_used: str
    fallback_used: bool
    latency_ms: int
    intent_confidence: float
    warning_message: str | None = None
