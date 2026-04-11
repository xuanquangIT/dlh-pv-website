from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic


class SourceMetadata(BaseModel):
    layer: str = Field(description="Data layer name. Accepted values: Silver or Gold.")
    dataset: str = Field(description="Dataset name used to answer the query.")
    data_source: str = Field(default="databricks", description="Backend used to retrieve data. Value: databricks.")


class SolarChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    role: ChatRole | None = None
    session_id: str | None = Field(default=None, max_length=100)


class SolarChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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
