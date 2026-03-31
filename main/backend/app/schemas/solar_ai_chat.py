from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    DATA_ENGINEER = "data_engineer"
    ML_ENGINEER = "ml_engineer"
    DATA_ANALYST = "data_analyst"
    VIEWER = "viewer"
    ADMIN = "admin"


class ChatTopic(str, Enum):
    SYSTEM_OVERVIEW = "system_overview"
    ENERGY_PERFORMANCE = "energy_performance"
    ML_MODEL = "ml_model"
    PIPELINE_STATUS = "pipeline_status"
    FORECAST_72H = "forecast_72h"
    DATA_QUALITY_ISSUES = "data_quality_issues"


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
