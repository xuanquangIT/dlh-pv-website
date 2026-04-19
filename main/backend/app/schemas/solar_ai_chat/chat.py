from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.visualization import (
    ChartPayload,
    DataTablePayload,
    KpiCardsPayload,
)


class SourceMetadata(BaseModel):
    layer: str = Field(description="Data layer name. Accepted values: Silver or Gold.")
    dataset: str = Field(description="Dataset name used to answer the query.")
    data_source: str = Field(default="databricks", description="Backend used to retrieve data. Value: databricks.")
    url: str | None = Field(default=None, description="Source URL for web search results.")


ToolMode = Literal["auto", "none", "selected"]


class SolarChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    role: ChatRole | None = None
    session_id: str | None = Field(default=None, max_length=100)
    tool_mode: ToolMode = Field(
        default="auto",
        description=(
            "Tool selection policy. 'auto' = model chooses freely; 'none' = skip "
            "tool calling and answer directly from LLM knowledge; 'selected' = "
            "restrict the tool palette to `allowed_tools`."
        ),
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional explicit tool whitelist (used when tool_mode='selected').",
    )
    tool_hints: list[str] | None = Field(
        default=None,
        description=(
            "Soft hints from the UI indicating which capabilities the user explicitly "
            "picked. Accepted values: 'web_search', 'visualize'. Hints do not force "
            "the agent; they tilt it toward using that capability when relevant."
        ),
    )


class ThinkingStep(BaseModel):
    step: str
    detail: str
    status: Literal["info", "warning", "success"] = "info"


class ThinkingTrace(BaseModel):
    summary: str
    steps: list[ThinkingStep] = Field(default_factory=list)
    trace_id: str | None = None


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
    thinking_trace: ThinkingTrace | None = None
    ui_features: dict[str, bool] = Field(default_factory=dict)
    data_table: DataTablePayload | None = None
    chart: ChartPayload | None = None
    kpi_cards: KpiCardsPayload | None = None
