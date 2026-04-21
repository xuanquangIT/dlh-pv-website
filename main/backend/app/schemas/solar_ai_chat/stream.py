"""SSE streaming event schemas for Solar AI Chat.

Each UiEvent is a discriminated union of event types:
  - thinking_step: a single tool call being started (with step #, tool name)
  - tool_result:   a tool call completed (with key metric names, duration ms)
  - status_update: a free-form status text shown in the chat header
  - text_delta:    an incremental chunk of the final LLM answer
  - done:          the full SolarChatResponse payload; marks stream end
  - error:         an error that terminated the stream early
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ThinkingStepEvent(BaseModel):
    event: Literal["thinking_step"] = "thinking_step"
    step: int
    tool_name: str
    label: str                      # Human-readable label, e.g. "Fetching system overview"
    trace_id: str = ""


class ToolResultEvent(BaseModel):
    event: Literal["tool_result"] = "tool_result"
    step: int
    tool_name: str
    status: Literal["ok", "error", "denied", "skipped"]
    metric_keys: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    trace_id: str = ""


class StatusUpdateEvent(BaseModel):
    event: Literal["status_update"] = "status_update"
    text: str                       # e.g. "Analyzing intent…", "Synthesizing answer…"


class TextDeltaEvent(BaseModel):
    event: Literal["text_delta"] = "text_delta"
    delta: str                      # partial answer chunk


class DoneEvent(BaseModel):
    model_config = {"protected_namespaces": ()}
    event: Literal["done"] = "done"
    answer: str
    topic: str
    role: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    fallback_used: bool = False
    latency_ms: int = 0
    intent_confidence: float = 0.0
    warning_message: str | None = None
    thinking_trace: dict[str, Any] | None = None
    ui_features: dict[str, bool] = Field(default_factory=dict)
    trace_id: str = ""


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str
    code: str = "stream_error"


# Union type for type hints
UiEvent = (
    ThinkingStepEvent
    | ToolResultEvent
    | StatusUpdateEvent
    | TextDeltaEvent
    | DoneEvent
    | ErrorEvent
)


# Human-readable labels for tool names (shown in Task Tracker)
TOOL_DISPLAY_LABELS: dict[str, str] = {
    "get_system_overview":       "Fetching system overview",
    "get_energy_performance":    "Fetching energy performance",
    "get_facility_info":         "Fetching facility information",
    "get_pipeline_status":       "Checking pipeline status",
    "get_forecast_72h":          "Fetching 72-hour forecast",
    "get_data_quality_issues":   "Checking data quality",
    "get_ml_model_info":         "Fetching ML model info",
    "get_station_daily_report":  "Fetching daily station report",
    "get_station_hourly_report": "Fetching hourly station report",
    "web_lookup":                "Searching the web",
    "web_lookup_direct":         "Searching the web",
    "answer_directly":           "Preparing direct answer",
    "synthesize":                "Synthesizing answer",
}


def tool_label(tool_name: str) -> str:
    return TOOL_DISPLAY_LABELS.get(tool_name, f"Running {tool_name}")
