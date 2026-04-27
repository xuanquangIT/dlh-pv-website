"""Solar AI Chat — public re-exports.

Phase 4 cleanup: dropped v1-only exports (VietnameseIntentService, ToolExecutor,
DeepPlanner, ExtremeMetricQuery, normalize_vietnamese_text). The v2 engine
in `solar_ai_chat/v2/` is the only runtime path.

Phase 4.5 cleanup: dropped RAG / embedding stack (GeminiEmbeddingClient,
RagIngestionService, EmbeddingUnavailableError) — never wired into the v2
chat path; manual ingest endpoints had no live consumers.
"""
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.llm_client import (
    FunctionCallRequest,
    GeminiGenerationResult,
    GeminiModelRouter,
    GeminiToolResult,
    LLMGenerationResult,
    LLMModelRouter,
    LLMToolResult,
    ModelUnavailableError,
    ToolCallRequest,
)
from app.services.solar_ai_chat.permissions import (
    ROLE_TOOL_PERMISSIONS,
    ROLE_TOPIC_PERMISSIONS,
)

__all__ = [
    "FunctionCallRequest",
    "GeminiGenerationResult",
    "GeminiModelRouter",
    "GeminiToolResult",
    "LLMGenerationResult",
    "LLMModelRouter",
    "LLMToolResult",
    "ModelUnavailableError",
    "ROLE_TOOL_PERMISSIONS",
    "ROLE_TOPIC_PERMISSIONS",
    "SolarAIChatService",
    "ToolCallRequest",
]
