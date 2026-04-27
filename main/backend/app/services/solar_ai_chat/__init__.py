"""Solar AI Chat — public re-exports.

Phase 4 cleanup: dropped v1-only exports (VietnameseIntentService, ToolExecutor,
DeepPlanner, ExtremeMetricQuery, normalize_vietnamese_text). The v2 engine
in `solar_ai_chat/v2/` is the only runtime path.
"""
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.embedding_client import (
    EmbeddingUnavailableError,
    GeminiEmbeddingClient,
)
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
from app.services.solar_ai_chat.rag_ingestion_service import RagIngestionService

__all__ = [
    "EmbeddingUnavailableError",
    "FunctionCallRequest",
    "GeminiEmbeddingClient",
    "GeminiGenerationResult",
    "GeminiModelRouter",
    "GeminiToolResult",
    "LLMGenerationResult",
    "LLMModelRouter",
    "LLMToolResult",
    "ModelUnavailableError",
    "RagIngestionService",
    "ROLE_TOOL_PERMISSIONS",
    "ROLE_TOPIC_PERMISSIONS",
    "SolarAIChatService",
    "ToolCallRequest",
]
