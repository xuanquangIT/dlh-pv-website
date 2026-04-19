from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.chat import (
    SourceMetadata,
    SolarChatRequest,
    SolarChatResponse,
    ThinkingStep,
    ThinkingTrace,
)
from app.schemas.solar_ai_chat.session import (
    ChatMessage,
    ChatSessionSummary,
    ChatSessionDetail,
    CreateSessionRequest,
    ForkSessionRequest,
    UpdateSessionTitleRequest,
)
from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS, TOOL_NAME_TO_TOPIC
from app.schemas.solar_ai_chat.agent import (
    PlannerAction,
    PlannerOutput,
    EvidenceItem,
    EvidenceStore,
    ToolResultEnvelope,
)
from app.schemas.solar_ai_chat.rag import (
    RagChunk,
    RetrievedChunk,
    RagSearchResult,
    IngestDocumentRequest,
    IngestDocumentResponse,
    RagStatsResponse,
)
from app.schemas.solar_ai_chat.ui_features import (
    UiFeature,
    ROLE_UI_FEATURES,
    ALL_FEATURES as UI_FEATURE_KEYS,
    resolve_ui_features,
)
from app.schemas.solar_ai_chat.stream import (
    ThinkingStepEvent,
    ToolResultEvent,
    StatusUpdateEvent,
    TextDeltaEvent,
    DoneEvent,
    ErrorEvent,
    UiEvent,
    tool_label,
)

__all__ = [
    "ChatRole",
    "ChatTopic",
    "SourceMetadata",
    "SolarChatRequest",
    "SolarChatResponse",
    "ThinkingStep",
    "ThinkingTrace",
    "ChatMessage",
    "ChatSessionSummary",
    "ChatSessionDetail",
    "CreateSessionRequest",
    "ForkSessionRequest",
    "UpdateSessionTitleRequest",
    "TOOL_DECLARATIONS",
    "TOOL_NAME_TO_TOPIC",
    "RagChunk",
    "RetrievedChunk",
    "RagSearchResult",
    "IngestDocumentRequest",
    "IngestDocumentResponse",
    "RagStatsResponse",
    "PlannerAction",
    "PlannerOutput",
    "EvidenceItem",
    "EvidenceStore",
    "ToolResultEnvelope",
    "ThinkingStepEvent",
    "ToolResultEvent",
    "StatusUpdateEvent",
    "TextDeltaEvent",
    "DoneEvent",
    "ErrorEvent",
    "UiEvent",
    "tool_label",
    "UiFeature",
    "ROLE_UI_FEATURES",
    "UI_FEATURE_KEYS",
    "resolve_ui_features",
]
