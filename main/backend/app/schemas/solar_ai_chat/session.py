from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.chat import SourceMetadata


class ChatMessage(BaseModel):
    id: str
    session_id: str
    sender: str = Field(description="Message sender: 'user' or 'assistant'.")
    content: str
    timestamp: datetime
    topic: ChatTopic | None = None
    sources: list[SourceMetadata] | None = None


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    role: ChatRole
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatSessionDetail(BaseModel):
    session_id: str
    title: str
    role: ChatRole
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage]


class CreateSessionRequest(BaseModel):
    role: ChatRole
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ForkSessionRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    role: ChatRole | None = None


class UpdateSessionTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
