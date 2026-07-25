from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    title: str = "New chat"


class ChatSessionResponse(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionDetail(ChatSessionResponse):
    messages: List[ChatMessageResponse]
