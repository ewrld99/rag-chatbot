from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

from app.services.model_catalog import validate_model_preference
from app.core.config import settings


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=settings.CHAT_MAX_MESSAGE_CHARS)
    history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=settings.CHAT_MAX_HISTORY_MESSAGES,
    )
    # Optional: provided by the SSE stream endpoint to persist history
    session_id: Optional[int] = None
    user_id: Optional[int] = None
    model_preference: str = "auto"
    conversation_token: Optional[str] = Field(default=None, max_length=4096)

    _validate_model_preference = field_validator("model_preference")(
        validate_model_preference
    )


class Source(BaseModel):
    document_id: Optional[str] = None
    name: str
    url: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    debug: Optional[Dict[str, Any]] = None
    requested_model: str = "auto"
    selected_model: Optional[str] = None
    fallback_used: bool = False
    conversation: Dict[str, Any] = Field(default_factory=dict)
    conversation_token: Optional[str] = None


class MessageFeedbackRequest(BaseModel):
    feedback: int  # 1 for thumbs up, -1 for thumbs down, 0 to clear
