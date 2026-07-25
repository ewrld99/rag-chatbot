from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional


ModelPreference = Literal[
    "auto",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = None
    # Optional: provided by the SSE stream endpoint to persist history
    session_id: Optional[int] = None
    user_id: Optional[int] = None
    model_preference: ModelPreference = "auto"


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


class MessageFeedbackRequest(BaseModel):
    feedback: int  # 1 for thumbs up, -1 for thumbs down, 0 to clear
