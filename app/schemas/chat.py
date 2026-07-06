from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = None
    # Optional: provided by the SSE stream endpoint to persist history
    session_id: Optional[int] = None
    user_id: Optional[int] = None


class Source(BaseModel):
    content: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    debug: Optional[Dict[str, Any]] = None


class MessageFeedbackRequest(BaseModel):
    feedback: int  # 1 for thumbs up, -1 for thumbs down, 0 to clear
