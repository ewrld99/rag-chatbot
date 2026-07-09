import time
from collections import defaultdict
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from typing import Dict, List

from app.api.deps import get_rag_pipeline
from app.db.models import ChatMessage, ChatSession
from app.db.session import SessionLocal
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# ---------------------------------------------------------------------------
# Simple sliding-window rate limiter for WebSocket messages.
# slowapi only covers HTTP endpoints, so we track per-user_id timestamps here.
# ---------------------------------------------------------------------------
_WS_LIMIT = 20          # max messages per window
_WS_WINDOW = 60.0       # window size in seconds
_ws_timestamps: Dict[str, List[float]] = defaultdict(list)


def _ws_is_allowed(user_id: str) -> bool:
    """Return True if the user is within the rate limit, False otherwise."""
    now = time.monotonic()
    bucket = _ws_timestamps[user_id]
    # Evict timestamps outside the current window
    _ws_timestamps[user_id] = [t for t in bucket if now - t < _WS_WINDOW]
    if len(_ws_timestamps[user_id]) >= _WS_LIMIT:
        return False
    _ws_timestamps[user_id].append(now)
    return True


def save_chat_exchange(
    db: Session,
    user_id: int,
    session_id: int,
    user_message: str,
    assistant_message: str,
) -> bool:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )

    if not session:
        return False

    db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
    db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_message))

    if session.title == "New chat":
        session.title = user_message[:80]

    session.updated_at = func.now()

    db.commit()
    return True


def load_chat_history(
    db: Session,
    user_id: int,
    session_id: int,
    limit: int = 12,
) -> List[Dict[str, str]]:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )

    if not session:
        return []

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )

    return [
        {"role": message.role, "content": message.content}
        for message in reversed(messages)
    ]


def normalize_payload_history(history: object, limit: int = 12) -> List[Dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized = []

    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = str(item.get("content", "")).strip()

        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})

    return normalized


@router.websocket("/ws/chat/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            message = payload.get("message", "").strip()
            session_id = payload.get("session_id")

            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Message cannot be empty",
                })
                continue

            # ── Rate limit check ────────────────────────────────────────────
            if not _ws_is_allowed(str(user_id)):
                await websocket.send_json({
                    "type": "error",
                    "message": f"Rate limit exceeded: max {_WS_LIMIT} messages per {int(_WS_WINDOW)}s. Please slow down.",
                })
                continue

            assistant_response = ""
            chat_history = normalize_payload_history(payload.get("history"))

            user_profile = None
            if session_id or user_id:
                db = SessionLocal()
                try:
                    from app.db.models import User
                    from datetime import datetime
                    if session_id:
                        chat_history = load_chat_history(db, int(user_id), int(session_id))
                    
                    user = db.query(User).filter(User.id == int(user_id)).first()
                    if user:
                        year = None
                        if user.admission_year:
                            now = datetime.now()
                            current_academic_year_start = now.year if now.month >= 9 else now.year - 1
                            year = max(1, (current_academic_year_start - user.admission_year) + 1)
                        
                        user_profile = {
                            "registration_number": user.registration_number,
                            "programme": user.programme,
                            "campus": user.campus,
                            "year_of_study": year,
                        }
                except (TypeError, ValueError):
                    if session_id:
                        chat_history = []
                finally:
                    db.close()

            async for token in rag_pipeline.stream(message, chat_history=chat_history, user_profile=user_profile):
                assistant_response += token
                await websocket.send_json({
                    "type": "stream",
                    "token": token,
                    "user_id": user_id,
                })

            if session_id:
                db = SessionLocal()
                try:
                    saved = save_chat_exchange(
                        db,
                        int(user_id),
                        int(session_id),
                        message,
                        assistant_response,
                    )
                except (TypeError, ValueError):
                    saved = False
                finally:
                    db.close()

                if not saved:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unable to save chat history",
                    })
                    continue

            await websocket.send_json({"type": "done", "user_id": user_id})
    except WebSocketDisconnect:
        return
