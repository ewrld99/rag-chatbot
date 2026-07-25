import logging
import time
from collections import defaultdict
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from typing import Dict, List

from app.api.deps import get_rag_pipeline, normalize_guest_history
from app.db.models import ChatMessage, ChatSession
from app.db.session import SessionLocal
from app.services.generation_resilience import GenerationUnavailableError
from app.services.model_router import InvalidModelPreference
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

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
    sources: List[Dict[str, object]] | None = None,
) -> bool:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )

    if not session:
        return False

    db.add(ChatMessage(session_id=session_id, role="user", content=user_message))
    db.add(
        ChatMessage(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            sources=sources or [],
        )
    )

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
            model_preference = str(payload.get("model_preference", "auto"))

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
            response_sources = []
            response_model = {
                "requested_model": model_preference,
                "selected_model": None,
                "fallback_used": False,
            }
            chat_history = normalize_guest_history(payload.get("history"))

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

            try:
                async for event in rag_pipeline.stream_events(
                    message,
                    chat_history=chat_history,
                    user_profile=user_profile,
                    model_preference=model_preference,
                ):
                    if event["type"] == "sources":
                        response_sources = event["sources"]
                        continue
                    if event["type"] == "model":
                        response_model = {
                            "requested_model": event["requested_model"],
                            "selected_model": event["selected_model"],
                            "fallback_used": event["fallback_used"],
                        }
                        continue
                    token = str(event["token"])
                    assistant_response += token
                    await websocket.send_json({
                        "type": "stream",
                        "token": token,
                        "user_id": user_id,
                    })
            except GenerationUnavailableError as exc:
                await websocket.send_json({
                    "type": "error",
                    **exc.public_payload(sources=response_sources or exc.sources),
                })
                continue
            except InvalidModelPreference as exc:
                await websocket.send_json({
                    "type": "error",
                    "code": "MODEL_NOT_ALLOWED",
                    "message": str(exc),
                    "sources": [],
                })
                continue
            except Exception:
                logger.exception("Unexpected WebSocket chat generation failure")
                await websocket.send_json({
                    "type": "error",
                    "code": "CHAT_STREAM_FAILED",
                    "message": "Unable to complete the answer right now. Please try again.",
                    "sources": response_sources,
                })
                continue

            if session_id:
                db = SessionLocal()
                try:
                    saved = save_chat_exchange(
                        db,
                        int(user_id),
                        int(session_id),
                        message,
                        assistant_response,
                        response_sources,
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

            await websocket.send_json({
                "type": "done",
                "user_id": user_id,
                "sources": response_sources,
                **response_model,
            })
    except WebSocketDisconnect:
        return
