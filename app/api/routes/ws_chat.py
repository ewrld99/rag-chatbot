import logging
import time
from collections import defaultdict
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Any, Dict, List

from app.api.deps import build_user_profile, get_rag_pipeline, normalize_guest_history
from app.core.security import decode_access_token
from app.db.models import ChatMessage, ChatSession, User
from app.db.session import SessionLocal
from app.services.generation_resilience import GenerationUnavailableError
from app.services.chat_persistence_service import (
    create_pending_user_turn,
    finalize_chat_turn,
    mark_user_turn_failed,
    update_user_turn_routing,
)
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


def load_chat_history(
    db: Session,
    user_id: int,
    session_id: int,
    limit: int = 12,
) -> List[Dict[str, Any]]:
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
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "turn_context": message.turn_context or {},
        }
        for message in reversed(messages)
    ]


def _authenticate_websocket_user(websocket: WebSocket, path_user_id: str) -> tuple[int, Dict[str, Any] | None] | None:
    token = websocket.query_params.get("token", "").strip()
    payload = decode_access_token(token) if token else None
    if not payload:
        return None

    try:
        token_user_id = int(payload.get("sub"))
        requested_user_id = int(path_user_id)
    except (TypeError, ValueError):
        return None
    if token_user_id != requested_user_id:
        return None

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == token_user_id).first()
        if not user:
            return None
        return token_user_id, build_user_profile(user)
    finally:
        db.close()


@router.websocket("/ws/chat/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    auth_result = _authenticate_websocket_user(websocket, user_id)
    if auth_result is None:
        await websocket.close(code=1008)
        return
    authenticated_user_id, authenticated_user_profile = auth_result

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
            response_conversation = {}
            pending_user_message_id = None
            response_model = {
                "requested_model": model_preference,
                "selected_model": None,
                "fallback_used": False,
            }
            chat_history = normalize_guest_history(payload.get("history"))

            user_profile = authenticated_user_profile
            if session_id:
                db = SessionLocal()
                try:
                    chat_history = load_chat_history(db, authenticated_user_id, int(session_id))
                    pending = create_pending_user_turn(
                        db,
                        user_id=authenticated_user_id,
                        session_id=int(session_id),
                        content=message,
                    )
                    pending_user_message_id = pending.id if pending is not None else None
                except (TypeError, ValueError):
                    chat_history = []
                finally:
                    db.close()
                if pending_user_message_id is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unable to use this chat session",
                    })
                    continue

            try:
                async for event in rag_pipeline.stream_events(
                    message,
                    chat_history=chat_history,
                    user_profile=user_profile,
                    model_preference=model_preference,
                ):
                    if event["type"] == "routing":
                        routing_intent = (event.get("routing") or {}).get("intent")
                        response_conversation = {
                            **(event.get("conversation") or {}),
                            "status": (
                                "clarification_requested"
                                if routing_intent == "CLARIFY"
                                else "completed"
                            ),
                        }
                        if pending_user_message_id is not None:
                            db = SessionLocal()
                            try:
                                update_user_turn_routing(
                                    db,
                                    user_id=authenticated_user_id,
                                    message_id=pending_user_message_id,
                                    conversation=response_conversation,
                                )
                            finally:
                                db.close()
                        continue
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
                    if event["type"] == "status":
                        await websocket.send_json({
                            "type": "status",
                            "message": event.get("message", "Working on the answer..."),
                            "user_id": user_id,
                        })
                        continue
                    if event["type"] == "replace":
                        assistant_response = str(event.get("answer") or "")
                        await websocket.send_json({
                            "type": "replace",
                            "answer": assistant_response,
                            "provisional": bool(event.get("provisional", False)),
                            "user_id": user_id,
                        })
                        continue
                    token = str(event["token"])
                    assistant_response += token
                    await websocket.send_json({
                        "type": "stream",
                        "token": token,
                        "user_id": user_id,
                    })
            except GenerationUnavailableError as exc:
                if pending_user_message_id is not None:
                    db = SessionLocal()
                    try:
                        mark_user_turn_failed(
                            db,
                            user_id=authenticated_user_id,
                            message_id=pending_user_message_id,
                            conversation=response_conversation,
                        )
                    finally:
                        db.close()
                await websocket.send_json({
                    "type": "error",
                    **exc.public_payload(sources=response_sources or exc.sources),
                })
                continue
            except InvalidModelPreference as exc:
                if pending_user_message_id is not None:
                    db = SessionLocal()
                    try:
                        mark_user_turn_failed(
                            db,
                            user_id=authenticated_user_id,
                            message_id=pending_user_message_id,
                            conversation=response_conversation,
                        )
                    finally:
                        db.close()
                await websocket.send_json({
                    "type": "error",
                    "code": "MODEL_NOT_ALLOWED",
                    "message": str(exc),
                    "sources": [],
                })
                continue
            except Exception:
                logger.exception("Unexpected WebSocket chat generation failure")
                if pending_user_message_id is not None:
                    db = SessionLocal()
                    try:
                        mark_user_turn_failed(
                            db,
                            user_id=authenticated_user_id,
                            message_id=pending_user_message_id,
                            conversation=response_conversation,
                        )
                    finally:
                        db.close()
                await websocket.send_json({
                    "type": "error",
                    "code": "CHAT_STREAM_FAILED",
                    "message": "Unable to complete the answer right now. Please try again.",
                    "sources": response_sources,
                })
                continue

            if session_id and pending_user_message_id is not None:
                db = SessionLocal()
                try:
                    assistant = finalize_chat_turn(
                        db,
                        user_id=authenticated_user_id,
                        user_message_id=pending_user_message_id,
                        assistant_message=assistant_response,
                        sources=response_sources,
                        conversation=response_conversation,
                    )
                finally:
                    db.close()

                if assistant is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unable to save chat history",
                    })
                    continue

            await websocket.send_json({
                "type": "done",
                "user_id": user_id,
                "sources": response_sources,
                "conversation": response_conversation,
                **response_model,
            })
    except WebSocketDisconnect:
        return
