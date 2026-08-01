"""Shared lifecycle helpers for persisted chat turns."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, ChatSession
from app.services.conversation_context import CONVERSATION_RESOLVER_VERSION, normalize_turn_context


def create_pending_user_turn(
    db: Session,
    *,
    user_id: int,
    session_id: int,
    content: str,
) -> ChatMessage | None:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if session is None:
        return None

    initial_context = {
        "version": CONVERSATION_RESOLVER_VERSION,
        "relation": "new_topic",
        "is_follow_up": False,
        "confidence": 0.0,
        "topic_id": str(uuid4()),
        "referenced_message_id": None,
        "intent": None,
        "standalone_query": content.strip()[:600],
        "source": "pending",
        "reason": "Turn is waiting for conversation resolution.",
        "status": "pending",
    }
    message = ChatMessage(
        session_id=session_id,
        role="user",
        content=content,
        turn_context=initial_context,
    )
    db.add(message)
    if session.title == "New chat":
        session.title = content[:80]
    session.updated_at = func.now()
    db.commit()
    db.refresh(message)
    return message


def update_user_turn_routing(
    db: Session,
    *,
    user_id: int,
    message_id: int,
    conversation: dict[str, Any] | None,
) -> bool:
    message = _owned_user_message(db, user_id=user_id, message_id=message_id)
    if message is None:
        return False
    message.turn_context = _context_with_status(
        conversation or message.turn_context,
        "routing_resolved",
    )
    db.commit()
    return True


def finalize_chat_turn(
    db: Session,
    *,
    user_id: int,
    user_message_id: int,
    assistant_message: str,
    sources: list[dict[str, Any]] | None,
    conversation: dict[str, Any] | None,
) -> ChatMessage | None:
    user_message = _owned_user_message(db, user_id=user_id, message_id=user_message_id)
    if user_message is None:
        return None

    resolved = conversation or user_message.turn_context or {}
    status = (
        "clarification_requested"
        if resolved.get("status") == "clarification_requested" or resolved.get("intent") == "CLARIFY"
        else "completed"
    )
    user_message.turn_context = _context_with_status(resolved, status)
    assistant_context = _context_with_status(resolved, status)

    assistant = ChatMessage(
        session_id=user_message.session_id,
        role="assistant",
        content=assistant_message,
        sources=sources or [],
        turn_context=assistant_context,
    )
    db.add(assistant)
    session = db.query(ChatSession).filter(ChatSession.id == user_message.session_id).first()
    if session is not None:
        session.updated_at = func.now()
    db.commit()
    db.refresh(assistant)
    return assistant


def mark_user_turn_failed(
    db: Session,
    *,
    user_id: int,
    message_id: int,
    conversation: dict[str, Any] | None = None,
) -> None:
    message = _owned_user_message(db, user_id=user_id, message_id=message_id)
    if message is None:
        return
    message.turn_context = _context_with_status(
        conversation or message.turn_context,
        "failed",
    )
    db.commit()


def _owned_user_message(db: Session, *, user_id: int, message_id: int) -> ChatMessage | None:
    return (
        db.query(ChatMessage)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .filter(
            ChatMessage.id == message_id,
            ChatMessage.role == "user",
            ChatSession.user_id == user_id,
        )
        .first()
    )


def _context_with_status(value: Any, status: str) -> dict[str, Any]:
    normalized = normalize_turn_context(value)
    if not normalized and isinstance(value, dict):
        normalized = dict(value)
    normalized["version"] = CONVERSATION_RESOLVER_VERSION
    normalized["status"] = status
    return normalized
