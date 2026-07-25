from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from app.core.config import settings
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, ChatMessage, ChatSession
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval_service import RetrievalService


def get_retrieval_service(db: Session = Depends(get_db)):
    return RetrievalService(db=db, top_k=settings.HYBRID_TOP_K)


def get_rag_pipeline(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    return RAGPipeline(retrieval_service)


# ---------------------------------------------------------------------------
# Fix #3: Single reusable user-profile dependency (replaces copy-paste code)
# ---------------------------------------------------------------------------
def build_user_profile(user: User) -> Optional[Dict[str, Any]]:
    """Convert a User ORM object into the profile dict used by the RAG pipeline."""
    year = None
    if user.admission_year:
        now = datetime.now()
        current_academic_year_start = now.year if now.month >= 9 else now.year - 1
        year = max(1, (current_academic_year_start - user.admission_year) + 1)

    return {
        "registration_number": user.registration_number,
        "programme": user.programme,
        "campus": user.campus,
        "year_of_study": year,
    }


def get_user_profile(user_id: Optional[int], db: Session) -> Optional[Dict[str, Any]]:
    """
    Fetch user profile from the database using the already-open request session.
    Returns None for guests (user_id is None).
    """
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    return build_user_profile(user) if user else None


# ---------------------------------------------------------------------------
# Fix #4: Load history from the DB — never trust client-supplied history
# ---------------------------------------------------------------------------
def get_db_history(
    user_id: Optional[int],
    session_id: Optional[int],
    db: Session,
    limit: int = 20,
) -> List[Dict[str, str]]:
    """
    Load the most recent `limit` messages for this session from the database.
    Falls back to an empty list for guests or missing sessions.

    Security: only returns messages where the session belongs to `user_id`,
    preventing any cross-user history injection.
    """
    if not user_id or not session_id:
        return []

    # Fix #5: Enforce ownership — session must belong to the declared user_id
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,   # ownership check
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=403,
            detail="Session does not belong to this user.",
        )

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    # Reverse so oldest-first order matches LLM expectations
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


def normalize_guest_history(
    history: object,
    limit: int = 12,
    max_message_chars: int = 2000,
    max_total_chars: int = 12000,
) -> List[Dict[str, str]]:
    """Sanitize the bounded in-memory history supplied by an anonymous client."""
    if not isinstance(history, list) or limit <= 0:
        return []

    normalized_reversed: List[Dict[str, str]] = []
    total_chars = 0

    for item in reversed(history):
        if len(normalized_reversed) >= limit or total_chars >= max_total_chars:
            break
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        raw_content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(raw_content, str):
            continue

        remaining = max_total_chars - total_chars
        content = raw_content.strip()[:min(max_message_chars, remaining)].strip()
        if not content:
            continue

        normalized_reversed.append({"role": role, "content": content})
        total_chars += len(content)

    return list(reversed(normalized_reversed))


def get_chat_history(
    user_id: Optional[int],
    session_id: Optional[int],
    client_history: object,
    db: Session,
) -> List[Dict[str, str]]:
    """Resolve trusted DB history for users or bounded client history for guests."""
    if user_id is None and session_id is None:
        return normalize_guest_history(client_history)
    return get_db_history(user_id, session_id, db)
