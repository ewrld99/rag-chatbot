from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Depends, Header, HTTPException
from app.core.config import settings
from sqlalchemy.orm import Session

from app.db.session import (
    SessionFactory,
    get_db,
    get_session_factory,
    session_scope,
)
from app.db.models import User, ChatMessage, ChatSession
from app.core.security import decode_access_token
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval_service import RetrievalService
from app.services.conversation_context import attach_guest_context


EAT = timezone(timedelta(hours=3), name="EAT")


@dataclass(frozen=True)
class CurrentUserIdentity:
    id: int
    username: str


def get_retrieval_service(
    session_factory: SessionFactory = Depends(get_session_factory),
):
    return RetrievalService(
        top_k=settings.HYBRID_TOP_K,
        session_factory=session_factory,
    )


def get_rag_pipeline(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    return RAGPipeline(retrieval_service)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def get_user_role(username: str) -> str:
    admin_usernames = {
        normalize_username(admin_username)
        for admin_username in settings.ADMIN_USERNAMES.split(",")
        if admin_username.strip()
    }
    return "admin" if normalize_username(username) in admin_usernames else "user"


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    user = get_optional_current_user(authorization=authorization, db=db)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def get_optional_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def get_optional_current_user_identity(
    authorization: Optional[str] = Header(None),
    session_factory: SessionFactory = Depends(get_session_factory),
) -> Optional[CurrentUserIdentity]:
    """Authenticate without retaining an ORM session through a streamed response."""
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token subject")

    with session_scope(session_factory) as db:
        row = db.query(User.id, User.username).filter(User.id == user_id).first()
        if row is None:
            raise HTTPException(status_code=401, detail="User no longer exists")
        return CurrentUserIdentity(id=int(row.id), username=str(row.username))


def get_current_user_identity(
    current_user: Optional[CurrentUserIdentity] = Depends(
        get_optional_current_user_identity
    ),
) -> CurrentUserIdentity:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


def require_admin_identity(
    current_user: CurrentUserIdentity = Depends(get_current_user_identity),
) -> str:
    if get_user_role(current_user.username) != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user.username


def require_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if get_user_role(current_user.username) != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_requested_user(user_id: int, current_user: User = Depends(get_current_user)) -> User:
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's resource")
    return current_user


# ---------------------------------------------------------------------------
# Fix #3: Single reusable user-profile dependency (replaces copy-paste code)
# ---------------------------------------------------------------------------
def build_user_profile(user: User) -> Optional[Dict[str, Any]]:
    """Convert a User ORM object into the profile dict used by the RAG pipeline."""
    year = None
    if user.admission_year:
        now = datetime.now(EAT)
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
) -> List[Dict[str, Any]]:
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
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "turn_context": m.turn_context or {},
        }
        for m in reversed(messages)
    ]


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
        content = truncate_history_content(raw_content, min(max_message_chars, remaining))
        if not content:
            continue

        normalized_reversed.append({"role": role, "content": content})
        total_chars += len(content)

    return list(reversed(normalized_reversed))


def truncate_history_content(content: str, max_chars: int) -> str:
    """Trim history at a natural boundary when possible without exceeding caps."""
    if max_chars <= 0:
        return ""

    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped

    candidate = stripped[:max_chars].rstrip()
    if not candidate:
        return ""

    min_useful_length = max(20, int(max_chars * 0.5))
    sentence_boundary = max(
        candidate.rfind(". "),
        candidate.rfind("! "),
        candidate.rfind("? "),
        candidate.rfind("\n"),
    )
    if sentence_boundary >= min_useful_length:
        return candidate[: sentence_boundary + 1].strip()

    word_boundary = max(candidate.rfind(" "), candidate.rfind("\t"))
    if word_boundary >= min_useful_length:
        return candidate[:word_boundary].rstrip(" ,;:")

    return candidate


def get_chat_history(
    user_id: Optional[int],
    session_id: Optional[int],
    client_history: object,
    db: Session,
    guest_conversation: dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Resolve trusted DB history for users or bounded client history for guests.

    A request is treated as a guest only when *both* user_id and session_id are
    absent. If either is present we route to get_db_history, which enforces
    ownership and returns [] when session_id is None — preventing a client from
    omitting session_id to silently receive guest-history treatment while
    passing a user_id.
    """
    if user_id is None and session_id is None:
        return attach_guest_context(
            normalize_guest_history(client_history),
            guest_conversation,
        )
    return get_db_history(user_id, session_id, db)
