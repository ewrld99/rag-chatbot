import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.schemas.chat import ChatRequest, ChatResponse, MessageFeedbackRequest
from app.schemas.history import ChatSessionCreate, ChatSessionDetail, ChatSessionResponse
from app.services.rag_pipeline import RAGPipeline
from app.api.deps import get_rag_pipeline
from app.db.models import ChatMessage, ChatSession, User
from app.db.session import get_db, SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)


def get_user_or_404(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_session_or_404(user_id: int, session_id: int, db: Session) -> ChatSession:
    session = (
        db.query(ChatSession)
        .options(selectinload(ChatSession.messages))
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


@router.get("/users/{user_id}/sessions", response_model=list[ChatSessionResponse])
def list_chat_sessions(user_id: int, db: Session = Depends(get_db)):
    get_user_or_404(user_id, db)
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .all()
    )


@router.post("/users/{user_id}/sessions", response_model=ChatSessionResponse, status_code=201)
def create_chat_session(
    user_id: int,
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    get_user_or_404(user_id, db)
    title = payload.title.strip() or "New chat"
    session = ChatSession(user_id=user_id, title=title[:80])
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/users/{user_id}/sessions/{session_id}", response_model=ChatSessionDetail)
def read_chat_session(user_id: int, session_id: int, db: Session = Depends(get_db)):
    return get_session_or_404(user_id, session_id, db)


@router.delete("/users/{user_id}/sessions/{session_id}")
def delete_chat_session(user_id: int, session_id: int, db: Session = Depends(get_db)):
    session = get_session_or_404(user_id, session_id, db)
    db.delete(session)
    db.commit()
    return {"message": "Chat deleted successfully", "id": session_id}


# ---------------------------------------
# 1. Standard Chat Endpoint (IMPROVED)
# ---------------------------------------
@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    debug: bool = Query(False, description="Enable debug mode"),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Chat endpoint for RAG system:
    - Normal mode → clean response
    - Debug mode → includes retrieval insights
    """

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user_profile = None
    if request.user_id:
        from app.db.session import SessionLocal
        from app.db.models import User
        from datetime import datetime
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == request.user_id).first()
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

    try:
        # ✅ Debug Mode
        if debug:
            result = rag_pipeline.run_debug(request.message, chat_history=request.history, user_profile=user_profile)

            return {
                "response": result["answer"],
                "sources": [],
                "debug": result["debug"]
            }

        # ✅ Normal Mode
        result = rag_pipeline.run(request.message, chat_history=request.history, user_profile=user_profile)

        return ChatResponse(
            response=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "message": str(e)
            }
        )


# -----------------------------------------------------------------------
# 2. SSE Streaming Chat Endpoint
# -----------------------------------------------------------------------
@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Server-Sent Events (SSE) streaming endpoint.

    Emits newline-delimited JSON events over text/event-stream:

      data: {"type": "stream", "token": "hello"}\n\n
      data: {"type": "done"}\n\n
      data: {"type": "error", "message": "..."}\n\n

    If `user_id` and `session_id` are provided the full exchange is
    persisted to the database after the stream completes.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user_profile = None
    if request.user_id:
        from app.db.session import SessionLocal
        from app.db.models import User
        from datetime import datetime
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == request.user_id).first()
            if user:
                year = None
                if user.admission_year:
                    now = datetime.now()
                    # Academic year starts in September
                    current_academic_year_start = now.year if now.month >= 9 else now.year - 1
                    year = max(1, (current_academic_year_start - user.admission_year) + 1)
                
                user_profile = {
                    "registration_number": user.registration_number,
                    "programme": user.programme,
                    "campus": user.campus,
                    "year_of_study": year,
                }

    async def event_stream():
        full_response = ""
        assistant_message_id = None

        try:
            # ── Stream tokens from RAG pipeline ────────────────────────────
            async for token in rag_pipeline.stream(
                request.message,
                chat_history=request.history,
                user_profile=user_profile,
            ):
                full_response += token
                payload = json.dumps({"type": "stream", "token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        except Exception as exc:
            logger.exception("SSE stream error")
            err_payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {err_payload}\n\n"
            return

        # ── Persist to DB if caller supplied session context ────────────────
        if request.user_id and request.session_id and full_response:
            db: Session = SessionLocal()
            try:
                session = (
                    db.query(ChatSession)
                    .filter(
                        ChatSession.id == request.session_id,
                        ChatSession.user_id == request.user_id,
                    )
                    .first()
                )
                if session:
                    user_msg = ChatMessage(
                        session_id=request.session_id,
                        role="user",
                        content=request.message,
                    )
                    db.add(user_msg)
                    
                    asst_msg = ChatMessage(
                        session_id=request.session_id,
                        role="assistant",
                        content=full_response,
                    )
                    db.add(asst_msg)
                    
                    if session.title == "New chat":
                        session.title = request.message[:80]
                    session.updated_at = func.now()
                    db.commit()
                    db.refresh(asst_msg)
                    assistant_message_id = asst_msg.id
            except Exception as exc:
                logger.warning("SSE: failed to persist chat exchange: %s", exc)
                db.rollback()
            finally:
                db.close()

        # ── Signal completion ───────────────────────────────────────────────
        yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_message_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.post("/messages/{message_id}/feedback")
def submit_message_feedback(
    message_id: int,
    payload: MessageFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit thumbs up (1) or thumbs down (-1) feedback for a specific chat message.
    Pass 0 to clear feedback.
    """
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
        
    if payload.feedback not in [-1, 0, 1]:
        raise HTTPException(status_code=400, detail="Feedback must be -1, 0, or 1")
        
    message.feedback = payload.feedback if payload.feedback != 0 else None
    db.commit()
    
    return {"status": "success", "message_id": message_id, "feedback": message.feedback}


@router.get("/feedback/messages")
def list_feedback_messages(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    List messages that received feedback, for the admin dashboard.
    """
    messages = (
        db.query(ChatMessage)
        .options(selectinload(ChatMessage.session).selectinload(ChatSession.user))
        .filter(ChatMessage.feedback.isnot(None))
        .order_by(ChatMessage.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    result = []
    for msg in messages:
        result.append({
            "id": msg.id,
            "session_id": msg.session_id,
            "user_id": msg.session.user_id if msg.session else None,
            "username": msg.session.user.username if msg.session and msg.session.user else "Guest",
            "role": msg.role,
            "content": msg.content,
            "feedback": msg.feedback,
            "created_at": msg.created_at
        })
    return result
