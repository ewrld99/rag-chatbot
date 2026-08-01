import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload

from app.api.limiter import RateLimiter
from app.schemas.chat import ChatRequest, ChatResponse, MessageFeedbackRequest
from app.schemas.history import ChatSessionCreate, ChatSessionDetail, ChatSessionResponse
from app.services.rag_pipeline import RAGPipeline
from app.api.deps import (
    get_chat_history,
    get_optional_current_user,
    get_rag_pipeline,
    get_user_role,
    get_user_profile,
    require_admin_user,
    require_requested_user,
)
from app.db.models import ChatMessage, ChatSession, User
from app.db.session import get_db
from app.services.generation_resilience import GenerationUnavailableError
from app.services.chat_persistence_service import (
    create_pending_user_turn,
    finalize_chat_turn,
    mark_user_turn_failed,
    update_user_turn_routing,
)
from app.services.conversation_context import (
    create_guest_conversation_token,
    decode_guest_conversation_token,
)
from app.services.model_router import ModelRouter
from app.services.settings_service import SettingsService

router = APIRouter()
logger = logging.getLogger(__name__)


def _result_conversation(result: dict) -> dict:
    conversation = result.get("conversation")
    if isinstance(conversation, dict):
        return conversation
    routing = result.get("routing")
    if not isinstance(routing, dict):
        routing = result.get("debug", {}).get("routing", {})
    nested = routing.get("conversation") if isinstance(routing, dict) else None
    return nested if isinstance(nested, dict) else {}


def _result_routing_intent(result: dict) -> str | None:
    routing = result.get("routing")
    if not isinstance(routing, dict):
        routing = result.get("debug", {}).get("routing", {})
    intent = routing.get("intent") if isinstance(routing, dict) else None
    return str(intent) if intent else None


def _conversation_with_status(conversation: dict, routing_intent: str | None) -> dict:
    if not conversation:
        return {}
    return {
        **conversation,
        "status": "clarification_requested" if routing_intent == "CLARIFY" else "completed",
    }


@router.get("/models")
def list_generation_models(db: Session = Depends(get_db)):
    """Return the admin-approved model options exposed to chat users."""
    return ModelRouter(SettingsService(db)).public_policy()


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
def list_chat_sessions(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_requested_user),
):
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
    _: User = Depends(require_requested_user),
):
    get_user_or_404(user_id, db)
    title = payload.title.strip() or "New chat"
    session = ChatSession(user_id=user_id, title=title[:80])
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/users/{user_id}/sessions/{session_id}", response_model=ChatSessionDetail)
def read_chat_session(
    user_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_requested_user),
):
    return get_session_or_404(user_id, session_id, db)


@router.delete("/users/{user_id}/sessions/{session_id}")
def delete_chat_session(
    user_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_requested_user),
):
    session = get_session_or_404(user_id, session_id, db)
    db.delete(session)
    db.commit()
    return {"message": "Chat deleted successfully", "id": session_id}


# ---------------------------------------
# 1. Standard Chat Endpoint (IMPROVED)
# ---------------------------------------
@router.post("/", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    debug: bool = Query(False, description="Enable debug mode"),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
    _: None = Depends(RateLimiter(limit=20, window=60)),
):
    """
    Chat endpoint for RAG system:
    - Normal mode → clean response
    - Debug mode → includes retrieval insights
    """

    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if debug and (current_user is None or get_user_role(current_user.username) != "admin"):
        raise HTTPException(status_code=403, detail="Debug mode requires admin access")
    if body.user_id is not None or body.session_id is not None:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required for saved chat sessions")
        if current_user.id != body.user_id:
            raise HTTPException(status_code=403, detail="Cannot use another user's chat session")

    # Fix #3: single call, uses the already-open request DB session
    user_profile = get_user_profile(body.user_id, db)
    guest_conversation = (
        decode_guest_conversation_token(body.conversation_token)
        if body.user_id is None and body.session_id is None
        else None
    )

    # Authenticated history comes from the owned DB session. Guests use only
    # bounded, role-validated in-memory history from the current browser chat.
    chat_history = get_chat_history(
        body.user_id,
        body.session_id,
        body.history,
        db,
        guest_conversation,
    )

    pending_user_message_id = None
    if body.user_id is not None and body.session_id is not None:
        pending = create_pending_user_turn(
            db,
            user_id=body.user_id,
            session_id=body.session_id,
            content=body.message,
        )
        if pending is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        pending_user_message_id = pending.id

    try:
        if debug:
            result = rag_pipeline.run_debug(
                body.message,
                chat_history=chat_history,
                user_profile=user_profile,
                model_preference=body.model_preference,
            )
        else:
            result = rag_pipeline.run(
                body.message,
                chat_history=chat_history,
                user_profile=user_profile,
                model_preference=body.model_preference,
            )

        # Persist exchange to DB if caller supplied session context
        answer_text = result["answer"]
        sources = result.get("sources", [])
        conversation = _conversation_with_status(
            _result_conversation(result),
            _result_routing_intent(result),
        )
        conversation_token = (
            create_guest_conversation_token(conversation)
            if body.user_id is None and body.session_id is None
            else None
        )
        if body.user_id is not None and pending_user_message_id is not None and answer_text:
            try:
                update_user_turn_routing(
                    db,
                    user_id=body.user_id,
                    message_id=pending_user_message_id,
                    conversation=conversation,
                )
                finalize_chat_turn(
                    db,
                    user_id=body.user_id,
                    user_message_id=pending_user_message_id,
                    assistant_message=answer_text,
                    sources=sources,
                    conversation=conversation,
                )
            except Exception as exc:
                logger.warning("Chat endpoint: failed to persist exchange: %s", exc)
                db.rollback()

        if debug:
            return {
                "response": answer_text,
                "sources": sources,
                "debug": result.get("debug", {}),
                "requested_model": result.get("requested_model", body.model_preference),
                "selected_model": result.get("selected_model"),
                "fallback_used": result.get("fallback_used", False),
                "conversation": conversation,
                "conversation_token": conversation_token,
            }

        return ChatResponse(
            response=answer_text,
            sources=sources,
            requested_model=result.get("requested_model", body.model_preference),
            selected_model=result.get("selected_model"),
            fallback_used=result.get("fallback_used", False),
            conversation=conversation,
            conversation_token=conversation_token,
        )

    except HTTPException:
        raise
    except GenerationUnavailableError as exc:
        if body.user_id is not None and pending_user_message_id is not None:
            mark_user_turn_failed(
                db,
                user_id=body.user_id,
                message_id=pending_user_message_id,
            )
        raise HTTPException(status_code=503, detail=exc.public_payload()) from exc
    except Exception as exc:
        if body.user_id is not None and pending_user_message_id is not None:
            db.rollback()
            mark_user_turn_failed(
                db,
                user_id=body.user_id,
                message_id=pending_user_message_id,
            )
        logger.exception("Unexpected chat processing failure")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_PROCESSING_FAILED",
                "message": "Unable to process your request right now. Please try again.",
            }
        ) from exc


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
    _: None = Depends(RateLimiter(limit=20, window=60)),
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
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if body.user_id is not None or body.session_id is not None:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required for saved chat sessions")
        if current_user.id != body.user_id:
            raise HTTPException(status_code=403, detail="Cannot use another user's chat session")

    # Fix #3: single call using the already-open request DB session
    user_profile = get_user_profile(body.user_id, db)
    guest_conversation = (
        decode_guest_conversation_token(body.conversation_token)
        if body.user_id is None and body.session_id is None
        else None
    )

    # This raises HTTP 403 for an unowned authenticated session. A true guest
    # has no IDs and receives sanitized recent browser history instead.
    chat_history = get_chat_history(
        body.user_id,
        body.session_id,
        body.history,
        db,
        guest_conversation,
    )

    pending_user_message_id = None
    if body.user_id is not None and body.session_id is not None:
        pending = create_pending_user_turn(
            db,
            user_id=body.user_id,
            session_id=body.session_id,
            content=body.message,
        )
        if pending is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        pending_user_message_id = pending.id

    async def event_stream():
        full_response = ""
        assistant_message_id = None
        response_sources = []
        response_conversation = {}
        response_model = {
            "requested_model": body.model_preference,
            "selected_model": None,
            "fallback_used": False,
        }

        try:
            # ── Stream tokens from RAG pipeline ───────────────────────────
            async for event in rag_pipeline.stream_events(
                body.message,
                chat_history=chat_history,
                user_profile=user_profile,
                model_preference=body.model_preference,
            ):
                if event["type"] == "routing":
                    routing_intent = (event.get("routing") or {}).get("intent")
                    response_conversation = _conversation_with_status(
                        event.get("conversation") or {},
                        routing_intent,
                    )
                    if body.user_id is not None and pending_user_message_id is not None:
                        update_user_turn_routing(
                            db,
                            user_id=body.user_id,
                            message_id=pending_user_message_id,
                            conversation=response_conversation,
                        )
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
                    status_payload = json.dumps(
                        {
                            "type": "status",
                            "message": event.get("message", "Working on the answer..."),
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {status_payload}\n\n"
                    continue
                if event["type"] == "replace":
                    full_response = str(event.get("answer") or "")
                    replace_payload = json.dumps(
                        {
                            "type": "replace",
                            "answer": full_response,
                            "provisional": bool(event.get("provisional", False)),
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {replace_payload}\n\n"
                    continue
                token = str(event["token"])
                full_response += token
                stream_payload = json.dumps({"type": "stream", "token": token}, ensure_ascii=False)
                yield f"data: {stream_payload}\n\n"

        except GenerationUnavailableError as exc:
            if body.user_id is not None and pending_user_message_id is not None:
                mark_user_turn_failed(
                    db,
                    user_id=body.user_id,
                    message_id=pending_user_message_id,
                    conversation=response_conversation,
                )
            safe_sources = response_sources or exc.sources
            err_payload = json.dumps(
                {
                    "type": "error",
                    **exc.public_payload(sources=safe_sources),
                },
                ensure_ascii=False,
            )
            yield f"data: {err_payload}\n\n"
            return
        except Exception:
            logger.exception("Unexpected SSE stream failure")
            if body.user_id is not None and pending_user_message_id is not None:
                db.rollback()
                mark_user_turn_failed(
                    db,
                    user_id=body.user_id,
                    message_id=pending_user_message_id,
                    conversation=response_conversation,
                )
            err_payload = json.dumps(
                {
                    "type": "error",
                    "code": "CHAT_STREAM_FAILED",
                    "message": "Unable to complete the answer right now. Please try again.",
                    "sources": response_sources,
                },
                ensure_ascii=False,
            )
            yield f"data: {err_payload}\n\n"
            return

        # ── Persist to DB if caller supplied session context ──────────────────
        if body.user_id is not None and pending_user_message_id is not None and full_response:
            try:
                asst_msg = finalize_chat_turn(
                    db,
                    user_id=body.user_id,
                    user_message_id=pending_user_message_id,
                    assistant_message=full_response,
                    sources=response_sources,
                    conversation=response_conversation,
                )
                if asst_msg is not None:
                    assistant_message_id = asst_msg.id
            except Exception as exc:
                logger.warning("SSE: failed to persist chat exchange: %s", exc)
                db.rollback()

        # ── Signal completion ─────────────────────────────────────────
        done_payload = json.dumps(
            {
                "type": "done",
                "message_id": assistant_message_id,
                "sources": response_sources,
                "conversation": response_conversation,
                "conversation_token": (
                    create_guest_conversation_token(response_conversation)
                    if body.user_id is None and body.session_id is None
                    else None
                ),
                **response_model,
            },
            ensure_ascii=False,
        )
        yield f"data: {done_payload}\n\n"

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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_current_user),
):
    """
    Submit thumbs up (1) or thumbs down (-1) feedback for a specific chat message.
    Pass 0 to clear feedback.
    """
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not message.session or message.session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot give feedback on another user's message")
        
    if payload.feedback not in [-1, 0, 1]:
        raise HTTPException(status_code=400, detail="Feedback must be -1, 0, or 1")
        
    message.feedback = payload.feedback if payload.feedback != 0 else None
    db.commit()
    
    return {"status": "success", "message_id": message_id, "feedback": message.feedback}


@router.get("/feedback/messages")
def list_feedback_messages(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin_user),
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
