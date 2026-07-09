import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, auth, chat, query, ws_chat
from app.db.session import init_db
from app.core.scheduler import start_scheduler, stop_scheduler
from app.core.config import settings

_START_TIME = time.monotonic()          # wall-clock seconds since process start
_START_DATETIME = datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ Single init_db() call — inside lifespan only
    init_db()
    from app.db.session import SessionLocal
    from app.services.settings_service import seed_default_settings
    db = SessionLocal()
    try:
        seed_default_settings(db)
    finally:
        db.close()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

# ── Static files ─────────────────────────────────────────────────────────────
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")

# ── CORS — origins loaded from ALLOWED_ORIGINS env var ──────────────────────
allowed_origins = [
    o.strip()
    for o in settings.ALLOWED_ORIGINS.split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(ws_chat.router, tags=["WebSocket"])
app.include_router(query.router, prefix="/api", tags=["Query"])


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], include_in_schema=True)
async def health_check() -> JSONResponse:
    """
    Liveness + readiness probe for load balancers and uptime monitors.

    Returns HTTP 200 when the service is healthy (DB reachable).
    Returns HTTP 503 when the database is unreachable so the load balancer
    can route traffic away from this instance immediately.
    """
    uptime_seconds = round(time.monotonic() - _START_TIME, 1)

    # ── DB probe: cheap SELECT 1 ─────────────────────────────────────────────
    db_status = "ok"
    try:
        from app.db.session import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        db_status = f"error: {exc}"

    healthy = db_status == "ok"
    payload = {
        "service": "UDOM RAG Chatbot",
        "status": "healthy" if healthy else "degraded",
        "version": "1.0.0",
        "started_at": _START_DATETIME,
        "uptime_seconds": uptime_seconds,
        "checks": {
            "database": db_status,
        },
    }
    return JSONResponse(
        content=payload,
        status_code=200 if healthy else 503,
    )
