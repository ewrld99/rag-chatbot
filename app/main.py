import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, auth, chat, query, ws_chat
from app.db.session import init_db
from app.core.scheduler import start_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
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

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(ws_chat.router, tags=["WebSocket"])
app.include_router(query.router, prefix="/api", tags=["Query"])
