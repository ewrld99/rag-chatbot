from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, text, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.config import settings
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(Text, nullable=False, unique=True, index=True)
    password_hash = Column(Text, nullable=False)
    registration_number = Column(Text, nullable=True)
    programme = Column(Text, nullable=True)
    campus = Column(Text, nullable=True)
    admission_year = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(Text, nullable=False, default="New chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    feedback = Column(Integer, nullable=True)  # 1: Thumbs Up, -1: Thumbs Down
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("ChatSession", back_populates="messages")


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    title = Column(Text, nullable=True)
    filename = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    category = Column(Text, nullable=True)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status = Column(Text, nullable=True, default="active")
    source_url = Column(Text, nullable=True, unique=True, index=True)
    content_hash = Column(Text, nullable=True)
    last_crawled_at = Column(DateTime(timezone=True), nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
    tsv = Column(TSVECTOR, nullable=True)
    page_number = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)  # Using metadata_ to avoid conflict with SQLAlchemy Base.metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("DocumentModel", back_populates="chunks")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    key = Column(Text, unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(Text, nullable=True)
    is_editable = Column(Boolean, default=True)
    updated_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    admin = Column(Text, nullable=False)
    setting_key = Column(Text, nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FAQModel(Base):
    __tablename__ = "faqs"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, server_default=text("gen_random_uuid()"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(Text, nullable=True)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    fts_vector = Column(TSVECTOR, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    creator = relationship("User")


class ExternalLinkModel(Base):
    __tablename__ = "external_links"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(Text, nullable=False, unique=True, index=True)
    found_on = Column(Text, nullable=True)
    discovered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class CrawlerJob(Base):
    __tablename__ = "crawler_jobs"

    job_type = Column(Text, primary_key=True) # "full" or "announcements"
    status = Column(Text, nullable=False, default="idle")
    crawled_count = Column(Integer, default=0)
    max_pages = Column(Integer, default=0)
    current_url = Column(Text, nullable=True)
    last_run = Column(DateTime(timezone=True), nullable=True)


class CrawlerQueue(Base):
    __tablename__ = "crawler_queue"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(Text, ForeignKey("crawler_jobs.job_type", ondelete="CASCADE"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="pending") # pending, processing, completed, failed
    
    from sqlalchemy import UniqueConstraint
    __table_args__ = (UniqueConstraint('job_type', 'url', name='uq_crawler_queue_job_url'),)
