"""Relational and published-vector schema introduced by incremental migrations."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IndexProfile(Base):
    """Immutable identity of a compatible embedding and chunking recipe."""

    __tablename__ = "index_profiles"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_index_profiles_fingerprint"),
        CheckConstraint("dimensions = 1536", name="ck_index_profiles_dimensions"),
        CheckConstraint("chunk_size_tokens > 0", name="ck_index_profiles_chunk_size"),
        CheckConstraint(
            "chunk_overlap_tokens >= 0 AND chunk_overlap_tokens < chunk_size_tokens",
            name="ck_index_profiles_overlap",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(40), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    tokenizer: Mapped[str] = mapped_column(String(80), nullable=False)
    normalization_version: Mapped[str] = mapped_column(String(40), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(80), nullable=False)
    chunk_size_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING','PROCESSING','COMPLETED','FAILED')", name="ck_document_status"
        ),
        CheckConstraint(
            "byte_size >= 0 AND attempt_count >= 0 AND ai_calls_reserved >= 0",
            name="ck_document_counts",
        ),
        ForeignKeyConstraint(
            ["id", "current_attempt_id"],
            ["processing_attempts.document_id", "processing_attempts.id"],
            name="fk_document_current_attempt",
            use_alter=True,
        ),
        ForeignKeyConstraint(
            ["id", "published_attempt_id"],
            ["processing_attempts.document_id", "processing_attempts.id"],
            name="fk_document_published_attempt",
            use_alter=True,
        ),
        Index("ix_documents_dispatch", "status", "next_dispatch_at"),
        Index("ix_documents_cursor", "created_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(100))
    byte_size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(64), unique=True)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("index_profiles.id", ondelete="RESTRICT"))
    upload_request_id: Mapped[UUID | None] = mapped_column(unique=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    stage: Mapped[str] = mapped_column(String(40), default="QUEUED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    extracted_tokens: Mapped[int | None] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer)
    block_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    extractor_version: Mapped[str | None] = mapped_column(String(100))
    extraction_coverage: Mapped[str | None] = mapped_column(String(40))
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=sql_text("'[]'::jsonb")
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    ai_calls_reserved: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    next_dispatch_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dispatch_reserved_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_attempt_id: Mapped[UUID | None] = mapped_column()
    published_attempt_id: Mapped[UUID | None] = mapped_column()
    claim_token: Mapped[UUID | None] = mapped_column()
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProcessingAttempt(Base):
    __tablename__ = "processing_attempts"
    __table_args__ = (
        UniqueConstraint("document_id", "attempt_number", name="uq_attempt_number"),
        UniqueConstraint("document_id", "id", name="uq_attempt_document_id"),
        CheckConstraint("attempt_number > 0", name="ck_attempt_number"),
        CheckConstraint(
            "outcome IN ('RUNNING','SUCCEEDED','FAILED','RETRY','INTERRUPTED')",
            name="ck_attempt_outcome",
        ),
        Index("ix_attempts_started", "started_at", "id"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"))
    attempt_number: Mapped[int] = mapped_column(Integer)
    claim_token: Mapped[UUID] = mapped_column(unique=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str] = mapped_column(String(20), default="RUNNING")
    stage: Mapped[str] = mapped_column(String(40), default="EXTRACTING")
    stage_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stage_durations: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    broker_task_id: Mapped[str | None] = mapped_column(String(100))


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "attempt_id"],
            ["processing_attempts.document_id", "processing_attempts.id"],
            name="fk_chunk_attempt",
        ),
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_ordinal"),
        CheckConstraint("ordinal >= 0 AND token_count > 0", name="ck_chunk_counts"),
        Index("ix_chunks_document", "document_id"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="RESTRICT"))
    attempt_id: Mapped[UUID] = mapped_column()
    profile_id: Mapped[UUID] = mapped_column(ForeignKey("index_profiles.id", ondelete="RESTRICT"))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    source_spans: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))


class DocumentInsights(Base):
    __tablename__ = "document_insights"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "attempt_id"],
            ["processing_attempts.document_id", "processing_attempts.id"],
            name="fk_insights_attempt",
        ),
        CheckConstraint(
            "covered_chunks > 0 AND covered_chunks = total_chunks", name="ck_insights_full_analysis"
        ),
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), primary_key=True
    )
    attempt_id: Mapped[UUID] = mapped_column(unique=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    covered_chunks: Mapped[int] = mapped_column(Integer)
    total_chunks: Mapped[int] = mapped_column(Integer)
    extraction_coverage: Mapped[str] = mapped_column(String(40))
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    model: Mapped[str] = mapped_column(String(120))
    prompt_version: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PerformanceSample(Base):
    __tablename__ = "system_performance_samples"
    __table_args__ = (
        CheckConstraint("duration_ms >= 0", name="ck_sample_duration"),
        Index("ix_samples_recorded_operation", "recorded_at", "operation"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    operation: Mapped[str] = mapped_column(String(120))
    duration_ms: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(30))
    error_class: Mapped[str | None] = mapped_column(String(80))
    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT")
    )
    attempt_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_attempts.id", ondelete="RESTRICT")
    )
    turn_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("chat_turns.id", ondelete="RESTRICT", name="fk_sample_chat_turn")
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        CheckConstraint("next_turn_number > 0", name="ck_session_next_turn"),
        ForeignKeyConstraint(
            ["id", "active_turn_id"],
            ["chat_turns.session_id", "chat_turns.id"],
            name="fk_session_active_turn",
            use_alter=True,
        ),
        Index("ix_sessions_cursor", "updated_at", "id"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    next_turn_number: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    active_turn_id: Mapped[UUID | None] = mapped_column()


class ChatSessionDocument(Base):
    __tablename__ = "chat_session_documents"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_session_single_document"),
        Index("ix_membership_document", "document_id", "session_id"),
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="RESTRICT"), primary_key=True
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), primary_key=True
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "id", name="uq_turn_session_id"),
        UniqueConstraint("session_id", "turn_number", name="uq_turn_number"),
        UniqueConstraint("session_id", "client_request_id", name="uq_turn_request"),
        CheckConstraint(
            "turn_number > 0 AND ai_calls_reserved >= 0 AND ai_calls_reserved <= 5",
            name="ck_turn_counts",
        ),
        CheckConstraint("status IN ('PENDING','COMPLETED','FAILED')", name="ck_turn_status"),
        Index("ix_turns_expiry", "status", "expires_at"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="RESTRICT"))
    turn_number: Mapped[int] = mapped_column(Integer)
    client_request_id: Mapped[UUID] = mapped_column()
    question_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    claim_token: Mapped[UUID] = mapped_column(unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ai_calls_reserved: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    planner_intent: Mapped[str | None] = mapped_column(String(20))
    retrieval_query: Mapped[str | None] = mapped_column(Text)
    evidence_scope: Mapped[str | None] = mapped_column(String(40))
    evidence_status: Mapped[str | None] = mapped_column(String(20))
    context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    versions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    timings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    response_status: Mapped[int | None] = mapped_column(Integer)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "turn_id"],
            ["chat_turns.session_id", "chat_turns.id"],
            name="fk_message_turn_session",
        ),
        UniqueConstraint("session_id", "sequence", name="uq_message_sequence"),
        UniqueConstraint("turn_id", "role", name="uq_message_role"),
        CheckConstraint(
            "sequence > 0 AND role IN ('user','assistant')", name="ck_message_role_sequence"
        ),
        CheckConstraint("status IN ('PENDING','COMPLETED','FAILED')", name="ck_message_status"),
        CheckConstraint(
            "content IS NOT NULL OR (role='assistant' AND status='PENDING')",
            name="ck_message_content",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="RESTRICT"))
    turn_id: Mapped[UUID] = mapped_column()
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
