"""Validated local-runtime settings, without provider calls or import-time loading."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError
from sqlalchemy.engine import URL

PositiveInt = Annotated[int, Field(gt=0)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Port = Annotated[int, Field(ge=1, le=65535)]


class ConfigurationError(ValueError):
    """Startup configuration failed; message contains no supplied input values."""


class Settings(BaseSettings):
    """Read environment overrides or a local dotenv file over explicit defaults.

    Model metadata is deliberately limited to the selected profiles. Adding a model
    requires reviewed context/output/tokenizer metadata and quality verification;
    arbitrary provider model names cannot establish compatible budgets or vectors.
    Persisted index-profile/schema compatibility is checked separately by the DB.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # A shared dotenv may also contain UI/Compose-only values.
        hide_input_in_errors=True,
        enable_decoding=False,
        frozen=True,
    )

    # Primary metadata: https://developers.openai.com/api/docs/models/gpt-5.4-mini.md
    # and /guides/embeddings. Not a claim of account access/live verification.
    CHAT_MODEL: ClassVar[str] = "gpt-5.4-mini-2026-03-17"
    # Reviewed published alias, not a dated snapshot. Its 1,050,000 context,
    # 922,000 input and 128,000 output capacities exceed the shared conservative
    # limits below: https://developers.openai.com/api/docs/models/gpt-6.1-sol.md
    SOURCE_ANSWER_MODEL: ClassVar[str] = "gpt-6.1-sol"
    SOURCE_ANSWER_REASONING_EFFORT: ClassVar[Literal["low"]] = "low"
    CHAT_CONTEXT_TOKENS: ClassVar[int] = 400_000
    CHAT_MODEL_MAX_INPUT_TOKENS: ClassVar[int] = 272_000
    CHAT_MODEL_MAX_OUTPUT_TOKENS: ClassVar[int] = 128_000
    EMBEDDING_MODEL: ClassVar[str] = "text-embedding-3-small"
    EMBEDDING_INPUT_TOKENS: ClassVar[int] = 8191
    EMBEDDING_REQUEST_TOKENS: ClassVar[int] = 300_000
    SCHEMA_VECTOR_DIMENSIONS: ClassVar[int] = 1536
    INSTRUCTION_RESERVE_TOKENS: ClassVar[int] = 1024

    openai_api_key: SecretStr = Field(repr=False)
    openai_chat_model: str = CHAT_MODEL
    openai_source_answer_model: str = SOURCE_ANSWER_MODEL
    openai_embedding_model: str = EMBEDDING_MODEL
    openai_embedding_dimensions: PositiveInt = SCHEMA_VECTOR_DIMENSIONS

    postgres_host: str = "postgres"
    postgres_port: Port = 5432
    postgres_db: str = "document_vault"
    postgres_user: str = "document_vault"
    postgres_password: SecretStr = Field(default=SecretStr("local_development_only"), repr=False)
    redis_url: SecretStr = Field(default=SecretStr("redis://redis:6379/0"), repr=False)
    storage_backend: Literal["local"] = "local"
    vector_backend: Literal["pgvector"] = "pgvector"
    local_storage_dir: Path = Path("/app/data/uploads")
    api_port: Port = 4000
    ui_port: Port = 4005

    max_upload_bytes: PositiveInt = 20_971_520
    max_pdf_pages: PositiveInt = 500
    max_docx_expanded_bytes: PositiveInt = 104_857_600
    max_docx_entries: PositiveInt = 2000
    max_extracted_text_bytes: PositiveInt = 2_097_152
    max_extracted_tokens: PositiveInt = 100_000
    max_document_chunks: PositiveInt = 500

    chunk_size_tokens: PositiveInt = 800
    chunk_overlap_tokens: Annotated[int, Field(ge=0)] = 120
    embedding_batch_size: Annotated[int, Field(gt=0, le=2048)] = 32
    embedding_batch_max_tokens: PositiveInt = 32_000
    analysis_group_max_tokens: PositiveInt = 12_000
    analysis_max_output_tokens: PositiveInt = 768
    summary_max_output_tokens: PositiveInt = 1500
    # Completion budgets include internal reasoning, unlike the visible insight cap.
    evidence_review_max_output_tokens: PositiveInt = 4096
    insight_max_output_tokens: PositiveInt = 3072
    insight_max_visible_tokens: PositiveInt = 600
    openai_review_reasoning_effort: Literal["none", "low"] = "low"
    max_document_ai_calls: PositiveInt = 64

    planner_max_output_tokens: PositiveInt = 256
    question_max_chars: PositiveInt = 4000
    question_max_tokens: PositiveInt = 1000
    chat_history_max_turns: PositiveInt = 4
    chat_history_max_tokens: PositiveInt = 2000
    retrieval_candidates: Annotated[int, Field(gt=0, le=100)] = 16
    retrieval_max_chunks: Annotated[int, Field(gt=0, le=8)] = 8
    retrieval_max_tokens: PositiveInt = 6400
    chat_prompt_max_tokens: PositiveInt = 12_000
    chat_max_output_tokens: PositiveInt = 1200
    openai_answer_reasoning_effort: Literal["none", "low"] = "low"

    openai_embedding_timeout_seconds: PositiveSeconds = 15
    openai_generation_timeout_seconds: PositiveSeconds = 30
    # Current turn reservation and server shutdown contracts assume <=90 seconds.
    chat_timeout_seconds: Annotated[float, Field(gt=0, le=90, allow_inf_nan=False)] = 90
    celery_worker_concurrency: PositiveInt = 2
    api_max_concurrent_ai_requests: PositiveInt = 2
    processing_max_attempts: PositiveInt = 3
    processing_deadline_seconds: PositiveSeconds = 1800

    def __init__(self, **values: Any) -> None:
        try:
            super().__init__(**values)
        except ValidationError as exc:
            # Even ValidationError.errors()/json() normally retain raw inputs.
            # Expose only input-free diagnostics at the settings boundary.
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            diagnostics = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
                for error in errors
            )
            raise ConfigurationError(f"Invalid configuration: {diagnostics}") from None
        except SettingsError:
            raise ConfigurationError("Unable to read configuration sources") from None

    @field_validator("openai_api_key")
    @classmethod
    def validate_provider_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if not key or any(char.isspace() for char in key):
            raise ValueError("OpenAI API key must be nonblank and contain no whitespace")
        return value

    @field_validator("postgres_password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("PostgreSQL password must be nonblank")
        return value

    @field_validator("postgres_host", "postgres_db", "postgres_user")
    @classmethod
    def validate_database_component(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Database component must be nonblank and contain no NUL")
        return value

    @field_validator("local_storage_dir")
    @classmethod
    def validate_storage_path(cls, value: Path) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError("Local storage directory must be an absolute path without NUL")
        return value

    @field_validator("redis_url")
    @classmethod
    def validate_broker_url(cls, value: SecretStr) -> SecretStr:
        try:
            raw = value.get_secret_value()
            parsed = urlsplit(raw)
            valid = (
                parsed.scheme in {"redis", "rediss"}
                and bool(parsed.hostname)
                and not any(char.isspace() for char in raw)
                and (parsed.port is None or 1 <= parsed.port <= 65535)
                and (not parsed.path or (parsed.path.startswith("/") and parsed.path[1:].isdigit()))
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Redis URL must specify redis/rediss host, valid port and database")
        return value

    @model_validator(mode="after")
    def validate_joint_budgets(self) -> Self:
        if self.openai_chat_model != self.CHAT_MODEL:
            raise ValueError("Unsupported chat model metadata; add a reviewed model profile")
        if self.openai_source_answer_model != self.SOURCE_ANSWER_MODEL:
            raise ValueError(
                "Unsupported source-answer model metadata; add a reviewed model profile"
            )
        if self.openai_embedding_model != self.EMBEDDING_MODEL:
            raise ValueError("Unsupported embedding model metadata; add a reviewed index profile")
        if self.openai_embedding_dimensions != self.SCHEMA_VECTOR_DIMENSIONS:
            raise ValueError("Embedding dimensions must match the fixed vector(1536) schema")
        if self.api_port == self.ui_port:
            raise ValueError("API and UI host ports must differ")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.chunk_size_tokens > self.EMBEDDING_INPUT_TOKENS:
            raise ValueError("Chunk size exceeds embedding model input capacity")
        if self.question_max_tokens > self.EMBEDDING_INPUT_TOKENS:
            raise ValueError("Question token budget exceeds embedding model input capacity")
        if not (
            self.chunk_size_tokens
            <= self.embedding_batch_max_tokens
            <= self.EMBEDDING_REQUEST_TOKENS
        ):
            raise ValueError(
                "Embedding batch budget must fit a chunk and provider request capacity"
            )
        if self.retrieval_max_chunks > self.retrieval_candidates:
            raise ValueError("Retrieval candidates must cover the selected chunk count")
        if self.retrieval_max_tokens < self.chunk_size_tokens:
            raise ValueError("Retrieval budget must fit at least one chunk")
        chat_input = (
            self.question_max_tokens
            + self.chat_history_max_tokens
            + self.retrieval_max_tokens
            + self.INSTRUCTION_RESERVE_TOKENS
        )
        if chat_input > self.chat_prompt_max_tokens:
            raise ValueError(
                "Chat prompt budget must reserve question, history, evidence and instructions"
            )
        planner_input = (
            self.question_max_tokens
            + self.chat_history_max_tokens
            + self.INSTRUCTION_RESERVE_TOKENS
        )
        analysis_input = self.analysis_group_max_tokens + self.INSTRUCTION_RESERVE_TOKENS
        if any(
            budget > self.CHAT_MODEL_MAX_INPUT_TOKENS
            for budget in (self.chat_prompt_max_tokens, planner_input, analysis_input)
        ):
            raise ValueError(
                "Chat, planner and analysis input budgets must fit model input capacity"
            )
        if self.chat_prompt_max_tokens + self.chat_max_output_tokens > self.CHAT_CONTEXT_TOKENS:
            raise ValueError("Chat input and output budgets exceed model context capacity")
        if planner_input + self.planner_max_output_tokens > self.CHAT_CONTEXT_TOKENS:
            raise ValueError("Planner input and output budgets exceed model context capacity")
        for output in (
            self.chat_max_output_tokens,
            self.analysis_max_output_tokens,
            self.summary_max_output_tokens,
            self.planner_max_output_tokens,
            self.evidence_review_max_output_tokens,
            self.insight_max_output_tokens,
        ):
            if output > self.CHAT_MODEL_MAX_OUTPUT_TOKENS:
                raise ValueError("Generation output budget exceeds model output capacity")
        if self.insight_max_visible_tokens > self.insight_max_output_tokens:
            raise ValueError("Visible insight budget must fit its total completion budget")
        if (
            analysis_input
            + max(
                self.analysis_max_output_tokens,
                self.summary_max_output_tokens,
                self.evidence_review_max_output_tokens,
                self.insight_max_output_tokens,
            )
            > self.CHAT_CONTEXT_TOKENS
        ):
            raise ValueError("Analysis input and output budgets exceed model context capacity")
        if self.analysis_max_output_tokens >= self.analysis_group_max_tokens:
            raise ValueError("Analysis reduction output must be smaller than its input budget")
        if max(
            self.openai_embedding_timeout_seconds,
            self.openai_generation_timeout_seconds,
        ) > min(self.chat_timeout_seconds, self.processing_deadline_seconds):
            raise ValueError("Provider timeouts must fit chat and processing deadlines")
        return self

    @property
    def chat_configuration(self) -> dict[str, str | int | float]:
        """Non-secret recipe persisted with turns for reproducible evaluation."""
        names = (
            "openai_chat_model",
            "openai_source_answer_model",
            "openai_embedding_model",
            "openai_embedding_dimensions",
            "planner_max_output_tokens",
            "question_max_chars",
            "question_max_tokens",
            "chat_history_max_turns",
            "chat_history_max_tokens",
            "retrieval_candidates",
            "retrieval_max_chunks",
            "retrieval_max_tokens",
            "chat_prompt_max_tokens",
            "chat_max_output_tokens",
            "openai_answer_reasoning_effort",
            "evidence_review_max_output_tokens",
            "openai_review_reasoning_effort",
            "openai_embedding_timeout_seconds",
            "openai_generation_timeout_seconds",
            "chat_timeout_seconds",
        )
        return {
            **{name: getattr(self, name) for name in names},
            "openai_source_answer_reasoning_effort": self.SOURCE_ANSWER_REASONING_EFFORT,
        }

    @property
    def database_url(self) -> URL:
        """Sync psycopg URL; pass the URL object directly to SQLAlchemy."""
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    @property
    def async_database_url(self) -> URL:
        """Explicit async psycopg dialect; URL repr masks the password."""
        return self.database_url.set(drivername="postgresql+psycopg_async")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load once at application startup/dependency resolution, never on import."""
    return Settings()
