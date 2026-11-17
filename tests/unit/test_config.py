"""Configuration boundaries require no database, network or provider account."""

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy.engine import make_url

from app.core.config import ConfigurationError, Settings, get_settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Callable[..., Settings]:
    def create(**overrides: Any) -> Settings:
        return Settings(
            _env_file=None,
            **{"openai_api_key": "test-key-not-a-real-credential", **overrides},
        )

    return create


def test_defaults_match_committed_template(settings: Callable[..., Settings]) -> None:
    configured = settings()
    template = Path(__file__).resolve().parents[2] / ".env.example"
    for line in template.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("=", 1)
        if name == "OPENAI_API_KEY":
            continue
        actual = getattr(configured, name.lower())
        if isinstance(actual, SecretStr):
            actual = actual.get_secret_value()
        if isinstance(actual, (int, float)):
            assert actual == float(expected), name
        else:
            assert str(actual) == expected, name


def test_environment_overrides_dotenv_and_ignores_ui_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / "settings.env"
    dotenv.write_text(
        "OPENAI_API_KEY=dotenv-placeholder\nAPI_PORT=4100\nUI_API_URL=http://api:8000\n"
    )
    monkeypatch.setenv("API_PORT", "4200")
    configured = Settings(_env_file=dotenv)
    assert configured.api_port == 4200
    assert configured.openai_api_key.get_secret_value() == "dotenv-placeholder"
    assert "ui_api_url" not in configured.model_dump()


@pytest.mark.parametrize("key", [None, "", "   ", "key\nsecret"])
def test_provider_key_is_required_and_nonblank(key: str | None) -> None:
    values = {} if key is None else {"openai_api_key": key}
    with pytest.raises(ConfigurationError, match="openai_api_key"):
        Settings(_env_file=None, **values)


@pytest.mark.parametrize(
    "overrides, diagnostic",
    [
        ({"storage_backend": "s3"}, "storage_backend"),
        ({"vector_backend": "pinecone"}, "vector_backend"),
        ({"openai_chat_model": "unknown-model"}, "chat model metadata"),
        ({"openai_embedding_model": "text-embedding-3-large"}, "embedding model metadata"),
        ({"openai_embedding_dimensions": 512}, "vector\\(1536\\)"),
        ({"max_upload_bytes": 0}, "max_upload_bytes"),
        ({"processing_max_attempts": -1}, "processing_max_attempts"),
        ({"openai_generation_timeout_seconds": "NaN"}, "finite"),
        ({"api_port": 65536}, "api_port"),
        ({"api_port": 4005}, "ports must differ"),
        ({"local_storage_dir": "relative/uploads"}, "absolute path"),
        ({"postgres_password": "  "}, "password must be nonblank"),
        ({"postgres_db": "\x00"}, "no NUL"),
        ({"redis_url": "http://redis:6379/0"}, "Redis URL"),
        ({"redis_url": "redis://redis:70000/0"}, "Redis URL"),
        ({"redis_url": "redis://redis/-1"}, "Redis URL"),
        ({"chunk_overlap_tokens": 800}, "overlap"),
        ({"chunk_size_tokens": 8192}, "embedding model input"),
        ({"question_max_tokens": 8192}, "Question token budget"),
        ({"embedding_batch_max_tokens": 799}, "Embedding batch budget"),
        ({"embedding_batch_max_tokens": 300001}, "Embedding batch budget"),
        ({"retrieval_candidates": 101}, "retrieval_candidates"),
        ({"retrieval_max_chunks": 9}, "retrieval_max_chunks"),
        ({"retrieval_candidates": 7}, "Retrieval candidates"),
        ({"retrieval_max_tokens": 799}, "Retrieval budget"),
        ({"chat_prompt_max_tokens": 10423}, "reserve question"),
        ({"chat_prompt_max_tokens": 300000}, "model input capacity"),
        ({"chat_max_output_tokens": 128001}, "model output capacity"),
        ({"evidence_review_max_output_tokens": 128001}, "model output capacity"),
        ({"insight_max_output_tokens": 128001}, "model output capacity"),
        ({"insight_max_visible_tokens": 3073}, "Visible insight budget"),
        ({"openai_review_reasoning_effort": "high"}, "openai_review_reasoning_effort"),
        ({"analysis_group_max_tokens": 270977}, "model input capacity"),
        ({"analysis_max_output_tokens": 12000}, "reduction output"),
        ({"processing_deadline_seconds": 29}, "Provider timeouts"),
        ({"chat_timeout_seconds": 91}, "chat_timeout_seconds"),
    ],
)
def test_rejects_incompatible_configuration(
    settings: Callable[..., Settings], overrides: dict[str, Any], diagnostic: str
) -> None:
    with pytest.raises(ConfigurationError, match=diagnostic):
        settings(**overrides)


def test_exact_budget_boundaries_and_zero_overlap_are_valid(
    settings: Callable[..., Settings],
) -> None:
    configured = settings(
        chunk_overlap_tokens=0,
        retrieval_candidates=8,
        embedding_batch_max_tokens=800,
        chat_prompt_max_tokens=10424,
        processing_deadline_seconds=30,
    )
    assert configured.chunk_overlap_tokens == 0
    assert configured.chat_prompt_max_tokens == 10424


def test_review_recipe_is_persistable_and_visible_insight_limit_is_independent(
    settings: Callable[..., Settings],
) -> None:
    configured = settings(
        evidence_review_max_output_tokens=8192,
        insight_max_output_tokens=4096,
        insight_max_visible_tokens=700,
        openai_review_reasoning_effort="none",
    )
    assert configured.chat_configuration["evidence_review_max_output_tokens"] == 8192
    assert configured.chat_configuration["openai_review_reasoning_effort"] == "none"
    assert configured.insight_max_visible_tokens == 700
    assert configured.insight_max_output_tokens == 4096


def test_exact_model_input_and_combined_context_bounds(
    settings: Callable[..., Settings],
) -> None:
    configured = settings(
        chat_prompt_max_tokens=272000,
        chat_max_output_tokens=128000,
        analysis_group_max_tokens=270976,
    )
    assert (
        configured.chat_prompt_max_tokens + configured.chat_max_output_tokens
        == configured.CHAT_CONTEXT_TOKENS
    )
    assert (
        configured.analysis_group_max_tokens + configured.INSTRUCTION_RESERVE_TOKENS
        == configured.CHAT_MODEL_MAX_INPUT_TOKENS
    )


def test_sync_async_urls_round_trip_special_credentials_and_mask_password(
    settings: Callable[..., Settings],
) -> None:
    password = "synthetic:@/percent%?# password"
    username = "synthetic@user:name"
    configured = settings(postgres_user=username, postgres_password=password)
    for url, dialect in (
        (configured.database_url, "postgresql+psycopg"),
        (configured.async_database_url, "postgresql+psycopg_async"),
    ):
        assert url.drivername == dialect
        assert password not in str(url)
        assert password not in repr(url)
        decoded = make_url(url.render_as_string(hide_password=False))
        assert decoded.username == username
        assert decoded.password == password
        assert decoded.host == "postgres"
        assert decoded.database == "document_vault"


def test_credentials_do_not_leak_from_settings_or_invalid_input_diagnostics(
    settings: Callable[..., Settings],
) -> None:
    sentinel = "synthetic-private-marker"
    configured = settings(
        openai_api_key=sentinel,
        postgres_password=sentinel,
        redis_url=f"redis://:{sentinel}@redis:6379/0",
    )
    assert sentinel not in repr(configured)
    assert sentinel not in configured.model_dump_json()
    for overrides in (
        {"openai_api_key": f"{sentinel} invalid"},
        {"max_upload_bytes": sentinel},
        {"redis_url": f"http://:{sentinel}@redis/0"},
        {"storage_backend": sentinel},
        {"openai_chat_model": sentinel},
    ):
        with pytest.raises(ConfigurationError) as captured:
            settings(**overrides)
        assert sentinel not in str(captured.value)
        assert sentinel not in repr(captured.value)
        assert sentinel not in "".join(traceback.format_exception(captured.value))
