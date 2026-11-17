"""Run only against the isolated Compose stack, never the personal installation."""

import os
import time
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import IndexProfile
from app.tasks.celery_app import PROBE_TASK_NAME, app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


def test_migrations_profile_constraints_and_real_vector_distance() -> None:
    settings = get_settings()
    assert settings.postgres_db == "vault_test", "Refusing a non-test database"
    command.upgrade(Config("alembic.ini"), "head")
    engine = create_engine(settings.database_url, hide_parameters=True)
    try:
        with Session(engine) as session, session.begin():
            profile = IndexProfile(
                fingerprint=uuid4().hex * 2,
                embedding_provider="openai",
                embedding_model=settings.openai_embedding_model,
                dimensions=1536,
                tokenizer="cl100k_base",
                normalization_version="test-v1",
                chunker_version="test-v1",
                chunk_size_tokens=800,
                chunk_overlap_tokens=120,
            )
            session.add(profile)
            session.flush()
            profile_id = profile.id
            session.expunge(profile)
            loaded = session.get(IndexProfile, profile_id)
            assert loaded is not None
            assert loaded.embedding_model == settings.openai_embedding_model
            session.execute(text("CREATE TEMP TABLE vector_probe (id int, embedding vector(1536))"))
            basis = "[" + ",".join(["1"] + ["0"] * 1535) + "]"
            session.execute(
                text("INSERT INTO vector_probe VALUES (1, CAST(:vector AS vector))"),
                {"vector": basis},
            )
            distance = session.scalar(
                text("SELECT embedding <=> CAST(:vector AS vector) FROM vector_probe"),
                {"vector": basis},
            )
            assert distance == pytest.approx(0.0)
            assert session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
            with pytest.raises(DBAPIError), session.begin_nested():
                session.execute(text("INSERT INTO vector_probe VALUES (2, '[1,0]'::vector)"))
            with pytest.raises(IntegrityError), session.begin_nested():
                session.execute(
                    text("UPDATE index_profiles SET dimensions = 512 WHERE id = :id"),
                    {"id": profile_id},
                )
            session.delete(loaded)
    finally:
        engine.dispose()


def test_actual_redis_worker_consumes_probe_and_shares_nonroot_storage() -> None:
    settings = get_settings()
    assert settings.postgres_db == "vault_test", "Refusing non-test configuration"
    nonce = str(uuid4())
    receipt = settings.local_storage_dir / ".probes" / nonce
    assert not receipt.exists()
    app.send_task(PROBE_TASK_NAME, args=[nonce], queue="ingestion", retry=False)
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            if receipt.exists():
                assert receipt.read_text() == nonce
                assert receipt.stat().st_uid == 10001
                break
            time.sleep(0.2)
        else:
            pytest.fail("Real worker did not consume the fresh probe within 20 seconds")
    finally:
        receipt.unlink(missing_ok=True)


def test_running_api_and_swagger() -> None:
    response = httpx.get("http://api:8000/health/ready", timeout=10)
    assert response.status_code == 200
    assert response.json()["components"]["database"] == "ready"
    assert httpx.get("http://api:8000/openapi.json", timeout=5).status_code == 200
