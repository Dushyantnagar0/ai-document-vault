"""Content-addressed index recipes prevent accidental mixing of vector spaces."""

import hashlib
import json
from importlib.metadata import version
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import IndexProfile


def profile_values(settings: Settings) -> dict[str, Any]:
    recipe = {
        "embedding_provider": "openai",
        "embedding_model": settings.openai_embedding_model,
        "dimensions": settings.openai_embedding_dimensions,
        "tokenizer": f"cl100k_base:tiktoken-{version('tiktoken')}",
        "normalization_version": "nfc-v1",
        "chunker_version": f"recursive-token-v1:lc-{version('langchain-text-splitters')}",
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return {"id": uuid4(), "fingerprint": fingerprint, **recipe}


async def ensure_profile(session: AsyncSession, settings: Settings) -> UUID:
    values = profile_values(settings)
    await session.execute(
        insert(IndexProfile).values(**values).on_conflict_do_nothing(index_elements=["fingerprint"])
    )
    result = await session.scalar(
        select(IndexProfile.id).where(IndexProfile.fingerprint == values["fingerprint"])
    )
    assert result is not None
    return result
