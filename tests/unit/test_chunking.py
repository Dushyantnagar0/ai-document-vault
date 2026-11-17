"""Chunk provenance and real tokenizer bounds, including repeated and Unicode text."""

from typing import Any

import pytest
import tiktoken

from app.core.config import Settings
from app.domain.documents import ExtractionResult, ProcessingError, TextBlock
from app.services.chunking import TOKENIZER, chunk_document


def configured(**overrides: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="test-placeholder", **overrides)


def extracted(*parts: str) -> ExtractionResult:
    blocks = []
    cursor = 0
    for index, part in enumerate(parts):
        blocks.append(
            TextBlock(f"block-{index}", part, {"page": index + 1}, cursor, cursor + len(part))
        )
        cursor += len(part) + 2
    return ExtractionResult(
        "\n\n".join(parts),
        tuple(blocks),
        "fixture",
        "1",
        "complete_within_capability",
        (),
        len(parts),
    )


@pytest.mark.parametrize(
    "parts",
    [
        ("ordinary repeated text " * 250, "Last fact: the final limit is 72."),
        (("日本語の事実。 हिन्दी café 🧑🏽‍💻 " * 80).strip(), "終わりの重要な事実。"),
        ("<|endoftext|> is literal untrusted document data. " * 50,),
    ],
)
def test_token_bounds_unicode_and_exact_source_spans(parts: tuple[str, ...]) -> None:
    source = extracted(*parts)
    settings = configured(chunk_size_tokens=80, chunk_overlap_tokens=12)
    chunks = chunk_document(source, settings)
    encoding = tiktoken.get_encoding(TOKENIZER)
    assert len(chunks) > 1
    covered: set[int] = set()
    for ordinal, chunk in enumerate(chunks):
        assert chunk.ordinal == ordinal
        assert chunk.token_count == len(encoding.encode(chunk.text, disallowed_special=()))
        assert 0 < chunk.token_count <= settings.chunk_size_tokens
        assert "\ufffd" not in chunk.text
        assert chunk.source_spans
        for span in chunk.source_spans:
            original = source.text[span.document_start : span.document_end]
            assert original == chunk.text[span.chunk_start : span.chunk_end]
            block = next(block for block in source.blocks if block.block_id == span.block_id)
            assert span.locator == block.locator
            assert block.start <= span.document_start < span.document_end <= block.end
            covered.update(range(span.document_start, span.document_end))
    assert all(
        index in covered for block in source.blocks for index in range(block.start, block.end)
    )
    assert parts[-1][-15:] in chunks[-1].text
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_zero_overlap_and_cross_block_offsets() -> None:
    source = extracted("First fact. " * 60, "Last fact. " * 60)
    chunks = chunk_document(source, configured(chunk_size_tokens=50, chunk_overlap_tokens=0))
    assert "".join(chunk.text for chunk in chunks).replace("\n", "") == source.text.replace(
        "\n", ""
    )
    assert sum(
        span.document_end - span.document_start for chunk in chunks for span in chunk.source_spans
    ) == sum(len(block.text) for block in source.blocks)
    assert any(span.locator == {"page": 2} for chunk in chunks for span in chunk.source_spans)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"max_extracted_tokens": 10}, "EXTRACTED_TOKEN_LIMIT"),
        (
            {"max_document_chunks": 1, "chunk_size_tokens": 40, "chunk_overlap_tokens": 8},
            "CHUNK_COUNT_LIMIT",
        ),
        ({"max_extracted_text_bytes": 4}, "EXTRACTED_TEXT_LIMIT"),
    ],
)
def test_limits_reject_instead_of_truncating(overrides: dict[str, Any], code: str) -> None:
    with pytest.raises(ProcessingError) as captured:
        chunk_document(extracted("Bounded evidence. " * 200), configured(**overrides))
    assert captured.value.code == code


def test_empty_text_and_too_small_unicode_unit_fail_clearly() -> None:
    with pytest.raises(ProcessingError) as captured:
        chunk_document(extracted(""), configured())
    assert captured.value.code == "NO_EXTRACTABLE_TEXT"
    with pytest.raises(ProcessingError) as captured:
        chunk_document(extracted("🧑"), configured(chunk_size_tokens=1, chunk_overlap_tokens=0))
    assert captured.value.code == "CHUNK_TOKEN_LIMIT"
