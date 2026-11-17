"""Token-aware Unicode-preserving chunks mapped to normalized source blocks."""

from uuid import uuid4

import tiktoken
from celery.exceptions import SoftTimeLimitExceeded
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.domain.documents import Chunk, ExtractionResult, ProcessingError, SourceSpan

CHUNKER_VERSION = "recursive-token-v1"
TOKENIZER = "cl100k_base"


def chunk_document(extraction: ExtractionResult, settings: Settings) -> tuple[Chunk, ...]:
    try:
        return _chunk_document(extraction, settings)
    except (ProcessingError, SoftTimeLimitExceeded):
        raise
    except Exception:
        raise ProcessingError(
            "CHUNKING_FAILED", "Document text could not be safely chunked."
        ) from None


def _chunk_document(extraction: ExtractionResult, settings: Settings) -> tuple[Chunk, ...]:
    encoding = tiktoken.get_encoding(TOKENIZER)
    text = extraction.text
    if not text.strip():
        raise ProcessingError("NO_EXTRACTABLE_TEXT", "No useful native text was found.")
    if len(text.encode("utf-8")) > settings.max_extracted_text_bytes:
        raise ProcessingError("EXTRACTED_TEXT_LIMIT", "Extracted text exceeds the byte limit.")
    if len(encoding.encode(text, disallowed_special=())) > settings.max_extracted_tokens:
        raise ProcessingError("EXTRACTED_TOKEN_LIMIT", "Extracted text exceeds the token limit.")
    # Partition without overlap first so offsets remain exact even for repeated
    # text. Then reserve and prepend a bounded original-text overlap. LangChain's
    # add_start_index uses character overlap, which cannot stand in for tokens.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER,
        chunk_size=settings.chunk_size_tokens - settings.chunk_overlap_tokens,
        chunk_overlap=0,
        disallowed_special=(),
        keep_separator=True,
        strip_whitespace=False,
    )
    result: list[Chunk] = []
    cursor = 0
    for part in splitter.split_text(text):
        if not text.startswith(part, cursor):
            raise ProcessingError("CHUNK_MAPPING_FAILED", "Chunk source mapping is incomplete.")
        start = cursor
        end = cursor + len(part)
        if settings.chunk_overlap_tokens and cursor:
            preceding = encoding.encode(text[:cursor], disallowed_special=())
            # Dropping an incomplete leading UTF-8 character affects overlap only;
            # every original character remains in its complete primary partition.
            overlap = encoding.decode(preceding[-settings.chunk_overlap_tokens :], errors="ignore")
            if not text[:cursor].endswith(overlap):
                raise ProcessingError("CHUNK_MAPPING_FAILED", "Chunk source mapping is incomplete.")
            while overlap and (
                len(encoding.encode(overlap, disallowed_special=())) > settings.chunk_overlap_tokens
                or len(encoding.encode(overlap + part, disallowed_special=()))
                > settings.chunk_size_tokens
            ):
                overlap = overlap[1:]
            start -= len(overlap)
        chunk_text = text[start:end]
        token_count = len(encoding.encode(chunk_text, disallowed_special=()))
        if token_count > settings.chunk_size_tokens:
            raise ProcessingError(
                "CHUNK_TOKEN_LIMIT", "A text segment exceeds the chunk token limit."
            )
        spans = tuple(
            SourceSpan(
                block.block_id,
                block.locator,
                max(start, block.start),
                min(end, block.end),
                max(start, block.start) - start,
                min(end, block.end) - start,
            )
            for block in extraction.blocks
            if block.start < end and block.end > start
        )
        if not spans:
            if not chunk_text.strip():
                # Join separators are not source blocks and need no embedding.
                cursor = end
                continue
            raise ProcessingError("CHUNK_MAPPING_FAILED", "A chunk has no source text.")
        result.append(Chunk(uuid4(), len(result), chunk_text, token_count, spans))
        if len(result) > settings.max_document_chunks:
            raise ProcessingError(
                "CHUNK_COUNT_LIMIT", "The document exceeds the chunk count limit."
            )
        cursor = end
    if not result or cursor != len(text):
        raise ProcessingError("CHUNK_MAPPING_FAILED", "Chunk source mapping is incomplete.")
    return tuple(result)
