"""Real parser fixtures verify source order, bounded inputs and honest coverage."""

import asyncio
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import docx
import pymupdf
import pytest
from celery.exceptions import SoftTimeLimitExceeded

from app.adapters import extraction
from app.adapters.extraction import DOCX_MEDIA_TYPE, PDF_MEDIA_TYPE, extractor_for
from app.core.config import Settings
from app.domain.documents import ProcessingError


def configured(**overrides: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="test-placeholder", **overrides)


def pdf(*texts: str, image: bool = False, encrypted: bool = False) -> bytes:
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
            if image:
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
                pixmap.clear_with(255)
                page.insert_image(pymupdf.Rect(80, 100, 100, 120), stream=pixmap.tobytes("png"))
        if encrypted:
            return document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret"
            )
        return document.tobytes()


def word_document() -> bytes:
    document = docx.Document()
    document.add_paragraph("Cafe\u0301 first")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Left cell"
    table.cell(0, 1).text = "Right cell"
    document.add_paragraph("Last fact: limit is 72.")
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_pdf_real_page_locators_offsets_and_blank_page() -> None:
    result = extractor_for(PDF_MEDIA_TYPE, configured()).extract(
        pdf("First page fact", "", "Late page fact")
    )
    assert result.page_count == 3
    assert [block.locator["page"] for block in result.blocks] == [1, 3]
    assert result.text == "First page fact\n\nLate page fact"
    assert result.coverage == "unknown"
    assert any(
        warning.code == "TEXTLESS_PAGE" and warning.locator == {"page": 2}
        for warning in result.warnings
    )
    for block in result.blocks:
        assert result.text[block.start : block.end] == block.text
    assert result.extractor_name == "pymupdf"
    assert result.extractor_version


def test_pdf_images_are_visible_coverage_limitations() -> None:
    result = extractor_for(PDF_MEDIA_TYPE, configured()).extract(pdf("A useful fact", image=True))
    assert result.coverage == "unknown"
    assert any(warning.code == "IMAGES_NOT_INTERPRETED" for warning in result.warnings)


def test_image_bytes_are_not_accepted_as_a_pdf() -> None:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
    pixmap.clear_with(255)
    with pytest.raises(ProcessingError) as captured:
        extractor_for(PDF_MEDIA_TYPE, configured()).extract(pixmap.tobytes("png"))
    assert captured.value.code == "INVALID_PDF"


def test_docx_order_nfc_and_table_locators() -> None:
    result = extractor_for(DOCX_MEDIA_TYPE, configured()).extract(word_document())
    assert [block.text for block in result.blocks] == [
        "Café first",
        "Left cell",
        "Right cell",
        "Last fact: limit is 72.",
    ]
    assert [block.locator["path"] for block in result.blocks] == [
        "body/paragraph-1",
        "body/table-1/row-1/cell-1/paragraph-1",
        "body/table-1/row-1/cell-2/paragraph-1",
        "body/paragraph-2",
    ]
    assert result.coverage == "complete_within_capability"
    assert result.page_count is None
    for block in result.blocks:
        assert result.text[block.start : block.end] == block.text
        assert "page" not in block.locator


@pytest.mark.parametrize("interruption", [SoftTimeLimitExceeded, asyncio.CancelledError])
@pytest.mark.parametrize("media_type", [PDF_MEDIA_TYPE, DOCX_MEDIA_TYPE])
def test_actual_parser_boundary_preserves_interruptions(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
    media_type: str,
) -> None:
    payload = pdf("Known fact") if media_type == PDF_MEDIA_TYPE else word_document()
    calls = []

    def interrupted(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise interruption()

    if media_type == PDF_MEDIA_TYPE:
        monkeypatch.setattr(extraction.pymupdf, "open", interrupted)
    else:
        monkeypatch.setattr(extraction.docx, "Document", interrupted)
    with pytest.raises(interruption):
        extractor_for(media_type, configured()).extract(payload)
    assert len(calls) == 1


def test_docx_nested_table_and_header_warning() -> None:
    document = docx.Document()
    document.add_paragraph("Before")
    cell = document.add_table(rows=1, cols=1).cell(0, 0)
    cell.text = "Outer"
    cell.add_table(rows=1, cols=1).cell(0, 0).text = "Nested"
    document.sections[0].header.paragraphs[0].text = "Omitted header"
    document.add_paragraph("After")
    stream = BytesIO()
    document.save(stream)
    result = extractor_for(DOCX_MEDIA_TYPE, configured()).extract(stream.getvalue())
    assert [block.text for block in result.blocks] == ["Before", "Outer", "Nested", "After"]
    assert "table-1/row-1/cell-1/table-1" in str(result.blocks[2].locator)
    assert result.coverage == "partial"
    assert any(warning.code == "UNSUPPORTED_CONTENT" for warning in result.warnings)


@pytest.mark.parametrize(
    "media,content,overrides,code",
    [
        (PDF_MEDIA_TYPE, b"private corrupt bytes", {}, "INVALID_PDF"),
        (DOCX_MEDIA_TYPE, b"private corrupt bytes", {}, "INVALID_DOCX"),
        (PDF_MEDIA_TYPE, pdf(""), {}, "NO_EXTRACTABLE_TEXT"),
        (PDF_MEDIA_TYPE, pdf("Secret", encrypted=True), {}, "ENCRYPTED_PDF"),
        (PDF_MEDIA_TYPE, pdf("First", "Last"), {"max_pdf_pages": 1}, "PDF_PAGE_LIMIT"),
        (
            PDF_MEDIA_TYPE,
            pdf("Text too long"),
            {"max_extracted_text_bytes": 2},
            "EXTRACTED_TEXT_LIMIT",
        ),
        (DOCX_MEDIA_TYPE, word_document(), {"max_docx_entries": 1}, "DOCX_ENTRY_LIMIT"),
        (
            DOCX_MEDIA_TYPE,
            word_document(),
            {"max_docx_expanded_bytes": 100},
            "DOCX_EXPANSION_LIMIT",
        ),
        (DOCX_MEDIA_TYPE, word_document(), {"max_extracted_text_bytes": 2}, "EXTRACTED_TEXT_LIMIT"),
        (PDF_MEDIA_TYPE, b"anything", {"max_upload_bytes": 2}, "UPLOAD_TOO_LARGE"),
    ],
)
def test_invalid_and_excessive_inputs_fail_safely(
    media: str, content: bytes, overrides: dict[str, Any], code: str
) -> None:
    with pytest.raises(ProcessingError) as captured:
        extractor_for(media, configured(**overrides)).extract(content)
    assert captured.value.code == code
    assert captured.value.retryable is False
    assert "private" not in str(captured.value)


def test_recoverable_page_failure_is_partial_but_unexpected_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = extraction._pdf_page_blocks

    def broken(page: Any) -> tuple[list[Any], bool]:
        if page.number == 1:
            raise pymupdf.FileDataError("private parser error")
        return original(page)

    monkeypatch.setattr(extraction, "_pdf_page_blocks", broken)
    result = extractor_for(PDF_MEDIA_TYPE, configured()).extract(pdf("Good", "Broken", "Later"))
    assert result.coverage == "partial"
    assert [block.locator["page"] for block in result.blocks] == [1, 3]
    assert result.warnings[0].code == "PAGE_EXTRACTION_FAILED"

    def unexpected(page: Any) -> tuple[list[Any], bool]:
        raise RuntimeError("private structural error")

    monkeypatch.setattr(extraction, "_pdf_page_blocks", unexpected)
    with pytest.raises(ProcessingError, match="PDF text extraction failed") as captured:
        extractor_for(PDF_MEDIA_TYPE, configured()).extract(pdf("Good"))
    assert captured.value.code == "EXTRACTION_FAILED"


def test_actual_archive_read_limit_precedes_docx_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    content = BytesIO()
    with ZipFile(content, "w", ZIP_DEFLATED) as archive:
        archive.writestr("test", b"x")
    original_info = ZipFile.infolist

    def misleading_info(archive: ZipFile) -> list[Any]:
        entries = original_info(archive)
        for entry in entries:
            entry.file_size = 0
        return entries

    monkeypatch.setattr(ZipFile, "infolist", misleading_info)
    monkeypatch.setattr(ZipFile, "open", lambda *args, **kwargs: BytesIO(b"x" * 101))
    with pytest.raises(ProcessingError) as captured:
        extractor_for(DOCX_MEDIA_TYPE, configured(max_docx_expanded_bytes=100)).extract(
            content.getvalue()
        )
    assert captured.value.code == "DOCX_EXPANSION_LIMIT"


def test_unsupported_format_and_empty_docx() -> None:
    with pytest.raises(ProcessingError) as captured:
        extractor_for("text/html", configured())
    assert captured.value.code == "UNSUPPORTED_FORMAT"
    stream = BytesIO()
    docx.Document().save(stream)
    with pytest.raises(ProcessingError) as captured:
        extractor_for(DOCX_MEDIA_TYPE, configured()).extract(stream.getvalue())
    assert captured.value.code == "NO_EXTRACTABLE_TEXT"
