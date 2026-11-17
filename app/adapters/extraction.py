"""Bounded native-text extraction with explicit source and capability coverage."""

import unicodedata
from io import BytesIO
from typing import Any
from zipfile import BadZipFile, ZipFile

import docx
import pymupdf
from celery.exceptions import SoftTimeLimitExceeded
from docx.oxml.ns import qn

from app.core.config import Settings
from app.domain.documents import (
    Coverage,
    ExtractionResult,
    ExtractionWarning,
    Extractor,
    Locator,
    ProcessingError,
    TextBlock,
)

NORMALIZATION_VERSION = "nfc-v1"
PDF_MEDIA_TYPE = "application/pdf"
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class _Blocks:
    """Offsets address the normalized document, including explicit separators."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.blocks: list[TextBlock] = []
        self.bytes = 0
        self.characters = 0

    def append(self, text: str, locator: Locator) -> None:
        normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        normalized = normalized.strip()
        if not normalized:
            return
        separator = 2 if self.blocks else 0
        self.bytes += len(normalized.encode("utf-8")) + separator
        if self.bytes > self.settings.max_extracted_text_bytes:
            raise ProcessingError("EXTRACTED_TEXT_LIMIT", "Extracted text exceeds the byte limit.")
        start = self.characters + separator
        self.blocks.append(
            TextBlock(
                f"block-{len(self.blocks) + 1}", normalized, locator, start, start + len(normalized)
            )
        )
        self.characters = start + len(normalized)

    def result(
        self,
        name: str,
        version: str,
        coverage: Coverage,
        warnings: list[ExtractionWarning],
        pages: int | None,
    ) -> ExtractionResult:
        if not self.blocks:
            raise ProcessingError(
                "NO_EXTRACTABLE_TEXT",
                "No useful native text was found. The file may be blank or need unsupported OCR.",
            )
        return ExtractionResult(
            "\n\n".join(block.text for block in self.blocks),
            tuple(self.blocks),
            name,
            version,
            coverage,
            tuple(warnings),
            pages,
        )


def _check_input(content: bytes, settings: Settings) -> None:
    if len(content) > settings.max_upload_bytes:
        raise ProcessingError("UPLOAD_TOO_LARGE", "The file exceeds the upload byte limit.")
    if not content:
        raise ProcessingError("INVALID_DOCUMENT", "The file is empty or corrupt.")


def _pdf_page_blocks(page: Any) -> tuple[list[Any], bool]:
    return page.get_text("blocks", sort=True), bool(page.get_images(full=True))


class PDFExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(self, content: bytes) -> ExtractionResult:
        _check_input(content, self.settings)
        try:
            return self._extract(content)
        except SoftTimeLimitExceeded:
            raise
        except ProcessingError:
            raise
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError):
            raise ProcessingError(
                "INVALID_PDF", "The PDF could not be read or is corrupt."
            ) from None
        except Exception:
            raise ProcessingError("EXTRACTION_FAILED", "PDF text extraction failed.") from None

    def _extract(self, content: bytes) -> ExtractionResult:
        blocks = _Blocks(self.settings)
        warnings: list[ExtractionWarning] = []
        coverage: Coverage = "complete_within_capability"
        with pymupdf.open(stream=content, filetype="pdf") as document:  # type: ignore[no-untyped-call]
            if not document.is_pdf:
                raise ProcessingError("INVALID_PDF", "The file is not a PDF document.")
            if document.needs_pass:
                raise ProcessingError("ENCRYPTED_PDF", "Password-protected PDFs are unsupported.")
            if len(document) > self.settings.max_pdf_pages:
                raise ProcessingError("PDF_PAGE_LIMIT", "The PDF exceeds the page limit.")
            for page_index in range(len(document)):
                locator: Locator = {"page": page_index + 1}
                try:
                    page_blocks, has_images = _pdf_page_blocks(document.load_page(page_index))
                except pymupdf.FileDataError:
                    coverage = "partial"
                    warnings.append(
                        ExtractionWarning(
                            "PAGE_EXTRACTION_FAILED",
                            "Native text could not be read on this page.",
                            locator,
                        )
                    )
                    continue
                before = len(blocks.blocks)
                for index, block in enumerate(page_blocks):
                    if block[6] == 0:
                        blocks.append(block[4], {"page": page_index + 1, "block": index + 1})
                if has_images:
                    if coverage != "partial":
                        coverage = "unknown"
                    warnings.append(
                        ExtractionWarning(
                            "IMAGES_NOT_INTERPRETED",
                            "Embedded images are not interpreted.",
                            locator,
                        )
                    )
                if len(blocks.blocks) == before:
                    if coverage != "partial":
                        coverage = "unknown"
                    warnings.append(
                        ExtractionWarning(
                            "TEXTLESS_PAGE",
                            "This page has no useful native text; it may be blank or visual.",
                            locator,
                        )
                    )
            return blocks.result("pymupdf", pymupdf.VersionBind, coverage, warnings, len(document))


def _validate_archive(content: bytes, settings: Settings) -> list[ExtractionWarning]:
    """Read every entry under an actual expansion cap before the DOCX parser runs."""
    warnings: list[ExtractionWarning] = []
    with ZipFile(BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > settings.max_docx_entries:
            raise ProcessingError("DOCX_ENTRY_LIMIT", "The DOCX exceeds the archive entry limit.")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ProcessingError("INVALID_DOCX", "The DOCX contains ambiguous archive entries.")
        if sum(entry.file_size for entry in entries) > settings.max_docx_expanded_bytes:
            raise ProcessingError(
                "DOCX_EXPANSION_LIMIT", "The DOCX exceeds the expanded byte limit."
            )
        expanded = 0
        for entry in entries:
            if entry.flag_bits & 1:
                raise ProcessingError("ENCRYPTED_DOCX", "Encrypted DOCX archives are unsupported.")
            with archive.open(entry) as stream:
                while data := stream.read(
                    min(65536, settings.max_docx_expanded_bytes - expanded + 1)
                ):
                    expanded += len(data)
                    if expanded > settings.max_docx_expanded_bytes:
                        raise ProcessingError(
                            "DOCX_EXPANSION_LIMIT", "The DOCX exceeds the expanded byte limit."
                        )
        omitted = (
            "word/header",
            "word/footer",
            "word/footnotes",
            "word/endnotes",
            "word/comments",
            "word/media/",
            "word/embeddings/",
        )
        if any(entry.filename.startswith(omitted) for entry in entries):
            warnings.append(
                ExtractionWarning(
                    "UNSUPPORTED_CONTENT",
                    "Headers, footers, notes, comments, images or embedded objects may be omitted.",
                )
            )
    return warnings


def _paragraph_text(element: Any) -> str:
    result: list[str] = []
    for node in element.iter():
        if node.tag == qn("w:t"):
            result.append(node.text or "")
        elif node.tag in {qn("w:br"), qn("w:cr")}:
            result.append("\n")
        elif node.tag == qn("w:tab"):
            result.append("\t")
    return "".join(result)


def _body_blocks(element: Any, output: _Blocks, path: str = "body") -> None:
    counts = {"paragraph": 0, "table": 0}
    for child in element:
        if child.tag == qn("w:p"):
            counts["paragraph"] += 1
            output.append(
                _paragraph_text(child), {"path": f"{path}/paragraph-{counts['paragraph']}"}
            )
        elif child.tag == qn("w:tbl"):
            counts["table"] += 1
            table_path = f"{path}/table-{counts['table']}"
            for row_index, row in enumerate(child.findall(qn("w:tr")), 1):
                for cell_index, cell in enumerate(row.findall(qn("w:tc")), 1):
                    _body_blocks(cell, output, f"{table_path}/row-{row_index}/cell-{cell_index}")


class DOCXExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(self, content: bytes) -> ExtractionResult:
        _check_input(content, self.settings)
        try:
            warnings = _validate_archive(content, self.settings)
            document = docx.Document(BytesIO(content))
            body = document.element.body
            # These XML constructs need richer layout/revision processing than native body text.
            unsupported = {
                qn(name)
                for name in (
                    "w:sdt",
                    "w:altChunk",
                    "w:drawing",
                    "w:pict",
                    "w:del",
                    "w:ins",
                    "w:txbxContent",
                    "w:fldSimple",
                    "w:fldChar",
                    "w:instrText",
                    "w:object",
                )
            }
            if any(
                node.tag in unsupported
                or isinstance(node.tag, str)
                and node.tag.startswith(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
                )
                for node in body.iter()
            ):
                warnings.append(
                    ExtractionWarning(
                        "UNSUPPORTED_LAYOUT",
                        "Some layout, revision, field or embedded content may be omitted.",
                    )
                )
            output = _Blocks(self.settings)
            _body_blocks(body, output)
            coverage: Coverage = "partial" if warnings else "complete_within_capability"
            return output.result("python-docx", docx.__version__, coverage, warnings, None)
        except SoftTimeLimitExceeded:
            raise
        except ProcessingError:
            raise
        except (BadZipFile, KeyError, ValueError):
            raise ProcessingError(
                "INVALID_DOCX", "The DOCX could not be read or is corrupt."
            ) from None
        except Exception:
            raise ProcessingError("EXTRACTION_FAILED", "DOCX text extraction failed.") from None


def extractor_for(media_type: str, settings: Settings) -> Extractor:
    if media_type == PDF_MEDIA_TYPE:
        return PDFExtractor(settings)
    if media_type == DOCX_MEDIA_TYPE:
        return DOCXExtractor(settings)
    raise ProcessingError("UNSUPPORTED_FORMAT", "Only PDF and DOCX documents are supported.")
