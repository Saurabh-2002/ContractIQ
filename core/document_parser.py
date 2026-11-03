"""PDF validation, text extraction and cleaning using PyMuPDF."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pymupdf


class DocumentParseError(Exception):
    """Base error. Its message is safe to show to the user."""


class InvalidFileTypeError(DocumentParseError):
    """The file is not a PDF."""


class CorruptedPDFError(DocumentParseError):
    """The file claims to be a PDF but cannot be read."""


class EncryptedPDFError(DocumentParseError):
    """The PDF requires a password."""


class EmptyPDFError(DocumentParseError):
    """The PDF has no extractable text."""


@dataclass(frozen=True)
class PageText:
    """Cleaned text of one page. ``page_number`` is 1-based."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Result of parsing one PDF."""

    doc_id: str
    filename: str
    page_count: int
    pages: list[PageText] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)


def compute_file_hash(data: bytes) -> str:
    """Return a SHA-256 hash, used as the document ID and to detect duplicates."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Drop any directory part, handling both Windows and POSIX separators."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name or "unnamed.pdf"


def validate_pdf_bytes(filename: str, data: bytes) -> None:
    """Check the extension and the PDF signature."""
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileTypeError(f"{filename}: only PDF files are supported.")
    if not data:
        raise EmptyPDFError(f"{filename}: the file is empty.")
    if b"%PDF-" not in data[:1024]:
        raise CorruptedPDFError(
            f"{filename}: this is not a valid PDF, or the file is damaged."
        )


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping paragraph breaks.

    Paragraph breaks are kept because the chunker prefers to split on them.
    """
    text = text.replace("\x00", "").replace("­", "").replace("\r\n", "\n")
    text = text.replace("\r", "\n").replace("\f", "\n")
    # Rejoin words hyphenated across a line break: "pay-\nment" -> "payment".
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    """Extract cleaned, page-level text from PDF bytes.

    Raises:
        DocumentParseError: A subclass describes why the file cannot be used.
    """
    safe_name = sanitize_filename(filename)
    validate_pdf_bytes(safe_name, data)

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError(
            f"{safe_name}: this PDF is damaged and could not be opened."
        ) from exc

    try:
        if document.needs_pass:
            raise EncryptedPDFError(
                f"{safe_name}: this PDF is password-protected. "
                "Remove the password and upload it again."
            )

        page_count = document.page_count
        if page_count == 0:
            raise CorruptedPDFError(f"{safe_name}: this PDF has no readable pages.")

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index in range(page_count):
            try:
                raw_text = document.load_page(index).get_text("text")
            except Exception as exc:
                raise CorruptedPDFError(
                    f"{safe_name}: page {index + 1} could not be read."
                ) from exc

            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append(PageText(page_number=index + 1, text=cleaned))
            else:
                empty_pages.append(index + 1)
    finally:
        document.close()

    if not pages:
        raise EmptyPDFError(
            f"{safe_name}: no text was found. Scanned PDFs need OCR, "
            "which ContractIQ does not support."
        )

    return ParsedDocument(
        doc_id=compute_file_hash(data),
        filename=safe_name,
        page_count=page_count,
        pages=pages,
        empty_pages=empty_pages,
    )
