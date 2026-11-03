import pytest
from conftest import make_pdf

from core.document_parser import (
    CorruptedPDFError,
    EmptyPDFError,
    InvalidFileTypeError,
    clean_text,
    compute_file_hash,
    parse_pdf,
    sanitize_filename,
)


def test_extracts_text_from_every_sample(sample_bytes):
    assert len(sample_bytes) == 3
    for name, data in sample_bytes.items():
        parsed = parse_pdf(data, name)
        assert parsed.filename == name
        assert parsed.page_count >= 2
        assert parsed.pages[0].page_number == 1
        assert "FICTIONAL DEMONSTRATION DOCUMENT" in parsed.pages[0].text


def test_page_numbers_match_source_pages():
    parsed = parse_pdf(make_pdf(["alpha clause", "bravo clause"]), "two.pdf")
    assert [(p.page_number, p.text) for p in parsed.pages] == [
        (1, "alpha clause"),
        (2, "bravo clause"),
    ]


def test_blank_pages_are_reported_and_numbering_is_kept():
    parsed = parse_pdf(make_pdf(["first", "", "third"]), "gap.pdf")
    assert parsed.page_count == 3
    assert parsed.empty_pages == [2]
    assert [p.page_number for p in parsed.pages] == [1, 3]


def test_rejects_non_pdf_extension():
    with pytest.raises(InvalidFileTypeError):
        parse_pdf(b"%PDF-1.7", "contract.docx")


def test_rejects_zero_byte_file():
    with pytest.raises(EmptyPDFError):
        parse_pdf(b"", "empty.pdf")


def test_rejects_file_without_pdf_signature():
    with pytest.raises(CorruptedPDFError):
        parse_pdf(b"this is plain text", "fake.pdf")


def test_rejects_corrupted_pdf_body():
    with pytest.raises(CorruptedPDFError):
        parse_pdf(b"%PDF-1.7\n" + b"\x00garbage" * 50, "bad.pdf")


def test_rejects_pdf_without_text():
    with pytest.raises(EmptyPDFError, match="OCR"):
        parse_pdf(make_pdf(["", ""]), "scan.pdf")


def test_clean_text():
    raw = "Pay-\nment   is  due.\r\n\r\n\r\n\r\nNext\x00 para\u00ad"
    assert clean_text(raw) == "Payment is due.\n\nNext para"


def test_clean_text_keeps_real_hyphens():
    assert clean_text("twenty-four months") == "twenty-four months"


def test_sanitize_filename_strips_paths():
    assert sanitize_filename("C:\\Users\\me\\deal.pdf") == "deal.pdf"
    assert sanitize_filename("../../deal.pdf") == "deal.pdf"


def test_file_hash_is_stable_and_content_based():
    assert compute_file_hash(b"a") == compute_file_hash(b"a")
    assert compute_file_hash(b"a") != compute_file_hash(b"b")
