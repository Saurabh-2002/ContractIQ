from conftest import make_pdf

from core.document_parser import PageText, ParsedDocument, parse_pdf
from core.text_chunker import Chunk, chunk_document


def _document(pages: list[str]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="a" * 64,
        filename="doc.pdf",
        page_count=len(pages),
        pages=[PageText(i, text) for i, text in enumerate(pages, start=1)],
    )


def test_every_sample_chunks_within_size_limit(sample_bytes):
    for name, data in sample_bytes.items():
        chunks = chunk_document(parse_pdf(data, name))
        assert len(chunks) > 3
        assert all(len(chunk.text) <= 800 for chunk in chunks)


def test_chunks_keep_required_metadata(sample_bytes):
    data = sample_bytes["Vendor_Service_Agreement.pdf"]
    parsed = parse_pdf(data, "Vendor_Service_Agreement.pdf")
    chunks = chunk_document(parsed)
    assert {c.filename for c in chunks} == {"Vendor_Service_Agreement.pdf"}
    assert {c.page_number for c in chunks} == {p.page_number for p in parsed.pages}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_text_comes_from_its_own_page():
    pages = [("alpha " * 300).strip(), ("bravo " * 300).strip()]
    for chunk in chunk_document(_document(pages)):
        word = "alpha" if chunk.page_number == 1 else "bravo"
        assert set(chunk.text.split()) == {word}


def test_long_text_overlaps_between_chunks():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_document(_document([text]))
    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        assert set(first.text.split()) & set(second.text.split())


def test_very_short_fragments_are_dropped():
    assert chunk_document(_document(["Page 4"])) == []


def test_chunk_round_trips_through_dict():
    chunk = chunk_document(parse_pdf(make_pdf(["A clause long enough to keep."]), "x.pdf"))[0]
    assert Chunk.from_dict(chunk.to_dict()) == chunk
