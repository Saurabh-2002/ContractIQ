"""Split parsed documents into overlapping chunks that keep their page number."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from core.document_parser import ParsedDocument

# Tried in order, so splits fall on paragraphs before sentences before words.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage and its source location."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def build_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=(
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        ),
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_document(
    document: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a document one page at a time.

    Splitting per page guarantees that every chunk has exactly one true page
    number. The trade-off is that a clause running across a page break becomes
    two chunks, because overlap does not cross pages.
    """
    settings = get_settings()
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for page in document.pages:
        for text in splitter.split_text(page.text):
            text = text.strip()
            if len(text) < settings.min_chunk_chars:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id[:12]}-p{page.page_number}-c{index}",
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks
