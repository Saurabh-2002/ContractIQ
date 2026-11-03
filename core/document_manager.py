"""Coordinates ingestion, duplicate detection, removal and local persistence.

This keeps the ingestion workflow out of the Streamlit layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings, get_settings
from core.document_parser import DocumentParseError, parse_pdf, sanitize_filename
from core.embeddings import Embedder, EmbeddingError
from core.text_chunker import chunk_document
from core.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

DOCUMENTS_FILENAME = "documents.json"


@dataclass
class DocumentRecord:
    """A processed contract, as listed in the Document Library."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    pages: list[dict] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)
    uploaded_at: str = ""
    status: str = "Indexed"


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one upload: ``indexed``, ``duplicate`` or ``failed``."""

    status: str
    filename: str
    message: str
    record: DocumentRecord | None = None


class DocumentManager:
    """Owns the document list and the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.store_dir = store_dir or self.settings.vector_store_dir
        self.store = FaissVectorStore.load(
            self.store_dir, self.settings.embedding_dimension
        )
        self._documents = self._load_documents()

    def documents(self) -> list[DocumentRecord]:
        """Return documents, most recently uploaded first."""
        return sorted(
            self._documents.values(), key=lambda d: d.uploaded_at, reverse=True
        )

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    @property
    def total_chunks(self) -> int:
        return self.store.size

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        """Run the ingestion pipeline for one file. Never raises for bad input."""
        name = sanitize_filename(filename)
        limit_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > limit_bytes:
            return IngestResult(
                "failed",
                name,
                f"{name}: larger than the "
                f"{self.settings.max_upload_mb} MB upload limit.",
            )

        try:
            parsed = parse_pdf(data, name)
        except DocumentParseError as exc:
            return IngestResult("failed", name, str(exc))

        existing = self._documents.get(parsed.doc_id)
        if existing is not None:
            return IngestResult(
                "duplicate",
                name,
                f"{name}: already processed as {existing.filename}.",
                existing,
            )

        chunks = chunk_document(parsed)
        if not chunks:
            return IngestResult(
                "failed", name, f"{name}: there was not enough text to index."
            )

        try:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
        except EmbeddingError as exc:
            return IngestResult("failed", name, str(exc))

        self.store.add_chunks(chunks, vectors)
        record = DocumentRecord(
            doc_id=parsed.doc_id,
            filename=parsed.filename,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            pages=[asdict(page) for page in parsed.pages],
            empty_pages=parsed.empty_pages,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._documents[record.doc_id] = record
        self._save()

        message = f"{name}: indexed {len(chunks)} chunks."
        if parsed.empty_pages:
            pages = ", ".join(str(page) for page in parsed.empty_pages)
            message += f" No text found on page(s) {pages}."
        return IngestResult("indexed", name, message, record)

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the library, the index and local storage."""
        if doc_id not in self._documents:
            return False
        self.store.remove_document(doc_id)
        del self._documents[doc_id]
        self._save()
        return True

    def _save(self) -> None:
        self.store.save(self.store_dir)
        payload = [asdict(record) for record in self._documents.values()]
        (self.store_dir / DOCUMENTS_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _load_documents(self) -> dict[str, DocumentRecord]:
        path = self.store_dir / DOCUMENTS_FILENAME
        if not path.exists():
            return {}
        try:
            records = [
                DocumentRecord(**item)
                for item in json.loads(path.read_text(encoding="utf-8"))
            ]
        except (OSError, TypeError, ValueError):
            logger.warning("Saved document list is unusable; ignoring it.")
            return {}
        # Keep only documents that are actually searchable.
        indexed = self.store.doc_ids
        return {r.doc_id: r for r in records if r.doc_id in indexed}
