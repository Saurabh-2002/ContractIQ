"""Retrieval-augmented generation and analysis context construction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from config.settings import Settings, get_settings
from core.document_manager import DocumentRecord
from core.embeddings import Embedder
from core.llm_client import LLMClient
from core.vector_store import FaissVectorStore, SearchResult
from utils.prompts import NOT_FOUND_TOKEN, RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

NOT_FOUND_MESSAGE = "This information was not found in the selected documents."

_CITATION_PATTERN = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.IGNORECASE)

ANALYSIS_QUERIES = [
    "contract title and parties to this agreement",
    "effective date and commencement",
    "term and duration of the agreement",
    "payment terms, fees, invoices and late payment",
    "renewal and automatic renewal",
    "termination and notice period",
    "confidentiality obligations",
    "penalty, liability, indemnity and damages",
]


@dataclass(frozen=True)
class Citation:
    """A source shown under an answer. It always comes from retrieval."""

    label: str
    filename: str
    page_number: int
    text: str
    score: float


@dataclass(frozen=True)
class RagAnswer:
    """``status`` is ``answered``, ``not_found`` or ``uncited``."""

    answer: str
    status: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentContext:
    """Text prepared for analysis. ``is_partial`` means some was left out."""

    text: str
    is_partial: bool


def escape_context(text: str) -> str:
    """Stop document text from closing the context block in the prompt."""
    return re.sub(r"<(/?)\s*document_context", r"(\1document_context", text, flags=re.IGNORECASE)


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks with labels the model can cite."""
    blocks = []
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[S{number}] file: {chunk.filename} | page: {chunk.page_number}\n"
            f"{escape_context(chunk.text)}"
        )
    return "\n\n".join(blocks)


def resolve_citations(
    answer: str, results: list[SearchResult]
) -> tuple[str, list[Citation]]:
    """Map the model's labels to retrieved chunks and drop invalid labels."""
    used: list[int] = []

    def replace(match: re.Match) -> str:
        valid = []
        for label in re.split(r"\s*,\s*", match.group(1).upper()):
            number = int(label[1:])
            if 1 <= number <= len(results):
                valid.append(label)
                if number not in used:
                    used.append(number)
        return f"[{', '.join(valid)}]" if valid else ""

    cleaned = _CITATION_PATTERN.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip()
    citations = [_to_citation(number, results[number - 1]) for number in sorted(used)]
    return cleaned, citations


def _to_citation(number: int, result: SearchResult) -> Citation:
    return Citation(
        label=f"S{number}",
        filename=result.chunk.filename,
        page_number=result.chunk.page_number,
        text=result.chunk.text,
        score=result.score,
    )


class RagPipeline:
    """Question -> embedding -> FAISS -> top chunks -> LLM -> cited answer."""

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.settings = settings or get_settings()

    def retrieve(
        self, question: str, doc_ids: Iterable[str] | None = None
    ) -> list[SearchResult]:
        """Return the top chunks that pass the relevance floor."""
        results = self.store.search(
            self.embedder.embed_query(question), self.settings.top_k, doc_ids
        )
        return [r for r in results if r.score >= self.settings.min_relevance_score]

    def ask(
        self, question: str, doc_ids: Iterable[str] | None = None
    ) -> RagAnswer:
        results = self.retrieve(question, doc_ids)
        if not results:
            # Nothing relevant was retrieved, so the LLM is never called.
            return RagAnswer(NOT_FOUND_MESSAGE, "not_found")

        prompt = RAG_USER_TEMPLATE.format(
            context=build_context(results), question=question.strip()
        )
        reply = self.llm.generate(RAG_SYSTEM_PROMPT, prompt).strip()

        answer, citations = resolve_citations(reply, results)
        if NOT_FOUND_TOKEN in reply and not citations:
            return RagAnswer(NOT_FOUND_MESSAGE, "not_found")

        if citations:
            answer = answer.replace(NOT_FOUND_TOKEN, "").strip()
            return RagAnswer(answer, "answered", citations)

        # The model cited nothing valid. Show what was retrieved, flagged as
        # uncited, so the answer is never presented as verified.
        retrieved = [_to_citation(i, r) for i, r in enumerate(results, start=1)]
        return RagAnswer(answer, "uncited", retrieved)


def format_pages(pages: Iterable[dict]) -> str:
    """Format pages with the [Page N] markers that analysis prompts rely on."""
    return "\n\n".join(
        f"[Page {page['page_number']}]\n{escape_context(page['text'])}"
        for page in pages
    )


def build_document_context(
    record: DocumentRecord,
    embedder: Embedder,
    store: FaissVectorStore,
    settings: Settings | None = None,
) -> DocumentContext:
    """Prepare a contract for analysis.

    Most contracts are sent whole. A very long one is reduced to its first page
    plus the passages most relevant to each analysis topic, and is flagged as
    partial so that "not found" is not mistaken for "not in the contract".
    """
    settings = settings or get_settings()
    full_text = format_pages(record.pages)
    if len(full_text) <= settings.max_analysis_chars:
        return DocumentContext(full_text, is_partial=False)

    selected: dict[str, SearchResult] = {}
    for query in ANALYSIS_QUERIES:
        vector = embedder.embed_query(query)
        for result in store.search(vector, settings.top_k, [record.doc_id]):
            selected.setdefault(result.chunk.chunk_id, result)

    ordered = sorted(selected.values(), key=lambda r: r.chunk.chunk_index)
    parts = [format_pages(record.pages[:1])]
    parts += [
        f"[Page {r.chunk.page_number}]\n{escape_context(r.chunk.text)}"
        for r in ordered
    ]

    kept: list[str] = []
    length = 0
    for part in parts:
        if length + len(part) > settings.max_analysis_chars:
            break
        kept.append(part)
        length += len(part) + 2
    return DocumentContext("\n\n".join(kept), is_partial=True)
