"""Local FAISS vector store with chunk metadata and document filtering."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from core.text_chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "chunks.json"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity to the query."""

    chunk: Chunk
    score: float


class FaissVectorStore:
    """Exact cosine-similarity search over document chunks.

    Vectors are unit length, so inner product equals cosine similarity. An ID
    map links each FAISS vector to its chunk and allows vectors to be removed.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._chunks: dict[int, Chunk] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def doc_ids(self) -> set[str]:
        return {chunk.doc_id for chunk in self._chunks.values()}

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk needs exactly one vector.")
        if not chunks:
            return
        matrix = self._as_matrix(vectors)
        ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
        for vector_id, chunk in zip(ids.tolist(), chunks):
            self._chunks[vector_id] = chunk
        self._next_id += len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        doc_ids: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        """Return the best chunks, optionally limited to some documents.

        ``doc_ids=None`` searches everything; an empty collection matches
        nothing. Filtering happens after an exhaustive search, which is simple
        and exact at this MVP's scale.
        """
        allowed = None if doc_ids is None else set(doc_ids)
        if self.size == 0 or top_k <= 0 or allowed == set():
            return []

        query = self._as_matrix(np.asarray(query_vector).reshape(1, -1))
        limit = self.size if allowed is not None else min(top_k, self.size)
        scores, ids = self._index.search(query, limit)

        results: list[SearchResult] = []
        for score, vector_id in zip(scores[0].tolist(), ids[0].tolist()):
            chunk = self._chunks.get(vector_id)
            if chunk is None or (allowed is not None and chunk.doc_id not in allowed):
                continue
            results.append(SearchResult(chunk=chunk, score=float(score)))
            if len(results) == top_k:
                break
        return results

    def chunks_for_document(self, doc_id: str) -> list[Chunk]:
        chunks = [c for c in self._chunks.values() if c.doc_id == doc_id]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def remove_document(self, doc_id: str) -> int:
        """Delete a document's vectors and return how many were removed."""
        ids = [i for i, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        if not ids:
            return 0
        self._index.remove_ids(np.array(ids, dtype=np.int64))
        for vector_id in ids:
            del self._chunks[vector_id]
        return len(ids)

    def save(self, directory: Path) -> None:
        """Persist the store.

        The index is serialised to bytes and written by Python because
        ``faiss.write_index`` can fail on non-ASCII Windows paths.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INDEX_FILENAME).write_bytes(
            faiss.serialize_index(self._index).tobytes()
        )
        payload = {
            "dimension": self.dimension,
            "next_id": self._next_id,
            "chunks": {str(k): v.to_dict() for k, v in self._chunks.items()},
        }
        (directory / METADATA_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, dimension: int) -> FaissVectorStore:
        """Load a saved store, or return an empty one if none is usable."""
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        store = cls(dimension)
        if not index_path.exists() or not metadata_path.exists():
            return store

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            buffer = np.frombuffer(index_path.read_bytes(), dtype=np.uint8).copy()
            index = faiss.deserialize_index(buffer)
            chunks = {
                int(k): Chunk.from_dict(v) for k, v in payload["chunks"].items()
            }
            if payload["dimension"] != dimension or index.ntotal != len(chunks):
                raise ValueError("Saved index does not match its metadata.")
        except Exception:
            logger.warning("Saved vector store is unusable; starting empty.")
            return cls(dimension)

        store._index = index
        store._chunks = chunks
        store._next_id = int(payload["next_id"])
        return store

    def _as_matrix(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors of dimension {self.dimension}, "
                f"got shape {matrix.shape}."
            )
        return matrix
