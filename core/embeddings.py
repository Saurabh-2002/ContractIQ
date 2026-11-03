"""Text embeddings using a pretrained Hugging Face Sentence Transformers model."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from config.settings import get_settings


class EmbeddingError(Exception):
    """The embedding model could not be loaded or used."""


class Embedder(Protocol):
    """Anything that can turn text into normalised vectors."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HuggingFaceEmbedder:
    """Wraps ``sentence-transformers/all-MiniLM-L6-v2``.

    The model is loaded on first use, so the app opens quickly. Documents and
    queries must use the same model, or their vectors cannot be compared.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    "The embedding model could not be loaded. The first run "
                    "needs internet access to download it."
                ) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Return one float32 vector of shape (dimension,)."""
        if not text or not text.strip():
            raise ValueError("The query is empty.")
        return self.embed_documents([text.strip()])[0]
