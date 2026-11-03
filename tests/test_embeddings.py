import numpy as np
import pytest

from core.embeddings import EmbeddingError, HuggingFaceEmbedder


class StubModel:
    def __init__(self):
        self.kwargs = None

    def encode(self, texts, **kwargs):
        self.kwargs = kwargs
        return np.ones((len(texts), 384), dtype=np.float64)


def _embedder_with_stub():
    embedder = HuggingFaceEmbedder()
    embedder._model = StubModel()
    return embedder


def test_uses_required_model_name():
    assert HuggingFaceEmbedder().model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_document_vectors_are_float32_and_normalised_on_request():
    embedder = _embedder_with_stub()
    vectors = embedder.embed_documents(["a", "b"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert embedder._model.kwargs["normalize_embeddings"] is True


def test_query_returns_single_vector():
    assert _embedder_with_stub().embed_query("notice period").shape == (384,)


def test_no_texts_need_no_model():
    assert HuggingFaceEmbedder().embed_documents([]).shape == (0, 384)


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(query):
    with pytest.raises(ValueError):
        _embedder_with_stub().embed_query(query)


def test_load_failure_gives_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise OSError("no model")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(EmbeddingError):
        HuggingFaceEmbedder().embed_documents(["text"])


@pytest.mark.integration
def test_real_model_produces_unit_vectors_and_semantic_ranking():
    embedder = HuggingFaceEmbedder()
    vectors = embedder.embed_documents(
        [
            "Either party may end this agreement with thirty days notice.",
            "The cafeteria menu changes every Tuesday.",
        ]
    )
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    query = embedder.embed_query("How can the contract be terminated?")
    assert float(query @ vectors[0]) > float(query @ vectors[1])
