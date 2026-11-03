import numpy as np
import pytest
from conftest import DIMENSION, make_pdf

from core.document_manager import DocumentManager
from core.text_chunker import Chunk
from core.vector_store import FaissVectorStore


def _chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(f"{doc_id}-{i}", doc_id, f"{doc_id}.pdf", i + 1, i, text)
        for i, text in enumerate(texts)
    ]


def _store(embedder) -> FaissVectorStore:
    store = FaissVectorStore(DIMENSION)
    for doc_id, texts in {
        "a": ["payment within thirty days", "renewal every year"],
        "b": ["payment within sixty days", "confidential information"],
    }.items():
        chunks = _chunks(doc_id, texts)
        store.add_chunks(chunks, embedder.embed_documents(texts))
    return store


def test_search_ranks_the_closest_chunk_first(embedder):
    results = _store(embedder).search(embedder.embed_query("confidential information"))
    assert results[0].chunk.text == "confidential information"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_top_k_limits_results(embedder):
    assert len(_store(embedder).search(embedder.embed_query("payment"), top_k=2)) == 2


def test_search_stays_inside_selected_documents(embedder):
    results = _store(embedder).search(
        embedder.embed_query("payment"), top_k=4, doc_ids=["b"]
    )
    assert {r.chunk.doc_id for r in results} == {"b"}
    assert len(results) == 2


def test_empty_selection_returns_nothing(embedder):
    store = _store(embedder)
    assert store.search(embedder.embed_query("payment"), doc_ids=[]) == []


def test_empty_store_returns_nothing(embedder):
    store = FaissVectorStore(DIMENSION)
    assert store.search(embedder.embed_query("payment")) == []


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError):
        FaissVectorStore(DIMENSION).add_chunks(
            _chunks("a", ["x"]), np.ones((1, 5), dtype=np.float32)
        )


def test_mismatched_lengths_are_rejected(embedder):
    with pytest.raises(ValueError):
        FaissVectorStore(DIMENSION).add_chunks(
            _chunks("a", ["x", "y"]), embedder.embed_documents(["x"])
        )


def test_removal_deletes_only_that_document(embedder):
    store = _store(embedder)
    assert store.remove_document("a") == 2
    assert store.size == 2
    assert store.doc_ids == {"b"}
    results = store.search(embedder.embed_query("renewal"), top_k=4)
    assert all(r.chunk.doc_id == "b" for r in results)


def test_ids_stay_correct_when_adding_after_removal(embedder):
    store = _store(embedder)
    store.remove_document("a")
    store.add_chunks(_chunks("c", ["brand new clause"]), embedder.embed_documents(["brand new clause"]))
    assert store.search(embedder.embed_query("brand new clause"))[0].chunk.doc_id == "c"


def test_save_and_load_round_trip(tmp_path, embedder):
    store = _store(embedder)
    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path, DIMENSION)
    assert loaded.size == 4
    query = embedder.embed_query("renewal every year")
    assert loaded.search(query)[0].chunk == store.search(query)[0].chunk


def test_damaged_store_starts_empty(tmp_path, embedder):
    _store(embedder).save(tmp_path)
    (tmp_path / "chunks.json").write_text("{broken", encoding="utf-8")
    assert FaissVectorStore.load(tmp_path, DIMENSION).size == 0


# --- DocumentManager ---


def test_multiple_uploads_are_indexed(loaded_manager):
    assert len(loaded_manager.documents()) == 3
    assert loaded_manager.total_chunks == sum(
        d.chunk_count for d in loaded_manager.documents()
    )


def test_duplicate_is_not_processed_twice(loaded_manager, sample_bytes):
    before = loaded_manager.total_chunks
    data = sample_bytes["Employment_Agreement.pdf"]
    result = loaded_manager.ingest("renamed_copy.pdf", data)
    assert result.status == "duplicate"
    assert loaded_manager.total_chunks == before


def test_bad_file_fails_without_raising(manager):
    result = manager.ingest("bad.pdf", b"not a pdf")
    assert result.status == "failed"
    assert manager.documents() == []


def test_oversized_file_is_refused(manager, settings):
    data = b"%PDF-" + b"0" * (settings.max_upload_mb * 1024 * 1024 + 1)
    assert manager.ingest("big.pdf", data).status == "failed"


def test_removal_updates_library_and_index(loaded_manager):
    record = loaded_manager.documents()[0]
    before = loaded_manager.total_chunks
    assert loaded_manager.remove(record.doc_id) is True
    assert loaded_manager.get(record.doc_id) is None
    assert loaded_manager.total_chunks == before - record.chunk_count
    assert loaded_manager.remove(record.doc_id) is False


def test_removed_document_can_be_uploaded_again(manager):
    data = make_pdf(["A clause long enough to be indexed."])
    first = manager.ingest("again.pdf", data)
    manager.remove(first.record.doc_id)
    assert manager.ingest("again.pdf", data).status == "indexed"


def test_library_survives_restart(tmp_path, embedder, settings, sample_bytes):
    first = DocumentManager(embedder, store_dir=tmp_path, settings=settings)
    data = sample_bytes["Vendor_Service_Agreement.pdf"]
    first.ingest("Vendor_Service_Agreement.pdf", data)

    second = DocumentManager(embedder, store_dir=tmp_path, settings=settings)
    assert [d.filename for d in second.documents()] == ["Vendor_Service_Agreement.pdf"]
    assert second.total_chunks == first.total_chunks
    assert second.ingest("copy.pdf", data).status == "duplicate"


def test_removal_clears_text_from_disk(tmp_path, embedder, settings):
    manager = DocumentManager(embedder, store_dir=tmp_path, settings=settings)
    result = manager.ingest("x.pdf", make_pdf(["Uniquemarker clause text here."]))
    manager.remove(result.record.doc_id)
    for path in tmp_path.iterdir():
        assert b"Uniquemarker" not in path.read_bytes()
