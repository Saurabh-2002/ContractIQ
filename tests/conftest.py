"""Shared fixtures.

Unit tests replace the embedding model and the LLM with deterministic fakes, so
they run offline and test ContractIQ's own logic. Tests marked ``integration``
use the real model: run them with ``pytest -m integration``.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from core.document_manager import DocumentManager

SAMPLES_DIR = PROJECT_ROOT / "data" / "sample_contracts"
DIMENSION = 384


class FakeEmbedder:
    """Deterministic bag-of-words embedder.

    Texts sharing words get similar vectors. This exercises retrieval plumbing
    but says nothing about the quality of real semantic search.
    """

    dimension = DIMENSION

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIMENSION, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % DIMENSION] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("The query is empty.")
        return self._vector(text)


class FakeLLM:
    """Returns scripted replies and records the prompts it received."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def make_pdf(pages: list[str]) -> bytes:
    """Build a PDF in which each list item is the text of one page."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings() -> Settings:
    # FakeEmbedder's scores are not cosine-calibrated, so the relevance floor is
    # disabled here and tested on its own in test_rag_pipeline.py.
    return Settings(gemini_api_key="test-key", min_relevance_score=-1.0)


@pytest.fixture
def manager(tmp_path, embedder, settings) -> DocumentManager:
    return DocumentManager(embedder, store_dir=tmp_path, settings=settings)


@pytest.fixture
def sample_bytes() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in SAMPLES_DIR.glob("*.pdf")}


@pytest.fixture
def loaded_manager(manager, sample_bytes) -> DocumentManager:
    for name, data in sample_bytes.items():
        assert manager.ingest(name, data).status == "indexed"
    return manager
