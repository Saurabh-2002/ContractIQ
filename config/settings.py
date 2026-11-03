"""Central configuration for ContractIQ.

Every tunable value lives here so that behaviour can be changed in one place.
Paths are derived from this file's location, so the project does not depend on
where it was extracted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# PyTorch and FAISS can both load an OpenMP runtime on Windows, which aborts the
# process with "OMP: Error #15". These must be set before either is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

API_KEY_PLACEHOLDER = "your_gemini_api_key_here"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 20
    max_upload_mb: int = 25

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Retrieval
    top_k: int = 4
    # Cosine similarity below which a chunk is treated as unrelated. This is a
    # conservative floor; see LIMITATIONS_AND_FUTURE_SCOPE.md on calibration.
    min_relevance_score: float = 0.15

    # Contract analysis
    max_analysis_chars: int = 60_000
    max_risk_findings: int = 12

    # LLM
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # Local storage
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    sample_contracts_dir: Path = PROJECT_ROOT / "data" / "sample_contracts"

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)


def get_settings() -> Settings:
    """Build settings from the environment."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key == API_KEY_PLACEHOLDER:
        api_key = ""
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    return Settings(gemini_api_key=api_key, gemini_model=model)
