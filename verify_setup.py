"""Check the real ContractIQ stack on your machine and record what happened.

Run from the project folder, inside the virtual environment:

    python verify_setup.py

Results are written to docs/LOCAL_VERIFICATION_RESULTS.md. Checks that need
Gemini are marked SKIP when no API key is configured.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_FILE = PROJECT_ROOT / "docs" / "LOCAL_VERIFICATION_RESULTS.md"

# Question, the document that should be retrieved, text the top passages should
# contain. These are checks on real semantic search over the sample contracts.
RETRIEVAL_CASES = [
    ("How many days does the client have to pay an invoice?",
     "Vendor_Service_Agreement.pdf", "forty-five"),
    ("How much notice is needed to stop the vendor contract renewing?",
     "Vendor_Service_Agreement.pdf", "ninety"),
    ("What is the annual licence fee for the software?",
     "Software_Licensing_Agreement.pdf", "96,000"),
    ("What is the cap on the licensor's liability?",
     "Software_Licensing_Agreement.pdf", "three (3) months"),
    ("What is the employee's notice period after probation?",
     "Employment_Agreement.pdf", "ninety (90)"),
    ("Does the employee have a non-compete restriction?",
     "Employment_Agreement.pdf", "competes"),
]

results: list[tuple[str, str, str]] = []
state: dict = {}


def check(name: str, function) -> None:
    """Run one check and record PASS, FAIL or SKIP."""
    try:
        detail = function()
        if detail == "SKIP":
            status, detail = "SKIP", "No Gemini API key configured."
        else:
            status = "PASS"
    except Exception as exc:  # Every failure should be reported, not fatal.
        status = "FAIL"
        detail = f"{type(exc).__name__}: {exc}"
        if "--debug" in sys.argv:
            traceback.print_exc()
    results.append((status, name, str(detail or "")))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def check_python():
    version = sys.version_info
    assert version >= (3, 10), "Python 3.10 or later is required."
    return platform.python_version()


def check_imports():
    from importlib import metadata

    names = ["streamlit", "pymupdf", "langchain-core", "langchain-text-splitters",
             "langchain-google-genai", "sentence-transformers", "faiss-cpu"]
    import faiss  # noqa: F401
    import langchain_google_genai  # noqa: F401
    import langchain_text_splitters  # noqa: F401
    import pymupdf  # noqa: F401
    import sentence_transformers  # noqa: F401
    import streamlit  # noqa: F401

    return ", ".join(f"{name} {metadata.version(name)}" for name in names)


def check_parsing():
    from core.document_parser import parse_pdf
    from core.text_chunker import chunk_document

    samples = sorted((PROJECT_ROOT / "data" / "sample_contracts").glob("*.pdf"))
    assert len(samples) == 3, f"Expected 3 sample PDFs, found {len(samples)}."
    state["samples"] = samples
    summary = []
    for path in samples:
        parsed = parse_pdf(path.read_bytes(), path.name)
        chunks = chunk_document(parsed)
        assert chunks and all(len(c.text) <= 800 for c in chunks)
        assert {c.page_number for c in chunks} <= set(range(1, parsed.page_count + 1))
        summary.append(f"{path.name}: {parsed.page_count} pages, {len(chunks)} chunks")
    return "; ".join(summary)


def check_ingestion():
    from config.settings import get_settings
    from core.document_manager import DocumentManager
    from core.embeddings import HuggingFaceEmbedder

    settings = get_settings()
    state["settings"] = settings
    state["temp"] = tempfile.TemporaryDirectory()
    manager = DocumentManager(
        HuggingFaceEmbedder(), Path(state["temp"].name), settings
    )
    for path in state["samples"]:
        result = manager.ingest(path.name, path.read_bytes())
        assert result.status == "indexed", result.message
    state["manager"] = manager
    return f"{manager.total_chunks} vectors in FAISS"


def check_retrieval():
    from config.settings import Settings
    from core.rag_pipeline import RagPipeline

    manager = state["manager"]
    pipeline = RagPipeline(manager.embedder, manager.store, None, Settings())
    misses = []
    lines = []
    for question, filename, expected in RETRIEVAL_CASES:
        retrieved = pipeline.retrieve(question)
        top = retrieved[0].score if retrieved else 0.0
        lines.append(f"{top:.2f}")
        found = any(
            r.chunk.filename == filename and expected in r.chunk.text
            for r in retrieved
        )
        if not found:
            misses.append(question)
    assert not misses, f"Not retrieved in the top 4: {misses}"
    return f"{len(RETRIEVAL_CASES)}/{len(RETRIEVAL_CASES)} found. Top scores: {', '.join(lines)}"


def check_relevance_floor():
    from core.rag_pipeline import RagPipeline

    manager = state["manager"]
    pipeline = RagPipeline(manager.embedder, manager.store, None, state["settings"])
    question = "What is the boiling point of water on Mars?"
    results_found = manager.store.search(manager.embedder.embed_query(question), 1)
    state["unrelated_score"] = results_found[0].score
    retrieved = pipeline.retrieve(question)
    return (
        f"Unrelated question scored {results_found[0].score:.2f}; "
        f"{len(retrieved)} chunk(s) passed the floor. Compare with the scores above."
    )


def check_persistence():
    from core.document_manager import DocumentManager
    from core.embeddings import HuggingFaceEmbedder

    manager = state["manager"]
    reloaded = DocumentManager(
        HuggingFaceEmbedder(), Path(state["temp"].name), state["settings"]
    )
    assert reloaded.total_chunks == manager.total_chunks
    record = reloaded.documents()[0]
    reloaded.remove(record.doc_id)
    assert reloaded.total_chunks == manager.total_chunks - record.chunk_count
    return "Reload and document removal both worked."


def _llm():
    from core.llm_client import create_llm_client

    if not state["settings"].has_api_key:
        return None
    return create_llm_client(state["settings"])


def check_gemini_answer():
    from core.rag_pipeline import RagPipeline

    llm = _llm()
    if llm is None:
        return "SKIP"
    manager = state["manager"]
    pipeline = RagPipeline(manager.embedder, manager.store, llm, state["settings"])
    answer = pipeline.ask("What is the annual licence fee for the software?")
    assert answer.status == "answered", f"Status was {answer.status}."
    assert "96,000" in answer.answer, "Expected USD 96,000 in the answer."
    assert any(
        c.filename == "Software_Licensing_Agreement.pdf" for c in answer.citations
    )
    return f"Cited {[(c.filename, c.page_number) for c in answer.citations]}"


def check_gemini_not_found():
    from core.rag_pipeline import RagPipeline

    llm = _llm()
    if llm is None:
        return "SKIP"
    manager = state["manager"]
    pipeline = RagPipeline(manager.embedder, manager.store, llm, state["settings"])
    answer = pipeline.ask("What is the arbitrator's hourly rate under the agreements?")
    assert answer.status == "not_found", f"Status was {answer.status}: {answer.answer}"
    return "Reported as not found."


def check_gemini_analysis():
    from core.rag_pipeline import build_document_context
    from features.clause_extraction import extract_clauses
    from features.contract_summary import generate_summary
    from features.risk_analysis import analyze_risks

    llm = _llm()
    if llm is None:
        return "SKIP"
    manager = state["manager"]
    record = next(
        d for d in manager.documents() if d.filename.startswith("Vendor")
    )
    context = build_document_context(
        record, manager.embedder, manager.store, state["settings"]
    )
    summary = generate_summary(llm, record, context)
    assert summary.text
    fields = extract_clauses(llm, record, context)
    found = sum(f.status == "found" for f in fields)
    assert found >= 7, f"Only {found} of 9 fields were verified."
    findings = analyze_risks(llm, record, context, state["settings"])
    assert findings, "No risk findings were returned."
    verified = sum(f.verified for f in findings)
    return (
        f"{found}/9 fields verified; {len(findings)} findings, {verified} verified."
    )


def write_results() -> None:
    lines = [
        "# Local Verification Results",
        "",
        f"Generated by `verify_setup.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        f"Platform: {platform.platform()}. Python {platform.python_version()}.",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    for status, name, detail in results:
        lines.append(f"| {status} | {name} | {detail.replace('|', '/')} |")
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from config.settings import get_settings  # noqa: F401  Sets process options.

    check("Python version", check_python)
    check("Real packages import", check_imports)
    check("PyMuPDF parsing and chunking", check_parsing)
    check("Hugging Face embeddings into FAISS", check_ingestion)

    if "manager" in state:
        check("Semantic retrieval on sample contracts", check_retrieval)
        check("Relevance floor on an unrelated question", check_relevance_floor)
        check("Persistence and removal", check_persistence)
        check("Gemini grounded answer with citation", check_gemini_answer)
        check("Gemini reports missing information", check_gemini_not_found)
        check("Gemini summary, clauses and risks", check_gemini_analysis)

    if "temp" in state:
        state["temp"].cleanup()

    write_results()
    failed = sum(status == "FAIL" for status, _, _ in results)
    skipped = sum(status == "SKIP" for status, _, _ in results)
    print(
        f"\n{len(results) - failed - skipped} passed, {failed} failed, "
        f"{skipped} skipped. Saved to {RESULTS_FILE.relative_to(PROJECT_ROOT)}."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
