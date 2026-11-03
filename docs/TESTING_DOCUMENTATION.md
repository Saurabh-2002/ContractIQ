# Testing Documentation

## Read this first

**ContractIQ has not yet been run against its real libraries or against Gemini.**

The development workspace's network policy blocked PyPI, Hugging Face and the Gemini API. Streamlit, PyMuPDF, FAISS, LangChain, sentence-transformers and the Gemini client could not be installed, and no Gemini key was available.

Testing therefore covered ContractIQ's own logic, using temporary stand-ins that are not part of this project. Those results do not show that the application works with the real libraries. To close that gap, run `python verify_setup.py`, which records results in `docs/LOCAL_VERIFICATION_RESULTS.md`.

### How to read the results

| Label | Meaning |
|---|---|
| **Passed (real)** | Run with the real component |
| **Passed (stand-in)** | ContractIQ's logic passed, but a library or service was simulated |
| **Not run** | Not executed in any form |

### Test environment

All recorded results came from this environment on 3 November 2026:

- Linux, Python 3.13.18
- pytest 9.1.1, numpy 2.5.3, pandas 3.0.5, python-dotenv 1.2.3, reportlab 5.0.1 (real)

| Component | What stood in for it |
|---|---|
| PyMuPDF | `pypdfium2` for reading, `reportlab` for writing |
| FAISS | An exact NumPy implementation of the calls used |
| LangChain text splitter | A reimplementation of recursive splitting |
| Streamlit | A stub that runs control flow but renders nothing |
| `all-MiniLM-L6-v2` | A deterministic bag-of-words `FakeEmbedder` |
| Gemini | A scripted `FakeLLM` |

## Summary of actual results

| Check | Actual result |
|---|---|
| Unit tests | **102 passed, 0 failed**, with 5 integration tests deselected (0.60 s) |
| Integration tests | **Not run** (5 tests) |
| App walkthrough | **Passed (stand-in)**: all four pages |
| Mutation checks | **9 of 9 deliberate breaks detected** |
| `verify_setup.py` logic | **Behaved correctly with stand-ins**; real run pending |
| Mermaid diagrams | **Passed (real)**: 7 of 7 rendered with Mermaid CLI |
| Python 3.10 compatibility | **Passed (real)**: Ruff syntax and name checks only |
| Windows installation | **Not run** |
| ZIP integrity | **Passed (real)**: see [ZIP verification](#zip-verification) |

Two of the first 98 tests failed in the first run. Both were test-setup issues: the fake embedder scored below the real relevance floor. The fixture now disables the floor, and a separate test covers it. Review then added four more tests.

## Test cases

### Document parsing: `tests/test_document_parser.py` (12 tests)

| Test case | Expected | Actual |
|---|---|---|
| Extract text from all three samples | Text and page 1 are returned | Passed (stand-in) |
| Page numbers match source pages | Text maps to the right page | Passed (stand-in) |
| Blank page in a document | Reported, and numbering is preserved | Passed (stand-in) |
| Non-PDF extension | `InvalidFileTypeError` | Passed (stand-in) |
| Zero-byte file | `EmptyPDFError` | Passed (stand-in) |
| Missing PDF signature | `CorruptedPDFError` | Passed (stand-in) |
| Corrupted PDF body | `CorruptedPDFError` | Passed (stand-in) |
| PDF with no text | `EmptyPDFError` mentioning OCR | Passed (stand-in) |
| Text cleaning | Hyphens, spaces and null characters are fixed | Passed (real logic) |
| Real hyphens are kept | `twenty-four` is unchanged | Passed (real logic) |
| Filename sanitising | Paths are removed | Passed (real logic) |
| Content hash | Stable and content-based | Passed (real logic) |

### Chunking: `tests/test_text_chunker.py` (6 tests)

| Test case | Expected | Actual |
|---|---|---|
| Size limit | No chunk exceeds 800 characters | Passed (stand-in) |
| Metadata | Filename, page and unique ID are kept | Passed (stand-in) |
| Page accuracy | Chunk text comes from its stated page | Passed (stand-in) |
| Overlap | Adjacent chunks share text | Passed (stand-in) |
| Tiny fragments | Dropped | Passed (stand-in) |
| Serialisation | Round trip preserves the chunk | Passed (stand-in) |

### Embeddings: `tests/test_embeddings.py` (7 unit tests, 1 integration test)

| Test case | Expected | Actual |
|---|---|---|
| Required model name | `all-MiniLM-L6-v2` | Passed (stand-in) |
| Vector shape and type | `(n, 384)`, float32 | Passed (stand-in) |
| Query vector | `(384,)` | Passed (stand-in) |
| Empty input | No model is needed | Passed (stand-in) |
| Blank query (2 cases) | Rejected | Passed (stand-in) |
| Load failure | Clear `EmbeddingError` | Passed (stand-in) |
| Real unit vectors and semantic ranking | Relevant text ranks higher | **Not run** |

### Vector store and document manager: `tests/test_vector_store.py` (19 tests)

| Area | Cases | Actual |
|---|---|---|
| Search | Ranking, top-k, document filter, empty selection, empty store | Passed (stand-in) |
| Validation | Wrong dimension, mismatched lengths | Passed (stand-in) |
| Removal | Only the target is removed; IDs stay correct afterwards | Passed (stand-in) |
| Persistence | Save and load; a damaged store starts empty | Passed (stand-in) |
| Uploads | Multiple files, duplicates, bad files, oversized files | Passed (stand-in) |
| Library | Removal, re-upload, survival across restart, text cleared from disk | Passed (stand-in) |

### RAG, analysis and LLM client: `tests/test_rag_pipeline.py` (58 tests)

| Area | Cases | Actual |
|---|---|---|
| RAG flow | Four chunks, filtering, not-found handling, relevance floor, uncited answers | Passed (stand-in) |
| Citation validation | Invented labels removed, mixed labels, repeated labels | Passed (real logic) |
| Untrusted content | Tag escaping, injection stays in context, prompts, image and link removal | Passed (real logic) |
| Output parsing | JSON extraction, retry, clear failure | Passed (real logic) |
| Quote verification | Correct page, curly quotes, small differences, changed numbers, invented text | Passed (real logic) |
| Clause extraction | Nine fields, true page, unverifiable quotes, invented figures, malformed output | Passed (stand-in) |
| Summary | Generation and invalid page references | Passed (stand-in) |
| Risk analysis | Sorting, verified pages, invented clauses, missing items, cap | Passed (stand-in) |
| Analysis context | Short and long contracts | Passed (stand-in) |
| LLM client | Missing key, placeholder key, unknown provider, content formats, error messages, no leaked text | Passed (stand-in) |

### Streamlit: `tests/test_app_smoke.py` (4 integration tests)

| Test case | Expected | Actual |
|---|---|---|
| Each of the four pages renders | No exception | **Not run** |

## App walkthrough

A temporary script drove `app.py` through a stub of Streamlit using the fake embedder and LLM. It passed these steps:

1. All pages open while empty.
2. Three samples and one bad file are uploaded together. The bad file's error is surfaced, and the others are indexed (21 chunks).
3. The Dashboard shows the document table.
4. Deselecting a document removes it from the assistant's search.
5. Chat produces a cited answer, and a model-supplied image link is removed.
6. History is displayed again after a rerun.
7. An API failure shows as an error and does not crash the page.
8. Clear chat works.
9. Analysis returns all three sections.
10. A partial failure keeps the summary while reporting the other two errors.
11. Document removal clears its analysis.

This shows that the control flow works. It says **nothing about layout, styling, or whether Streamlit accepts each call**.

## Mutation checks

To confirm that the tests can actually fail, safety controls were deliberately broken one at a time on a temporary copy. The test suite was then run.

| Deliberate break | Tests failed |
|---|---|
| Keep invented citation labels | 2 |
| Skip the changed-number guard | 1 |
| Ignore the document filter | 3 |
| Remove context escaping | 2 |
| Trust the model's page number | 2 |
| Skip the duplicate check | 2 |
| Call the LLM when nothing is retrieved | 1 |
| Leave vectors behind on removal | 2 |
| Show raw exception text | 1 |

The context-escaping mutation initially failed to apply, because a formatter had changed that line, and the tests passed. That run was discarded and repeated properly. The result above is from the repeat.

## Error handling

| Scenario | Expected behaviour | Actual |
|---|---|---|
| Damaged, empty or non-PDF upload | A clear message, with other files continuing | Passed (stand-in) |
| Password-protected PDF | A clear message | **Not run**: the code path exists, but no test covers it |
| Missing or placeholder API key | A clear message | Passed (stand-in) |
| Invalid key, rate limit, retired model, timeout, network failure | Mapped to a clear message | Passed (stand-in): simulated error text only, so real Gemini error wording is unconfirmed |
| Empty model response | A clear message | Passed (stand-in) |
| Malformed JSON | One retry, then a clear message | Passed (stand-in) |
| Embedding model download failure | A clear message | Passed (stand-in) |
| Damaged saved index | Starts empty | Passed (stand-in) |

## Retrieval validation

**Real retrieval quality has not been measured.** The fake embedder matches on shared words, so it can test the plumbing but not semantic search.

`verify_setup.py` has six labelled questions. With the fake embedder, one of these failed: "How much notice is needed to stop the vendor contract renewing?". This is a limit of bag-of-words matching, and it is the kind of question that real embeddings are meant to handle, but that has not been confirmed.

The relevance floor of 0.15 has not been calibrated. `verify_setup.py` prints real scores for relevant and unrelated questions, so you can judge it.

## Source citation validation

| Control | Status |
|---|---|
| Citations come only from retrieved chunks | Passed (stand-in) |
| Invented labels are removed | Passed (real logic) |
| Extraction pages come from located quotes | Passed (real logic) |
| Invented quotes are rejected | Passed (real logic) |
| Quotes with changed numbers are rejected | Passed (real logic) |
| Real Gemini output follows the citation format | **Not run** |

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Multiple PDF uploads work | Passed (stand-in) |
| 2 | PDF text extraction works | Passed (stand-in): `pypdfium2`, not PyMuPDF |
| 3 | Page-level metadata is preserved | Passed (stand-in) |
| 4 | Text chunking works | Passed (stand-in) |
| 5 | Hugging Face embeddings are generated | **Not run** |
| 6 | FAISS indexing works | Passed (stand-in): NumPy, not FAISS |
| 7 | Semantic retrieval returns relevant chunks | **Not run** |
| 8 | Gemini generates contextual answers | **Not run** |
| 9 | Source citations are displayed correctly | Logic passed (stand-in); display **not run** |
| 10 | Missing information is handled correctly | Logic passed (stand-in); real model behaviour **not run** |
| 11 | Summaries can be generated | Logic passed (stand-in); real generation **not run** |
| 12 | Key clauses can be extracted | Logic passed (stand-in); real extraction **not run** |
| 13 | Potential risk findings can be generated | Logic passed (stand-in); real generation **not run** |
| 14 | Documents can be removed | Passed (stand-in) |
| 15 | The application runs locally | **Not run** |
| 16 | Installation instructions are complete | Written; **not run** on Windows |
| 17 | Documentation is included | **Passed (real)** |
| 18 | Tests are executed and outcomes documented | Unit tests passed (stand-in); integration tests **not run** |

Criteria 5, 7, 8 and 15 are open. Only a real run can close them.

## ZIP verification

The final ZIP was checked after a fresh extraction. See `DEVELOPMENT_PROGRESS.md` for the checks and results.

## Known issues

1. The real stack has never been run.
2. `requirements.txt` uses version ranges that could not be test-installed. Versions were checked against package pages where possible, but the combination is unconfirmed.
3. The `ChatGoogleGenerativeAI` arguments (`model`, `google_api_key`, `temperature`, `timeout`, `max_retries`) could not be confirmed from its documentation or by running them.
4. Streamlit's `width="stretch"` arguments and other calls have not been run. An incompatibility would show as a page error.
5. The real tolerance of PyMuPDF for damaged files may differ from the stand-in's. The code handles either a failed open or zero pages.
6. Free-tier Gemini rate limits could not be confirmed. Each analysis uses three requests.
7. The relevance floor is uncalibrated.
8. Password-protected PDF handling has no test.

## Running the tests yourself

```powershell
python verify_setup.py
pytest
pytest -m integration
```

Then update this document with your real results.
