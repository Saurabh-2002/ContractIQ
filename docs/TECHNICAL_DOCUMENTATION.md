# Technical Documentation

## Module map

| File | Layer | Purpose |
|---|---|---|
| `app.py` | Interface | Streamlit pages |
| `verify_setup.py` | Tooling | Checks the real stack |
| `config/settings.py` | Configuration | Settings and process options |
| `core/document_parser.py` | Ingestion | PDF validation, extraction and cleaning |
| `core/text_chunker.py` | Ingestion | Page-aware chunking |
| `core/embeddings.py` | Retrieval | Hugging Face embeddings |
| `core/vector_store.py` | Retrieval | FAISS storage and search |
| `core/document_manager.py` | Ingestion | Ingestion workflow and persistence |
| `core/llm_client.py` | Generation | Provider-neutral LLM access |
| `core/rag_pipeline.py` | Generation | RAG flow and analysis context |
| `features/contract_summary.py` | Analysis | Executive summary |
| `features/clause_extraction.py` | Analysis | Nine structured fields |
| `features/risk_analysis.py` | Analysis | Attention-graded findings |
| `utils/prompts.py` | Generation | All prompts |
| `utils/parsing.py` | Validation | JSON parsing and output checks |

`core/document_manager.py`, `core/llm_client.py`, `utils/parsing.py`, `verify_setup.py`, `tests/conftest.py`, `tests/test_app_smoke.py`, `pytest.ini` and `.streamlit/config.toml` are additions to the requested structure. Each is explained below.

---

## config/settings.py

**Purpose:** a single source of configuration.

**Responsibilities:** load `.env`, define defaults, treat the placeholder key as missing, and set process options.

**Important items:** `Settings` (frozen dataclass), `get_settings()`, `PROJECT_ROOT`.

**Input and output:** reads environment variables and returns `Settings`.

**Dependencies:** `python-dotenv`.

**Interactions:** used by all other layers. `core/__init__.py` imports it first, because `KMP_DUPLICATE_LIB_OK` must be set before PyTorch and FAISS load. Without it, Windows can abort with OpenMP error #15 when both are present.

Paths derive from the file's own location, so the project works wherever it is extracted.

---

## core/document_parser.py

**Purpose:** convert PDF bytes into cleaned text for each page.

**Responsibilities:** check the extension and `%PDF-` signature; handle damaged, encrypted, empty and text-free files; report pages with no text; and clean extracted text.

**Important functions:**

| Function | Input | Output |
|---|---|---|
| `parse_pdf(data, filename)` | Bytes and a name | `ParsedDocument` |
| `validate_pdf_bytes(filename, data)` | Bytes and a name | Raises on failure |
| `clean_text(text)` | Raw text | Cleaned text |
| `compute_file_hash(data)` | Bytes | SHA-256 hex string |
| `sanitize_filename(filename)` | A name or path | Base name |

**Errors:** all derive from `DocumentParseError`, and messages are safe to display.

**Dependencies:** PyMuPDF.

**Interactions:** called by `DocumentManager`. `ParsedDocument` goes to `text_chunker`.

**Decision:** cleaning keeps paragraph breaks, because the chunker prefers to split on them. Blank pages keep their place in the numbering, so citations match the PDF viewer.

---

## core/text_chunker.py

**Purpose:** split pages into retrievable passages.

**Important items:** `Chunk`, `chunk_document(document, chunk_size, chunk_overlap)`, `build_splitter(...)`.

**Input and output:** takes a `ParsedDocument` and returns a list of `Chunk`, each with `chunk_id`, `doc_id`, `filename`, `page_number`, `chunk_index` and `text`.

**Dependencies:** `langchain-text-splitters`.

**Decision:** chunking happens one page at a time, so each chunk has exactly one true page number. The trade-off is that a clause crossing a page break becomes two chunks, and overlap does not cross pages. Accurate citations matter more to this product than perfect continuity. Fragments under 20 characters, such as a stray page number, are dropped.

---

## core/embeddings.py

**Purpose:** turn text into vectors.

**Important items:** `Embedder` protocol, `HuggingFaceEmbedder`, `EmbeddingError`.

| Method | Input | Output |
|---|---|---|
| `embed_documents(texts)` | `list[str]` | float32 array, shape `(n, 384)` |
| `embed_query(text)` | `str` | float32 array, shape `(384,)` |

**Dependencies:** `sentence-transformers`, NumPy.

**Interactions:** used by `DocumentManager` and `RagPipeline`.

**Decisions:** loading is lazy, so the interface opens without waiting for PyTorch. Normalising to unit length makes inner product equal to cosine similarity. The `Embedder` protocol lets tests inject a fake.

---

## core/vector_store.py

**Purpose:** store vectors and metadata, and run filtered semantic search.

**Important methods:** `add_chunks`, `search(query_vector, top_k, doc_ids)`, `remove_document`, `chunks_for_document`, `save`, `load`.

**Output:** `search` returns `SearchResult` objects, each with a `chunk` and a `score`.

**Dependencies:** `faiss-cpu`, NumPy.

**Decisions:**

- `IndexFlatIP` gives exact results and needs no training. An approximate index would add complexity with no benefit at this scale.
- `IndexIDMap2` allows removal by ID.
- `doc_ids=None` means all documents, while an empty set means none. This prevents "no documents selected" from silently searching everything.
- The index is serialised to bytes and written by Python, because `faiss.write_index` can fail on Windows paths with non-ASCII characters.
- A damaged saved store is ignored with a warning, so the app still opens.

---

## core/document_manager.py (added)

**Purpose:** coordinate ingestion and keep the workflow out of the interface.

**Important items:** `DocumentManager`, `DocumentRecord`, `IngestResult`.

| Method | Purpose |
|---|---|
| `ingest(filename, data)` | Runs validate, parse, duplicate check, chunk, embed, index and save |
| `remove(doc_id)` | Deletes from the library, FAISS and disk |
| `documents()` | Lists documents, newest first |
| `get(doc_id)` | Returns one record |

**Output:** `ingest` returns `IngestResult` with status `indexed`, `duplicate` or `failed`. It does not raise for bad input.

**Dependencies:** all ingestion modules.

**Why it was added:** the requested structure had no home for this workflow. Placing it in `app.py` would have made it impossible to test without Streamlit.

---

## core/llm_client.py (added)

**Purpose:** isolate the LLM provider.

**Important items:** `LLMClient` protocol, `GeminiClient`, `create_llm_client`, `content_to_text`, `describe_llm_failure`, `LLMError`.

**Input and output:** `generate(system_prompt, user_prompt)` returns text or raises `LLMError`.

**Dependencies:** `langchain-google-genai`, `langchain-core`.

**Decisions:**

- `content_to_text` accepts a string or a list of content blocks, since the format depends on the model and library version.
- `describe_llm_failure` converts provider errors into clear messages by matching text, so it does not depend on SDK exception classes. It never displays the original exception text, which could echo contract content.

**Why it was added:** the brief requires a modular LLM integration.

---

## core/rag_pipeline.py

**Purpose:** answer questions from retrieved context, and prepare contracts for analysis.

**Important items:**

| Item | Purpose |
|---|---|
| `RagPipeline.retrieve` | Embeds a question and returns chunks above the relevance floor |
| `RagPipeline.ask` | Runs the full flow and returns `RagAnswer` |
| `build_context` | Formats chunks with `[S#]` labels |
| `resolve_citations` | Keeps valid labels and removes invented ones |
| `escape_context` | Neutralises context tags in document text |
| `build_document_context` | Prepares text for analysis |

**Output statuses:**

| Status | Meaning |
|---|---|
| `answered` | At least one valid citation |
| `not_found` | Nothing relevant was retrieved, or the model reported no answer |
| `uncited` | The model answered without a valid citation, so the interface shows a warning |

**Decision:** if retrieval returns nothing above the floor, the LLM is not called. An unsupported question then costs nothing and cannot produce an invented answer.

---

## features/contract_summary.py

**Purpose:** produce a four-to-six-bullet executive summary.

**Important functions:** `generate_summary(llm, record, context)`, `remove_invalid_page_references(text, page_count)`.

**Output:** `SummaryResult`.

**Decision:** page references beyond the end of the document are removed. References within the valid range are not individually checked, because summaries do not carry quotes. This is noted in the limitations.

---

## features/clause_extraction.py

**Purpose:** extract nine fields.

**Important items:** `CLAUSE_FIELDS`, `extract_clauses`, `ExtractedField`.

**Field status:**

| Status | Condition |
|---|---|
| `found` | The model found a value, its quote exists in the document, and all its figures appear in the document |
| `not_found` | The model reported nothing, or the output for that field was unusable |
| `needs_review` | A value was returned, but its quote or figures could not be verified |

**Decision:** an unverified value stays visible with a warning. Hiding it could conceal a real term, while presenting it as verified would be misleading.

---

## features/risk_analysis.py

**Purpose:** identify clauses that may need human review.

**Important items:** `analyze_risks`, `RiskFinding`, `LEVELS`.

**Output:** a list sorted High to Low and capped at 12 findings.

**Decision:** findings are clause-based or missing-item-based. A missing item has no page by nature, and it is labelled as not stated in the document. A clause finding whose quote cannot be found is shown as not verified, with no page number.

---

## utils/prompts.py

**Purpose:** keep all prompts in one reviewable place.

All prompts share `SECURITY_RULES`, which mark document content as untrusted. `NOT_FOUND_TOKEN` gives the model an exact way to report a missing answer, which code can detect reliably.

---

## utils/parsing.py (added)

**Purpose:** treat model output as unverified until it is checked.

| Function | Purpose |
|---|---|
| `extract_json` | Parses JSON, even when wrapped in prose or code fences |
| `generate_json` | Calls the model and retries once on unreadable output |
| `locate_quote` | Finds the true page for a quote |
| `unsupported_figures` | Lists numbers that are absent from the document |
| `sanitize_markdown` | Removes model-supplied images and links |
| `escape_markdown` | Displays document text literally |

**Decision:** `locate_quote` tolerates small differences in case, punctuation and spacing, but it rejects a quote if any number differs. "(45) days" and "(65) days" are similar strings but different claims.

---

## app.py

**Purpose:** the four-page interface.

| Function | Page or role |
|---|---|
| `render_dashboard` | Dashboard |
| `render_library`, `process_uploads`, `remove_document` | Document Library |
| `render_intelligence`, `run_analysis` | Contract Intelligence |
| `render_assistant`, `render_message` | AI Assistant |
| `render_sidebar`, `main` | Navigation and routing |

**Decisions:**

- Each of the three analyses fails independently, so a rate limit on one does not discard the others.
- Filenames, quotes and reasons are escaped because they come from untrusted sources.

---

## verify_setup.py (added)

**Purpose:** verify the real stack on the user's machine and record the outcome in `docs/LOCAL_VERIFICATION_RESULTS.md`.

**Why it was added:** the real libraries were not available during development. See [TESTING_DOCUMENTATION.md](TESTING_DOCUMENTATION.md).

---

## Tests and tool configuration (added)

- `tests/conftest.py` provides `FakeEmbedder`, `FakeLLM` and fixtures.
- `tests/test_app_smoke.py` opens each page with Streamlit's test runner.
- `pytest.ini` keeps tests that need real models out of the default run.
- `.streamlit/config.toml` sets the theme and disables Streamlit's file watcher, which otherwise inspects PyTorch internals and logs errors.
