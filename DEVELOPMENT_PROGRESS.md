# Development Progress

Status as of 3 November 2026.

**Overall:** code, sample contracts, tests and documentation are complete. The application has **not** been run with its real libraries or with Gemini, because the development workspace blocked PyPI, Hugging Face and the Gemini API, and no API key was available. "Tested" below means tested against stand-ins unless stated otherwise. See [docs/TESTING_DOCUMENTATION.md](docs/TESTING_DOCUMENTATION.md).

## Phases

| Phase | Implemented | How it was tested | Status |
|---|---|---|---|
| 0. Planning | Architecture and environment check | Found the network limits | Done |
| 1. Setup | Folders, requirements, settings, theme | Real: file checks | Done; install not run |
| 2. Document parsing | `document_parser.py` | 12 unit tests, stand-in parser | Logic tested |
| 3. Text chunking | `text_chunker.py` | 6 unit tests, stand-in splitter | Logic tested |
| 4. Embeddings | `embeddings.py` | 7 unit tests, stub model | Logic tested; real model not run |
| 5. FAISS | `vector_store.py`, `document_manager.py` | 19 unit tests, NumPy stand-in | Logic tested |
| 6. RAG | `rag_pipeline.py`, `llm_client.py`, `prompts.py` | Unit tests, scripted LLM | Logic tested; Gemini not run |
| 7. Contract intelligence | `features/`, `parsing.py` | Unit tests, scripted LLM | Logic tested; Gemini not run |
| 8. Frontend | `app.py` | Walkthrough with a stub | Control flow tested; interface not rendered |
| 9. Testing | 102 unit tests, walkthrough, 9 mutation checks | Results recorded | Done for stand-ins; 5 integration tests not run |
| 10. Documentation and packaging | 10 docs, README, ZIP | Diagrams rendered; ZIP checked | Done |

## Issues found and fixed

| Issue | Fix |
|---|---|
| Tutorials commonly use `gemini-2.0-flash`, which shut down on 1 June 2026 | The default became `gemini-3.5-flash-lite`, and the model is configurable |
| Automatic lint fixing removed an import that set a Windows process option | `core/__init__.py` now loads settings first |
| Fuzzy quote matching could accept a quote with a changed number | Quotes must now match on all numbers |
| A client start-up failure could crash the page | It is now reported as `LLMError` |
| Upload messages disappeared on rerun | Messages are now kept in session state |
| `st.text` could display long quotes poorly | Document text now uses escaped Markdown |
| A not-found token wrapped in formatting was missed | Detection now handles it |
| Sample PDFs were only two pages each | Type size was increased, and two contracts now have three pages |
| Two tests failed on the fake embedder's low scores | The fixture disables the floor, and the floor has its own test |

## Additions to the requested structure

All are small and explained in [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md): `core/document_manager.py`, `core/llm_client.py`, `utils/parsing.py`, `verify_setup.py`, `tests/conftest.py`, `tests/test_app_smoke.py`, `pytest.ini`, `.streamlit/config.toml` and the package `__init__.py` files.

## Pending tasks

These need a Windows machine with internet access and a Gemini key:

- [ ] Complete the setup steps
- [ ] Run `python verify_setup.py`
- [ ] Run `pytest` and `pytest -m integration`
- [ ] Run `streamlit run app.py` and review all four pages
- [ ] Try the sample questions and review the answer quality
- [ ] Update `docs/TESTING_DOCUMENTATION.md` with the real results
- [ ] Add screenshots

## ZIP verification

The final archive was extracted into an empty folder and checked for these points:

- All 41 required files are present and not empty.
- No `.env`, virtual environment, cache, vector store or stand-in files are included.
- No API-key-like strings or development paths are present.
- No unfinished placeholder markers are present.
- The 102 unit tests passed from the extracted copy, using stand-ins kept outside the project.
