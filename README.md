# ContractIQ

**Understand Contracts. Extract Insights. Make Informed Decisions.**

ContractIQ is a portfolio-level MVP of an AI document intelligence tool. Upload PDF contracts, ask questions and get answers with page references, extract key clauses, and flag points that may need human review.

> ContractIQ is an informational aid, not legal advice.

## Verification status

**Run `python verify_setup.py` first.** ContractIQ was developed in a workspace that could not install Streamlit, PyMuPDF, FAISS, LangChain or sentence-transformers, or reach Hugging Face or Gemini. Its logic passed tests against temporary stand-ins, but it has **not yet been run against the real libraries**. See [docs/TESTING_DOCUMENTATION.md](docs/TESTING_DOCUMENTATION.md) for exactly what was and was not tested.

## Features

| Module | What it does |
|---|---|
| PDF upload | Handles multiple PDFs, validates them, and detects duplicates |
| Parsing and chunking | Extracts text with PyMuPDF and creates 800-character chunks with 150-character overlap, keeping page numbers |
| Embeddings and search | Uses `all-MiniLM-L6-v2` and FAISS, with per-document filtering and removal |
| AI Assistant | Answers from the top four retrieved chunks, with verifiable sources |
| Contract Intelligence | Produces an executive summary and nine structured fields |
| Risk identification | Sorts findings into High, Medium and Low Attention |

## Quick start (Windows PowerShell)

Requires 64-bit Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder in `.env` with your Gemini API key, save, and then run:

```powershell
python verify_setup.py
streamlit run app.py
```

Full instructions and troubleshooting are in [docs/SETUP_AND_INSTALLATION.md](docs/SETUP_AND_INSTALLATION.md).

## Technology

Python, Streamlit, PyMuPDF, LangChain, Hugging Face Sentence Transformers, FAISS, Google Gemini, python-dotenv, Pandas and Pytest.

## Project layout

```text
ContractIQ/
├── app.py                 Streamlit interface
├── verify_setup.py        Checks the real stack and records the results
├── config/settings.py     All configuration
├── core/                  Parsing, chunking, embeddings, FAISS, RAG, LLM access
├── features/              Summary, clause extraction, risk analysis
├── utils/                 Prompts and model-output validation
├── tests/                 Pytest suite
├── data/sample_contracts/ Three fictional PDFs
└── docs/                  Full documentation
```

## Documentation

| Document | Purpose |
|---|---|
| [Project Overview](docs/PROJECT_OVERVIEW.md) | The problem, users and MVP boundaries |
| [System Architecture](docs/SYSTEM_ARCHITECTURE.md) | Components and data flow |
| [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) | Every major module |
| [RAG Pipeline Explained](docs/RAG_PIPELINE_EXPLAINED.md) | A beginner-friendly walkthrough |
| [Setup and Installation](docs/SETUP_AND_INSTALLATION.md) | Windows setup steps |
| [User Guide](docs/USER_GUIDE.md) | How to use the app |
| [API Configuration](docs/API_CONFIGURATION.md) | Gemini setup and changing provider |
| [Testing Documentation](docs/TESTING_DOCUMENTATION.md) | Test cases and actual results |
| [Limitations and Future Scope](docs/LIMITATIONS_AND_FUTURE_SCOPE.md) | Current limits and possible next steps |
| [Interview Preparation](docs/INTERVIEW_PREPARATION.md) | Product and technical talking points |
