# Limitations and Future Scope

## Implemented features

The features below exist in the code. See [TESTING_DOCUMENTATION.md](TESTING_DOCUMENTATION.md) for how far each has been verified.

- Upload, validation and duplicate detection for multiple PDFs
- PyMuPDF extraction and page-aware chunking
- Hugging Face embeddings and local FAISS search
- Document filtering and removal
- A RAG assistant with validated citations
- Executive summary, nine-field extraction and graded risk findings
- A modular LLM client, with Gemini as the first provider

## Current limitations

### Verification

- **Not yet run against the real stack.** Development tests used stand-ins for PyMuPDF, FAISS, LangChain and Streamlit, and fakes for the embedding model and Gemini. Real answer quality, retrieval quality and interface appearance are unmeasured.
- **Dependency ranges are unconfirmed.** `requirements.txt` could not be installed during development.

### Documents

- Only text-based PDFs are supported. Scanned pages have no OCR.
- Tables can lose their structure when text is extracted.
- Multi-column layouts may be read in an unexpected order.
- Text is not separated from headers and footers, so a repeated banner can enter chunks.

### Retrieval

- Chunks stay within a page, so a clause crossing a page break becomes two chunks.
- The relevance floor of 0.15 is a conservative default that has not been calibrated on real scores.
- `all-MiniLM-L6-v2` is a general-purpose English model, and it may miss contract-specific meaning.
- Only the top four chunks are used, so questions spanning many clauses may get incomplete answers.
- The assistant does not compare or list across every document in a structured way.

### Generation

- Each question is independent, so follow-up questions need full context.
- Citation validation proves that a source was retrieved. It does not prove that the model interpreted it correctly.
- Quote matching proves that a quote exists. It does not prove that the extracted value follows from that quote.
- Page references in summaries are checked only for being within the document's range.
- Numbers written as words in a value are not checked.
- Prompt injection defences reduce the risk but cannot guarantee protection.
- Very long contracts receive a partial analysis, with a warning.

### Product and operations

- Single local user, with no authentication.
- Contract text is stored unencrypted in `data/vector_store/`.
- Text sent to Gemini leaves the machine, and free-tier usage may be used to improve Google's products.
- Analysis results and chat history last only for the session.
- Results vary by model, so a model change needs re-testing.
- ContractIQ does not give legal advice.

## Possible future improvements

**None of these are implemented.**

### Near term

- Run a retrieval evaluation on a labelled question set
- Calibrate the relevance floor from real scores
- Remove repeated headers and footers
- Add a cross-encoder reranker
- Combine keyword and semantic search
- Add conversational follow-up questions
- Export findings to CSV or PDF

### Medium term

- Add OCR for scanned PDFs
- Extract tables as structured data
- Compare two contracts clause by clause
- Add a contract playbook, so clauses can be checked against company standards
- Track renewal dates and notice windows
- Evaluate a contract-specific embedding model
- Support a locally hosted LLM for confidential work

### Long term

- Add authentication and role-based access
- Move to multi-tenant cloud deployment with encryption
- Use a managed vector database
- Add audit logs and human review workflows
- Integrate with contract lifecycle management systems

Each of these goes beyond the MVP scope and would need its own product decision.
