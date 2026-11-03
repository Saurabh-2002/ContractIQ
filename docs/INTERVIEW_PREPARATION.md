# Interview Preparation

**Before your interview, complete the setup and run `python verify_setup.py`.** Only claim what you have seen working. If you have not yet run it, say that the logic was tested with fakes and that real-stack verification was pending. Interviewers respect that kind of precision.

## Product introduction

> ContractIQ is an AI contract assistant. You upload PDF contracts, and it extracts key terms, flags clauses for human review, and answers questions with citations to the exact page. I built it as an MVP to learn how RAG products work and to explore how to make AI output trustworthy.

## Problem statement

Contracts hold commitments that affect money and risk, including renewal windows, liability caps and termination rights. Reviewing them is slow, and keyword search fails when the user's words differ from the contract's wording.

## Why this product was built

Three reasons:

1. It is a focused, high-value use case for document AI.
2. It requires the full RAG stack: parsing, chunking, embeddings, retrieval and generation.
3. Trust is the central product challenge. A wrong contract answer is costly, which forces real thinking about verification.

## Product architecture

There are two main flows:

- **Ingestion:** PDF, text extraction, cleaning, chunking, embeddings, FAISS.
- **Query:** question, embedding, FAISS search, top four chunks, Gemini, validated answer.

Contract Intelligence is a third flow. It sends the whole contract to Gemini and then verifies the output against the document.

The important design choice is modularity. The LLM sits behind a one-method interface, so changing provider does not touch the pipeline.

## Why RAG was selected

| Option | Issue |
|---|---|
| LLM alone | It has no knowledge of the user's contract |
| Send the full document each time | Higher cost, and more noise for specific questions |
| Fine-tuning | Unsuitable for documents that change per user, and out of scope |
| RAG | Uses the user's data, sends only relevant text, and enables citations |

## Why Hugging Face was selected

The `all-MiniLM-L6-v2` model is free, small, fast on a CPU and widely used. It runs locally, so chunks are embedded without an external API call. The trade-off is that it is a general model, not one trained on legal language.

## Why FAISS was selected

FAISS is free, local and fast, and it needs no server, which suits a local MVP. The trade-off is that it is a library, so metadata, filtering and persistence had to be built by hand. A multi-user product would likely need a managed vector database.

## Why Gemini was selected

It has a free tier, which suits a portfolio project, and it has a LangChain integration. It was a practical choice rather than a claim that it is the best model. The provider is swappable, so the decision is cheap to reverse.

## Embedding model explanation

An embedding model maps text to a list of numbers, so that similar meanings land close together. `all-MiniLM-L6-v2` produces 384 numbers. It lets "cancel the renewal" match "notice of non-renewal", even though the words differ.

## Chunking explanation

A contract is split into 800-character chunks with a 150-character overlap. Chunks must be small enough to focus on one topic but large enough to keep context, and the overlap protects sentences at the boundaries.

The interesting decision was chunking per page. It guarantees accurate page citations, but a clause crossing a page break gets split. I chose citation accuracy because it is the basis of user trust.

## Semantic search explanation

The question is embedded with the same model as the chunks, and FAISS returns the closest vectors by cosine similarity. It searches by meaning instead of exact words.

## LLM versus embedding model

| | Embedding model | LLM |
|---|---|---|
| Job | Converts text to numbers | Reads and writes text |
| Output | A 384-number vector | Natural language |
| Role | Finds relevant passages | Writes the answer |
| Where it runs | Locally | Google's API |

A helpful analogy: the embedding model is the librarian who finds the right pages, and the LLM is the analyst who reads them and writes the answer.

## Hallucination control

I used layers, because no single control is enough:

1. If nothing relevant is retrieved, the LLM is not called.
2. The prompt limits answers to the sources.
3. The code validates citation labels and removes invented ones.
4. Uncited answers carry a warning.
5. Extracted values need a quote that exists in the document.
6. Page numbers come from where the quote is found, not from the model.
7. Supporting text is shown to the user.

The key principle is to **treat model output as unverified until code checks it**. These controls reduce the risk but do not eliminate it, which is why a person stays in the loop.

## Technical challenges

| Challenge | Approach |
|---|---|
| Accurate citations | Chunk per page and carry metadata through the pipeline |
| Invented sources | Map labels to retrieved chunks in code |
| Prompt injection | Delimit documents, mark them untrusted, and neutralise closing tags |
| Removing documents | Use an ID-mapped FAISS index |
| Near-matching quotes | Allow small differences but reject any changed number |
| Unreliable JSON | Parse tolerantly and retry once |
| Retired models | Make the model configurable with a clear error message |
| Testing without network access | Use fakes behind interfaces, plus a real-stack verification script |

The last challenge makes a good story. The development environment could not reach the libraries or APIs, so I kept tested and untested claims separate and built a script to close that gap.

## Product success metrics

These are proposed metrics. **ContractIQ has no real usage data.**

| Type | Metric |
|---|---|
| North star | Contracts reviewed per user per month |
| Value | Time to find a clause, compared with manual review |
| Quality | Share of answers with a correct, verified citation |
| Quality | Retrieval hit rate on a labelled test set |
| Trust | Share of sources opened or checked by users |
| Trust | Thumbs-up rate on answers |
| Safety | Rate of unsupported claims in sampled answers |
| Adoption | Share of users who ask a second question |
| Cost and speed | Cost per analysis and response time |

## Possible future enhancements

Priorities would be a retrieval evaluation set, hybrid search with reranking, OCR, contract comparison, playbook checks and renewal tracking. See [LIMITATIONS_AND_FUTURE_SCOPE.md](LIMITATIONS_AND_FUTURE_SCOPE.md).

For product sequencing, I would measure quality first, because it is hard to prioritise features without knowing where answers fail.

---

## 20 likely technical interview questions

**1. What is RAG?**

Retrieval-augmented generation finds relevant passages from a knowledge source and gives them to an LLM to answer from. It grounds the output in specific text.

**2. Walk me through what happens when a user asks a question.**

The question is embedded, FAISS finds the four closest chunks, and chunks under the relevance floor are dropped. If any remain, they are labelled and sent to Gemini with the rules and question. The code then validates the citations and displays the answer and sources.

**3. What is an embedding?**

A numeric representation of text in which similar meanings are close together. ContractIQ's vectors have 384 dimensions.

**4. What is cosine similarity?**

It measures the angle between vectors, so it compares direction rather than length. ContractIQ normalises vectors to length 1, so inner product equals cosine similarity.

**5. Why use the same model for documents and queries?**

Different models create different vector spaces, so their distances mean nothing to each other.

**6. Why 800 and 150 for chunk size and overlap?**

They were specified in the project brief, and they are a reasonable starting point for clause-length text. In a real product, I would tune them against a retrieval evaluation set.

**7. What is the downside of chunking per page?**

A clause crossing a page break is split, and overlap does not bridge it. Retrieval might return only half of it. I accepted that to guarantee exact page citations.

**8. Why `IndexFlatIP` rather than an approximate index like HNSW?**

It gives exact results and needs no tuning. At this scale, brute force is fast enough. Approximate indexes make sense at much larger scale, when exact search becomes too slow.

**9. How does document filtering work?**

FAISS searches all vectors, and the code keeps results from the selected documents until it has four. That is simple and exact here, but filtering inside the index would be better at large scale.

**10. How do you remove a document from FAISS?**

`IndexIDMap2` gives each vector an application ID. I find a document's IDs in the metadata and call `remove_ids`.

**11. How do you stop fabricated citations?**

The model never supplies the filename or page. It only refers to labels, and the code maps valid labels to retrieved chunks and removes invalid ones.

**12. How do you verify extracted fields?**

Each value needs a verbatim quote. The code searches for it in the document, takes the page from the match, and checks that numbers in the value exist in the document. Failures are marked for review.

**13. What is prompt injection, and how do you handle it?**

It is text in a document that tries to give the model instructions, such as "ignore your rules". I separate document content with tags, label it untrusted, neutralise closing tags, and strip remote images from output. No defence is complete, so the model has no ability to act beyond producing text.

**14. Why a low temperature?**

Extraction and grounded answers need consistency more than creativity. A temperature of 0.1 makes responses more repeatable.

**15. How does the LLM abstraction work?**

Code depends on a protocol with a `generate` method. `GeminiClient` implements it, and `create_llm_client` chooses the provider. A new provider needs one class and one registration line.

**16. How did you test code that depends on an LLM?**

Unit tests use a scripted `FakeLLM` to test everything around the model: validation, parsing and error handling. They cannot judge real answer quality, so that needs a separate evaluation on real output.

**17. How would you evaluate RAG quality?**

Separately for each stage. For retrieval, I would use a labelled question set and measure hit rate and mean reciprocal rank. For generation, I would measure faithfulness to the sources, answer correctness, and correct handling of unanswerable questions.

**18. What happens with a scanned PDF?**

It has no text layer, so extraction returns nothing and the app explains that OCR is not supported. OCR was out of scope for the MVP.

**19. How would you scale this for many users?**

I would add authentication and tenant isolation, a managed vector database, background ingestion, encryption, audit logs, and a queue with rate-limit handling for LLM calls. I would also monitor cost and quality.

**20. What would you do differently?**

I would build the evaluation set earlier, since quality decisions need measurement. I would also explore hybrid search, because contracts contain exact defined terms that keyword search handles well.

---

## Questions to prepare from your own run

After running the app, note your real answers to these:

- How long does a contract take to process?
- Which question impressed you, and which failed?
- What did `verify_setup.py` report for retrieval scores?
- Did the risk findings match the planted issues in the sample contracts?
