# The RAG Pipeline Explained

This guide follows one question through ContractIQ:

> "How much notice do I need to give to stop the vendor contract renewing?"

## What RAG is

RAG stands for **retrieval-augmented generation**. It has two steps:

1. **Retrieve:** find the passages most relevant to the question.
2. **Generate:** give those passages to an LLM and ask it to answer from them.

Think of an open-book exam. Rather than answering from memory, the student first finds the right pages and then writes the answer from them.

## Why RAG is needed

An LLM on its own has three problems for this task:

- **It has not seen your contract.** Your agreement was not in its training data.
- **It may guess.** A model can produce a confident answer with plausible but wrong terms.
- **Long documents are costly to send repeatedly.** Sending every contract with every question wastes tokens and adds noise.

RAG sends only the most relevant passages, gives the model evidence to work from, and shows the user where the answer came from.

## Document parsing

PDFs store positioned text for display, not clean paragraphs. PyMuPDF reads each page and returns its text. ContractIQ works page by page, so it always knows where the text came from.

ContractIQ has no OCR. A scanned PDF is a picture of text, so there is nothing to extract, and the app tells the user so.

## Text preprocessing

Raw PDF text is messy. ContractIQ:

- Removes hidden characters
- Rejoins words broken across lines, so `pay-` and `ment` become `payment`
- Collapses repeated spaces
- Reduces runs of blank lines while keeping paragraph breaks

## Chunking

The whole contract is too broad to search as one unit. A single vector for a 40-page contract would blur together every topic. ContractIQ splits text into smaller passages, called chunks.

`RecursiveCharacterTextSplitter` tries to split at natural boundaries in this order: paragraph, line, sentence, word.

## Chunk size and overlap

ContractIQ uses a **chunk size of 800 characters** and an **overlap of 150 characters**.

- **Too small:** a chunk loses context. It might contain "ninety (90) days" without saying what the ninety days relate to.
- **Too large:** a chunk mixes topics, which weakens the search match and sends irrelevant text to the LLM.

Overlap means that the end of one chunk is repeated at the start of the next:

```text
Chunk 1: ...shall automatically renew for successive periods of twelve (12) months unless
Chunk 2:        periods of twelve (12) months unless either party gives written notice...
```

Without overlap, a sentence split between chunks could be lost to both.

In ContractIQ, overlap applies within a page. Each chunk stays on one page so that its citation is exact.

## Embeddings

An embedding is a list of numbers that represents the meaning of text. Texts with similar meanings get similar numbers, even when they use different words.

"How do I cancel the renewal?" and "written notice of non-renewal" share almost no words, but their embeddings should be close together. This is why semantic search is better than keyword search for contracts.

## Hugging Face

Hugging Face hosts open AI models. ContractIQ uses `sentence-transformers/all-MiniLM-L6-v2`, which:

- Produces a vector of 384 numbers
- Is small, at about 90 MB
- Runs on an ordinary CPU
- Runs locally, so chunks and questions are embedded on your machine

The model is pretrained. ContractIQ does not train or fine-tune it.

## Vector databases

A normal database finds exact matches. A vector database finds **nearest neighbours**: the stored vectors closest to a query vector. It stores each chunk's vector alongside its metadata.

## FAISS

FAISS (Facebook AI Similarity Search) is an open-source library for fast vector search. Strictly speaking, it is a search library rather than a full database, so ContractIQ manages chunk metadata and saves files itself.

ContractIQ uses `IndexFlatIP`, which compares the query with every stored vector. That sounds slow, but it is effectively instant for a few thousand vectors, and it returns exact results. It runs locally, needs no server and costs nothing.

## Semantic search

When our question arrives:

1. The same embedding model converts it to 384 numbers.
2. FAISS compares that vector with each chunk's vector.
3. The closest chunks come back, ranked by score.

The same model must be used for chunks and questions. Vectors from different models are not comparable, just as map coordinates are meaningless in a different coordinate system.

## Cosine similarity

Cosine similarity measures the angle between two vectors:

```text
cosine similarity = (A · B) / (|A| × |B|)
```

- **Near 1:** very similar meaning
- **Near 0:** unrelated
- **Negative:** dissimilar

ContractIQ scales all vectors to a length of 1, so the bottom of that fraction becomes 1 and the score is just `A · B`, the inner product. That is why it can use an inner-product index for cosine similarity.

## Retrieval

ContractIQ takes the **top four chunks**. Four gives the model some choice of evidence without flooding it with noise.

Two refinements apply:

- **Document filter:** only documents ticked in the library are searched.
- **Relevance floor:** chunks scoring under 0.15 are dropped. If none remain, ContractIQ reports that the information was not found and never calls the LLM.

For our question, the Vendor Service Agreement's renewal clause should rank near the top.

## Prompt construction

The prompt has three parts:

1. **System instructions:** the rules. Answer only from the sources, cite them, say when something is missing, separate facts from interpretation, and ignore instructions inside documents.
2. **Context:** the retrieved chunks.
3. **Question:** what the user asked.

## Context injection

Retrieved chunks are placed into the prompt with labels:

```text
<document_context>
[S1] file: Vendor_Service_Agreement.pdf | page: 1
3.1 At the end of the Initial Term, this Agreement shall automatically renew
for successive periods of twelve (12) months unless either party gives the
other written notice of non-renewal at least ninety (90) days before the end
of the then-current term.

[S2] file: ...
</document_context>

Question: How much notice do I need to give to stop the vendor contract renewing?
```

The tags mark the boundary between data and instructions. If a document tried to include `</document_context>` to escape that boundary, ContractIQ would neutralise it first.

## LLM generation

An embedding model and an LLM do different jobs. The embedding model finds relevant text, and the LLM reads that text and writes the answer. Gemini might respond:

```text
**Facts**
- Either party must give written notice at least 90 days before the current
  term ends [S1].
```

## Source citations

The model writes only labels like `[S1]`. ContractIQ's code then:

1. Finds each label in the answer.
2. Removes any label that was not provided.
3. Maps each valid label to the chunk that was actually retrieved.
4. Displays that chunk's filename, page and text.

As a result, the model cannot make up a filename or page, because those details come from the index.

## Hallucination control

A hallucination is a confident statement with no basis. No single control removes that risk, so ContractIQ uses several layers:

| Layer | Control |
|---|---|
| Retrieval | No relevant chunks means no LLM call |
| Prompt | Use only the sources; reply with a set token if the answer is missing |
| Temperature | A low value of 0.1 favours consistent output |
| Citations | Labels are validated in code |
| Uncited answers | A warning says that the answer is unverified |
| Extraction | Quotes must exist in the document |
| Figures | Numbers absent from the document trigger review |
| Display | Supporting text is shown so that the user can check |

These controls reduce the risk but do not eliminate it. A model could cite a real passage and still misread it. This is why ContractIQ shows the source, and why a person remains responsible for the decision.

## The complete flow

```mermaid
flowchart TD
    A[PDF] --> B[Extract page text]
    B --> C[Clean]
    C --> D[Chunk: 800 / 150]
    D --> E[Embed: 384 numbers]
    E --> F[(FAISS)]
    Q[Question] --> G[Embed question]
    G --> F
    F --> H[Top 4 chunks]
    H --> I[Prompt: rules, context, question]
    I --> J[Gemini]
    J --> K[Validate citations]
    K --> L[Answer with sources]
```
