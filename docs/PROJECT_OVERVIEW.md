# Project Overview

## What ContractIQ is

ContractIQ is an AI-powered contract intelligence assistant. It lets a user upload PDF contracts, ask questions in plain English, extract key terms, and identify clauses that may need closer human review. It belongs to the AI document intelligence category of SaaS products.

This is a portfolio-level MVP. It demonstrates how the components of a document intelligence product fit together. It is not a production legal technology platform.

## The business problem

Businesses rely on long contracts that hold important commitments, including:

- Payment obligations
- Contract duration
- Termination conditions
- Renewal clauses
- Penalties and liabilities
- Confidentiality obligations
- Responsibilities of each party

Reviewing these manually takes time, and terms can be missed. A renewal notice window or a one-sided liability cap may sit on page 14 of a long PDF. Keyword search helps only if the user knows the wording, because a contract may say "non-renewal notice" when the user searches for "how do I cancel".

## Target users

| User | Typical need |
|---|---|
| Business Analysts | Find specific terms across several agreements |
| Procurement Professionals | Compare vendor commitments and renewal dates |
| Contract Managers | Summarise agreements and track key clauses |
| Operations Managers | Understand service obligations |
| Legal Teams | Triage documents before a detailed review |

## Proposed solution

ContractIQ combines semantic search with a large language model through retrieval-augmented generation (RAG). The model answers from the passages retrieved from the user's contract, and each answer shows the supporting document, page and text. The user can check an answer rather than having to trust it.

## Core features

1. **Document upload:** multiple PDFs, validation and duplicate detection.
2. **Parsing and chunking:** page-aware text extraction and chunking.
3. **Semantic search:** Hugging Face embeddings in a local FAISS index.
4. **AI Assistant:** grounded answers with source citations.
5. **Contract Intelligence:** an executive summary and nine extracted fields.
6. **Risk identification:** findings graded High, Medium or Low Attention.

## Project objectives

The project demonstrates practical knowledge of:

- Large language models and prompt engineering
- Retrieval-augmented generation
- Document parsing, text preprocessing and chunking
- Hugging Face embeddings and vector databases
- Semantic similarity search
- Context-grounded generation
- LLM-based summarisation and structured extraction

It also shows product thinking: defining a narrow scope, designing for trust, and being clear about what is and is not verified.

## MVP boundaries

**In scope:** text-based PDFs, a single local user, session-based chat, local vector storage and a modular LLM provider.

**Out of scope:** OCR, DOCX or Excel support, multi-agent systems, model fine-tuning, voice, external legal databases, web scraping, authentication, multi-tenancy, payments, cloud deployment, real-time collaboration, email integrations, contract signing or modification, and complex analytics.

ContractIQ offers informational document review. It does not give legal advice, and it does not decide whether a clause is valid or enforceable.
