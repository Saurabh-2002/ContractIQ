"""Structured extraction of nine key contract fields."""

from __future__ import annotations

from dataclasses import dataclass

from core.document_manager import DocumentRecord
from core.llm_client import LLMClient
from core.rag_pipeline import DocumentContext
from utils.parsing import generate_json, locate_quote, unsupported_figures
from utils.prompts import CLAUSE_SYSTEM_PROMPT, CLAUSE_USER_TEMPLATE

CLAUSE_FIELDS: list[tuple[str, str]] = [
    ("contract_title", "Contract Title"),
    ("contracting_parties", "Contracting Parties"),
    ("effective_date", "Effective Date"),
    ("contract_duration", "Contract Duration"),
    ("payment_terms", "Payment Terms"),
    ("renewal_conditions", "Renewal Conditions"),
    ("termination_conditions", "Termination Conditions"),
    ("confidentiality_obligations", "Confidentiality Obligations"),
    ("penalty_or_liability_clauses", "Penalty or Liability Clauses"),
]

NOT_FOUND_VALUE = "Not found in this document"


@dataclass(frozen=True)
class ExtractedField:
    """``status`` is ``found``, ``not_found`` or ``needs_review``."""

    key: str
    label: str
    value: str
    status: str
    page: int | None = None
    quote: str = ""
    note: str = ""


def _validate_field(
    key: str, label: str, item: object, record: DocumentRecord
) -> ExtractedField:
    """Turn one raw model item into a field, verifying it against the text."""
    data = item if isinstance(item, dict) else {}
    value = str(data.get("value") or "").strip()
    quote = str(data.get("quote") or "").strip()

    if data.get("status") != "found" or not value:
        return ExtractedField(key, label, NOT_FOUND_VALUE, "not_found")

    page = locate_quote(quote, record.pages)
    if page is None:
        return ExtractedField(
            key,
            label,
            value,
            "needs_review",
            note="The supporting text could not be matched to the document.",
        )

    figures = unsupported_figures(value, record.pages)
    if figures:
        return ExtractedField(
            key,
            label,
            value,
            "needs_review",
            page,
            quote,
            f"These figures do not appear in the document: {', '.join(figures)}.",
        )
    return ExtractedField(key, label, value, "found", page, quote)


def extract_clauses(
    llm: LLMClient, record: DocumentRecord, context: DocumentContext
) -> list[ExtractedField]:
    """Extract all fields. Every field is returned, including missing ones."""
    system = CLAUSE_SYSTEM_PROMPT.format(
        field_keys=", ".join(key for key, _ in CLAUSE_FIELDS)
    )
    user = CLAUSE_USER_TEMPLATE.format(filename=record.filename, context=context.text)
    data = generate_json(llm, system, user)
    if not isinstance(data, dict):
        data = {}
    return [
        _validate_field(key, label, data.get(key), record)
        for key, label in CLAUSE_FIELDS
    ]
