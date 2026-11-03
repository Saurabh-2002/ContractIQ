"""Executive summary generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.document_manager import DocumentRecord
from core.llm_client import LLMClient
from core.rag_pipeline import DocumentContext
from utils.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE

_PAGE_REFERENCE = re.compile(r"\s*\(p(?:age|\.)?\s*(\d+)\)", re.IGNORECASE)


@dataclass(frozen=True)
class SummaryResult:
    text: str
    is_partial: bool


def remove_invalid_page_references(text: str, page_count: int) -> str:
    """Delete references to pages that do not exist in the document."""

    def replace(match: re.Match) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= page_count else ""

    return _PAGE_REFERENCE.sub(replace, text)


def generate_summary(
    llm: LLMClient, record: DocumentRecord, context: DocumentContext
) -> SummaryResult:
    prompt = SUMMARY_USER_TEMPLATE.format(
        filename=record.filename, context=context.text
    )
    reply = llm.generate(SUMMARY_SYSTEM_PROMPT, prompt)
    return SummaryResult(
        text=remove_invalid_page_references(reply, record.page_count).strip(),
        is_partial=context.is_partial,
    )
