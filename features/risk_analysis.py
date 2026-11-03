"""Identify clauses that may need human review. This is not legal advice."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings, get_settings
from core.document_manager import DocumentRecord
from core.llm_client import LLMClient
from core.rag_pipeline import DocumentContext
from utils.parsing import generate_json, locate_quote
from utils.prompts import RISK_SYSTEM_PROMPT, RISK_USER_TEMPLATE

LEVELS = {"high": "High Attention", "medium": "Medium Attention", "low": "Low Attention"}
LEVEL_ORDER = list(LEVELS.values())


@dataclass(frozen=True)
class RiskFinding:
    """``page`` is None when absent from the contract or not verified."""

    level: str
    issue: str
    reason: str
    filename: str
    page: int | None = None
    quote: str = ""
    is_missing_item: bool = False
    verified: bool = False


def _normalize_level(value: object) -> str:
    text = str(value or "").lower()
    for key, label in LEVELS.items():
        if key in text:
            return label
    return LEVELS["medium"]


def _validate_finding(item: dict, record: DocumentRecord) -> RiskFinding | None:
    issue = str(item.get("issue") or "").strip()
    reason = str(item.get("reason") or "").strip()
    if not issue or not reason:
        return None

    level = _normalize_level(item.get("level"))
    if item.get("type") == "missing":
        return RiskFinding(
            level, issue, reason, record.filename, is_missing_item=True, verified=True
        )

    quote = str(item.get("quote") or "").strip()
    page = locate_quote(quote, record.pages)
    if page is None:
        return RiskFinding(level, issue, reason, record.filename)
    return RiskFinding(level, issue, reason, record.filename, page, quote, verified=True)


def analyze_risks(
    llm: LLMClient,
    record: DocumentRecord,
    context: DocumentContext,
    settings: Settings | None = None,
) -> list[RiskFinding]:
    """Return findings, highest attention first."""
    settings = settings or get_settings()
    system = RISK_SYSTEM_PROMPT.format(max_findings=settings.max_risk_findings)
    user = RISK_USER_TEMPLATE.format(filename=record.filename, context=context.text)
    data = generate_json(llm, system, user)

    items = data.get("findings", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = []

    findings = []
    for item in items:
        if isinstance(item, dict):
            finding = _validate_finding(item, record)
            if finding is not None:
                findings.append(finding)

    findings.sort(key=lambda f: LEVEL_ORDER.index(f.level))
    return findings[: settings.max_risk_findings]
