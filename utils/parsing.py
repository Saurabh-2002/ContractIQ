"""Checks applied to model output before it reaches the user.

The model is not trusted to cite correctly: these helpers confirm its quotes
against the document, so a displayed page number always comes from real text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from core.llm_client import LLMClient, LLMError
from utils.prompts import JSON_RETRY_INSTRUCTION

MIN_QUOTE_CHARS = 8
FUZZY_MATCH_RATIO = 0.9

_TRANSLATIONS = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"',
     "–": "-", "—": "-", " ": " "}
)


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply."""
    text = text.strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON found in the reply.")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return value


def generate_json(llm: LLMClient, system_prompt: str, user_prompt: str) -> Any:
    """Call the model for JSON, retrying once if the reply cannot be parsed."""
    reply = llm.generate(system_prompt, user_prompt)
    try:
        return extract_json(reply)
    except ValueError:
        reply = llm.generate(system_prompt, user_prompt + JSON_RETRY_INSTRUCTION)
    try:
        return extract_json(reply)
    except ValueError:
        raise LLMError(
            "The model did not return a readable result. Please try again."
        ) from None


def normalize_for_match(text: str) -> str:
    """Lower-case text and standardise quotes, dashes and whitespace."""
    return re.sub(r"\s+", " ", text.translate(_TRANSLATIONS)).strip().lower()


def _fuzzy_contains(quote: str, page_text: str) -> bool:
    """Tolerate minor differences, such as words rejoined across a line break."""
    matcher = SequenceMatcher(None, page_text, quote, autojunk=False)
    match = matcher.find_longest_match(0, len(page_text), 0, len(quote))
    if match.size == 0:
        return False
    # Compare against the region of the page aligned with the whole quote.
    start = max(match.a - match.b, 0)
    window = page_text[start : start + len(quote) + 10]
    # A changed number is a different claim, however similar the wording is.
    if not set(re.findall(r"\d+", quote)) <= set(re.findall(r"\d+", window)):
        return False
    return SequenceMatcher(None, window, quote, autojunk=False).ratio() >= (
        FUZZY_MATCH_RATIO
    )


def locate_quote(quote: str, pages: Iterable[dict]) -> int | None:
    """Return the page containing ``quote``, or None if it is not in the text."""
    needle = normalize_for_match(quote or "")
    if len(needle) < MIN_QUOTE_CHARS:
        return None

    normalized = [
        (page["page_number"], normalize_for_match(page["text"])) for page in pages
    ]
    for page_number, text in normalized:
        if needle in text:
            return page_number
    for page_number, text in normalized:
        if _fuzzy_contains(needle, text):
            return page_number
    return None


def unsupported_figures(value: str, pages: Iterable[dict]) -> list[str]:
    """Return numbers in ``value`` that appear nowhere in the document."""
    document = " ".join(page["text"] for page in pages).replace(",", "")
    known = set(re.findall(r"\d+(?:\.\d+)?", document))
    figures = re.findall(r"\d[\d,]*(?:\.\d+)?", value or "")
    return [f for f in figures if f.replace(",", "") not in known]


def escape_markdown(text: str) -> str:
    """Show document text literally, whatever formatting characters it holds."""
    flat = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>$~:])", r"\\\1", flat)


def sanitize_markdown(text: str) -> str:
    """Remove model-supplied images and links before rendering.

    A hostile document could otherwise make the model emit a remote image whose
    URL leaks contract text when the browser loads it.
    """
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\[([^\]]*)\]\((?:[^)]*)\)", r"\1", text)
