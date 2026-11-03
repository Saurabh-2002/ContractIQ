"""Provider-neutral LLM access.

The rest of the application depends only on ``LLMClient``. To add a provider,
write one class with a ``generate`` method and register it in
``create_llm_client``. The RAG pipeline does not change.
"""

from __future__ import annotations

from typing import Any, Protocol

from config.settings import Settings, get_settings


class LLMError(Exception):
    """An LLM failure. The message is safe to show to the user."""


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text, or raise ``LLMError``."""
        ...


def content_to_text(content: Any) -> str:
    """Flatten a LangChain message's content into plain text.

    Content can be a string or a list of blocks, depending on the model and
    library version. Non-text blocks are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def describe_llm_failure(error: Exception) -> str:
    """Map a provider exception to a clear message.

    Matching on text keeps this independent of SDK exception classes, which
    vary across versions. The exception text is never shown, because it could
    echo prompt content.
    """
    detail = f"{type(error).__name__} {error}".lower()
    if any(s in detail for s in ("api key", "api_key", "401", "403", "permission")):
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if any(s in detail for s in ("429", "resource_exhausted", "quota", "rate limit")):
        return "The Gemini rate limit was reached. Wait a minute and try again."
    if any(s in detail for s in ("404", "not_found", "not found")):
        return (
            "The Gemini model was not found. It may have been retired. "
            "Set GEMINI_MODEL in your .env file to a current model."
        )
    if any(s in detail for s in ("timeout", "timed out", "deadline")):
        return "Gemini took too long to respond. Please try again."
    if any(s in detail for s in ("connect", "network", "dns", "unavailable", "503")):
        return "Gemini could not be reached. Check your internet connection."
    return "The Gemini request failed. Please try again."


class GeminiClient:
    """Google Gemini through LangChain."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None

    def _load(self):
        if self._model is None:
            if not self.settings.has_api_key:
                raise LLMError(
                    "No Gemini API key found. Add GEMINI_API_KEY to your .env "
                    "file and restart the app."
                )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._model = ChatGoogleGenerativeAI(
                    model=self.settings.gemini_model,
                    google_api_key=self.settings.gemini_api_key,
                    temperature=self.settings.llm_temperature,
                    timeout=self.settings.llm_timeout_seconds,
                    max_retries=self.settings.llm_max_retries,
                )
            except Exception:
                raise LLMError(
                    "The Gemini client could not be started. Check that the "
                    "packages in requirements.txt are installed."
                ) from None
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        model = self._load()
        try:
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            raise LLMError(describe_llm_failure(exc)) from None

        text = content_to_text(response.content).strip()
        if not text:
            raise LLMError(
                "Gemini returned an empty response. It may have been blocked "
                "by a safety filter. Please try again."
            )
        return text


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the configured provider's client."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return GeminiClient(settings)
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")
