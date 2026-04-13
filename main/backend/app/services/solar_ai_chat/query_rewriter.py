"""Query rewriter for Solar AI Chat.

Responsibilities:
- Language detection (Vietnamese / English)
- Input normalisation

Compound query decomposition has been removed: the LLM receives
_LAKEHOUSE_ARCHITECTURE_CONTEXT and handles multi-part questions natively
through the agentic tool-calling loop (see prompt_builder.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from unicodedata import normalize


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    without_marks = normalize("NFD", lowered)
    return "".join(c for c in without_marks if ord(c) < 128)


def detect_language(text: str) -> str:
    """Return 'vi' if Vietnamese diacritics detected, else 'en'."""
    return "vi" if any(ord(c) > 127 for c in text) else "en"


@dataclass
class RewriteResult:
    original: str
    normalized: str
    language: str


class QueryRewriter:
    """Normalise and detect the language of a user query."""

    def rewrite(self, message: str) -> RewriteResult:
        if not message or not message.strip():
            return RewriteResult(original=message, normalized="", language="en")
        return RewriteResult(
            original=message,
            normalized=_normalize(message),
            language=detect_language(message),
        )
