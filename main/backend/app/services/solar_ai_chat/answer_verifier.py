"""Answer verifier for Solar AI Chat.

Workstream C/D: checks that a candidate answer is grounded in the
collected evidence before it is returned to the user.

The verifier runs a lightweight LLM call when SOLAR_AI_VERIFIER_ENABLED=True.
When the LLM is unavailable it falls back to a heuristic keyword overlap check.

Returned VerifierResult signals:
- is_grounded: True when answer appears sufficiently supported
- unsupported_claims: list of detected unsupported claim snippets
- suggestion: optional guidance for the finalizer
- request_more_retrieval: True when the verifier believes another retrieval
  pass is warranted (triggers a second orchestrator loop in chat_service)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.schemas.solar_ai_chat.agent import EvidenceStore

if TYPE_CHECKING:
    from app.services.solar_ai_chat.llm_client import LLMModelRouter

logger = logging.getLogger(__name__)

_UNCERTAINTY_PHRASES = (
    "i don't know",
    "i'm not sure",
    "no data available",
    "unavailable",
    "không có dữ liệu",
    "không biết",
    "không chắc",
    "chưa có thông tin",
)

_MIN_OVERLAP_RATIO = 0.15  # heuristic: at least 15% keyword overlap required


@dataclass
class VerifierResult:
    is_grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)
    suggestion: str | None = None
    request_more_retrieval: bool = False
    method: str = "heuristic"


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]{3,}", text)}


def _evidence_text(store: EvidenceStore) -> str:
    parts: list[str] = []
    for item in store.items:
        for v in item.payload.values():
            parts.append(str(v))
    return " ".join(parts)


def _heuristic_check(answer: str, evidence_text: str) -> VerifierResult:
    """Keyword overlap heuristic when LLM verifier is unavailable."""
    if not evidence_text.strip():
        # No evidence at all — check if answer says so
        lower = answer.lower()
        if any(p in lower for p in _UNCERTAINTY_PHRASES):
            return VerifierResult(is_grounded=True, method="heuristic_no_evidence_ack")
        return VerifierResult(
            is_grounded=False,
            unsupported_claims=["Answer makes claims but no evidence was retrieved."],
            suggestion="Consider triggering another retrieval pass.",
            request_more_retrieval=True,
            method="heuristic_no_evidence",
        )

    answer_tokens = _tokenize(answer)
    evidence_tokens = _tokenize(evidence_text)

    if not answer_tokens:
        return VerifierResult(is_grounded=True, method="heuristic_empty_answer")

    overlap = answer_tokens & evidence_tokens
    ratio = len(overlap) / len(answer_tokens)

    if ratio >= _MIN_OVERLAP_RATIO:
        return VerifierResult(is_grounded=True, method="heuristic_overlap")

    return VerifierResult(
        is_grounded=False,
        unsupported_claims=[
            f"Low keyword overlap ({ratio:.0%}) between answer and retrieved evidence."
        ],
        suggestion="Evidence may be insufficient for this answer.",
        request_more_retrieval=True,
        method="heuristic_overlap",
    )


def _build_verifier_prompt(answer: str, evidence_summary: str) -> str:
    return f"""You are a grounding verifier for a solar-energy analytics assistant.

## Retrieved evidence (summary)
{evidence_summary[:1500] if evidence_summary else "(none)"}

## Candidate answer
{answer[:800]}

## Task
1. Check whether the candidate answer is supported by the evidence above.
2. List any specific claims in the answer that are NOT supported (max 3 items).
3. Decide whether another retrieval pass is needed.

## Output (JSON only, no markdown fences)
{{
  "is_grounded": <true|false>,
  "unsupported_claims": ["<claim 1>", ...],
  "suggestion": "<optional guidance>",
  "request_more_retrieval": <true|false>
}}"""


class AnswerVerifier:
    """Checks answer grounding against collected evidence."""

    def __init__(
        self,
        model_router: "LLMModelRouter | None" = None,
        *,
        max_output_tokens: int | None = None,
    ) -> None:
        self._model_router = model_router
        self._max_output_tokens = max_output_tokens

    def verify(self, answer: str, evidence: EvidenceStore) -> VerifierResult:
        ev_text = _evidence_text(evidence)

        if self._model_router is None:
            return _heuristic_check(answer, ev_text)

        try:
            import json
            prompt = _build_verifier_prompt(answer, ev_text)
            result = self._model_router.generate(
                prompt,
                max_output_tokens=self._max_output_tokens,
                temperature=0.0,
            )
            raw = result.text.strip()
            # Strip markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON in verifier response.")
            data: dict[str, Any] = json.loads(raw[start : end + 1])

            return VerifierResult(
                is_grounded=bool(data.get("is_grounded", True)),
                unsupported_claims=list(data.get("unsupported_claims", [])),
                suggestion=data.get("suggestion"),
                request_more_retrieval=bool(data.get("request_more_retrieval", False)),
                method="llm",
            )
        except Exception as exc:
            logger.warning("answer_verifier_llm_failed error=%s; falling back to heuristic", exc)
            return _heuristic_check(answer, ev_text)
