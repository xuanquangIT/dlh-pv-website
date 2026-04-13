"""Evidence store service helpers for Solar AI Chat.

Wraps the EvidenceStore schema with ranking, dedup, and
source-reliability scoring utilities used by the orchestrator.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

from app.schemas.solar_ai_chat.agent import EvidenceItem, EvidenceStore, ToolResultEnvelope

logger = logging.getLogger(__name__)

# Source trust tiers: higher = more trusted
_SOURCE_TRUST: dict[str, float] = {
    "databricks": 1.0,
    "pgvector": 0.85,
    "web-search": 0.6,
    "deterministic": 0.95,
}

# How old (in seconds) before an evidence item is considered stale
_FRESHNESS_THRESHOLD_SECONDS = 3600.0


def _trust_score(data_source: str) -> float:
    return _SOURCE_TRUST.get(data_source.lower(), 0.5)


def _freshness_score(timestamp: str | None) -> float:
    if not timestamp:
        return 0.9  # No timestamp → assume live fetch, near-fresh
    try:
        ts = datetime.datetime.fromisoformat(timestamp)
        age = (datetime.datetime.utcnow() - ts).total_seconds()
        if age <= 0:
            return 1.0
        return max(0.0, 1.0 - age / _FRESHNESS_THRESHOLD_SECONDS)
    except (ValueError, TypeError):
        return 0.8


def score_evidence(item: EvidenceItem) -> float:
    """Combined score: confidence × trust × freshness."""
    return item.confidence * _trust_score(item.data_source) * _freshness_score(item.timestamp)


def build_evidence_from_envelope(
    envelope: ToolResultEnvelope,
    tool_name: str,
    *,
    confidence_override: float | None = None,
) -> list[EvidenceItem]:
    """Convert a ToolResultEnvelope into EvidenceItem list."""
    now = datetime.datetime.utcnow().isoformat()
    items: list[EvidenceItem] = []

    if envelope.status == "error":
        return items  # Don't add failed tool results as evidence

    confidence = confidence_override if confidence_override is not None else envelope.confidence

    # Empty search_documents results (no matching chunks) must not be counted
    # as "sufficient" evidence — doing so would stop the orchestrator before
    # the real Databricks tool is ever executed.  Assign near-zero confidence
    # so score_evidence() falls below the min_score threshold.
    if tool_name == "search_documents":
        chunks = envelope.data.get("chunks")
        if not chunks:  # None, empty list, or missing key
            confidence = 0.05

    for src in envelope.sources:
        items.append(
            EvidenceItem(
                source=src.get("dataset", tool_name),
                tool=tool_name,
                confidence=confidence,
                timestamp=now,
                payload=dict(envelope.data),
                data_source=src.get("data_source", "databricks"),
            )
        )

    if not envelope.sources:
        items.append(
            EvidenceItem(
                source=tool_name,
                tool=tool_name,
                confidence=confidence,
                timestamp=now,
                payload=dict(envelope.data),
                data_source="databricks",
            )
        )

    return items


def rank_and_dedup(store: EvidenceStore) -> EvidenceStore:
    """Return a new EvidenceStore with items ranked by score, duplicates removed."""
    seen_payloads: set[int] = set()
    ranked: list[tuple[float, EvidenceItem]] = []

    for item in store.items:
        payload_hash = hash(str(sorted(item.payload.items())))
        if payload_hash in seen_payloads:
            continue
        seen_payloads.add(payload_hash)
        ranked.append((score_evidence(item), item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    deduped = EvidenceStore(items=[item for _, item in ranked])
    logger.debug(
        "evidence_rank_dedup original=%d deduped=%d",
        len(store.items),
        len(deduped.items),
    )
    return deduped


def evidence_is_sufficient(store: EvidenceStore, min_items: int = 1, min_score: float = 0.4) -> bool:
    """True when store has enough high-quality evidence to attempt answer synthesis."""
    qualifying = [i for i in store.items if score_evidence(i) >= min_score]
    return len(qualifying) >= min_items
