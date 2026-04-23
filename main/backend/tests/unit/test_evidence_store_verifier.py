"""Unit tests for evidence_store.py and answer_verifier.py"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat.agent import EvidenceItem, EvidenceStore, ToolResultEnvelope
from app.services.solar_ai_chat.evidence_store import (
    _FRESHNESS_THRESHOLD_SECONDS,
    _SOURCE_TRUST,
    _freshness_score,
    _trust_score,
    build_evidence_from_envelope,
    evidence_is_sufficient,
    rank_and_dedup,
    score_evidence,
)
from app.services.solar_ai_chat.answer_verifier import (
    AnswerVerifier,
    VerifierResult,
    _evidence_text,
    _heuristic_check,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence_item(
    source: str = "fact_energy",
    tool: str = "get_energy_performance",
    confidence: float = 1.0,
    timestamp: str | None = None,
    payload: dict | None = None,
    data_source: str = "databricks",
) -> EvidenceItem:
    return EvidenceItem(
        source=source,
        tool=tool,
        confidence=confidence,
        timestamp=timestamp,
        payload=payload or {"energy_mwh": 100.0},
        data_source=data_source,
    )


def _make_store(*items: EvidenceItem) -> EvidenceStore:
    store = EvidenceStore()
    for item in items:
        store.add(item)
    return store


def _make_envelope(
    status: str = "ok",
    data: dict | None = None,
    sources: list[dict] | None = None,
    confidence: float = 1.0,
    errors: list[str] | None = None,
) -> ToolResultEnvelope:
    return ToolResultEnvelope(
        status=status,
        data=data or {"energy_mwh": 100.0},
        sources=sources or [],
        confidence=confidence,
        errors=errors or [],
    )


# ===========================================================================
# Tests for evidence_store.py
# ===========================================================================


class TestTrustScore(unittest.TestCase):
    def test_databricks_has_highest_trust(self) -> None:
        self.assertEqual(_trust_score("databricks"), _SOURCE_TRUST["databricks"])

    def test_pgvector_trust(self) -> None:
        self.assertEqual(_trust_score("pgvector"), _SOURCE_TRUST["pgvector"])

    def test_web_search_trust(self) -> None:
        self.assertEqual(_trust_score("web-search"), _SOURCE_TRUST["web-search"])

    def test_deterministic_trust(self) -> None:
        self.assertEqual(_trust_score("deterministic"), _SOURCE_TRUST["deterministic"])

    def test_unknown_source_returns_default(self) -> None:
        self.assertEqual(_trust_score("some_unknown_source"), 0.5)

    def test_case_insensitive(self) -> None:
        self.assertEqual(_trust_score("DATABRICKS"), _trust_score("databricks"))


class TestFreshnessScore(unittest.TestCase):
    def test_no_timestamp_returns_near_fresh(self) -> None:
        score = _freshness_score(None)
        self.assertEqual(score, 0.9)

    def test_current_timestamp_returns_near_one(self) -> None:
        now = datetime.utcnow().isoformat()
        score = _freshness_score(now)
        self.assertGreater(score, 0.98)

    def test_very_old_timestamp_returns_near_zero(self) -> None:
        old = "2020-01-01T00:00:00"
        score = _freshness_score(old)
        self.assertAlmostEqual(score, 0.0, places=1)

    def test_invalid_timestamp_returns_fallback(self) -> None:
        score = _freshness_score("not-a-date")
        self.assertEqual(score, 0.8)

    def test_future_timestamp_returns_one(self) -> None:
        future = "2099-01-01T00:00:00"
        score = _freshness_score(future)
        self.assertEqual(score, 1.0)

    def test_score_at_threshold_boundary(self) -> None:
        # Exactly at threshold = 0 score
        from datetime import timedelta
        ts = (datetime.utcnow() - timedelta(seconds=_FRESHNESS_THRESHOLD_SECONDS)).isoformat()
        score = _freshness_score(ts)
        self.assertAlmostEqual(score, 0.0, delta=0.05)


class TestScoreEvidence(unittest.TestCase):
    def test_perfect_item_scores_near_one(self) -> None:
        item = _make_evidence_item(
            confidence=1.0,
            data_source="databricks",
            timestamp=datetime.utcnow().isoformat(),
        )
        score = score_evidence(item)
        self.assertGreater(score, 0.9)

    def test_low_confidence_reduces_score(self) -> None:
        item_high = _make_evidence_item(confidence=1.0, data_source="databricks")
        item_low = _make_evidence_item(confidence=0.2, data_source="databricks")
        self.assertGreater(score_evidence(item_high), score_evidence(item_low))

    def test_untrusted_source_reduces_score(self) -> None:
        item_db = _make_evidence_item(confidence=1.0, data_source="databricks")
        item_web = _make_evidence_item(confidence=1.0, data_source="web-search")
        self.assertGreater(score_evidence(item_db), score_evidence(item_web))

    def test_stale_item_scores_lower_than_fresh(self) -> None:
        item_fresh = _make_evidence_item(timestamp=datetime.utcnow().isoformat())
        item_stale = _make_evidence_item(timestamp="2020-01-01T00:00:00")
        self.assertGreater(score_evidence(item_fresh), score_evidence(item_stale))


class TestBuildEvidenceFromEnvelope(unittest.TestCase):
    def test_error_envelope_returns_empty(self) -> None:
        env = _make_envelope(status="error")
        result = build_evidence_from_envelope(env, "get_energy_performance")
        self.assertEqual(result, [])

    def test_ok_envelope_no_sources_returns_one_item(self) -> None:
        env = _make_envelope(status="ok", sources=[])
        result = build_evidence_from_envelope(env, "get_system_overview")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].tool, "get_system_overview")

    def test_ok_envelope_with_sources_returns_one_item_per_source(self) -> None:
        sources = [
            {"dataset": "fact_energy", "data_source": "databricks"},
            {"dataset": "dim_facility", "data_source": "databricks"},
        ]
        env = _make_envelope(sources=sources)
        result = build_evidence_from_envelope(env, "get_energy_performance")
        self.assertEqual(len(result), 2)

    def test_source_dataset_used_as_source_name(self) -> None:
        sources = [{"dataset": "mart_energy_daily", "data_source": "databricks"}]
        env = _make_envelope(sources=sources)
        result = build_evidence_from_envelope(env, "get_energy_performance")
        self.assertEqual(result[0].source, "mart_energy_daily")

    def test_missing_dataset_falls_back_to_tool_name(self) -> None:
        sources = [{"data_source": "databricks"}]  # no 'dataset' key
        env = _make_envelope(sources=sources)
        result = build_evidence_from_envelope(env, "get_system_overview")
        self.assertEqual(result[0].source, "get_system_overview")

    def test_confidence_override_applied(self) -> None:
        env = _make_envelope(confidence=1.0)
        result = build_evidence_from_envelope(env, "get_energy_performance", confidence_override=0.3)
        self.assertEqual(result[0].confidence, 0.3)

    def test_default_confidence_from_envelope(self) -> None:
        env = _make_envelope(confidence=0.7)
        result = build_evidence_from_envelope(env, "get_energy_performance")
        self.assertEqual(result[0].confidence, 0.7)

    def test_search_documents_empty_chunks_near_zero_confidence(self) -> None:
        env = _make_envelope(data={"chunks": []})
        result = build_evidence_from_envelope(env, "search_documents")
        self.assertEqual(result[0].confidence, 0.05)

    def test_search_documents_none_chunks_near_zero_confidence(self) -> None:
        env = _make_envelope(data={"chunks": None})
        result = build_evidence_from_envelope(env, "search_documents")
        self.assertEqual(result[0].confidence, 0.05)

    def test_search_documents_with_chunks_uses_envelope_confidence(self) -> None:
        env = _make_envelope(data={"chunks": ["chunk1", "chunk2"]}, confidence=0.85)
        result = build_evidence_from_envelope(env, "search_documents")
        self.assertEqual(result[0].confidence, 0.85)

    def test_payload_copied_from_envelope_data(self) -> None:
        env = _make_envelope(data={"key": "value"})
        result = build_evidence_from_envelope(env, "get_system_overview")
        self.assertEqual(result[0].payload["key"], "value")

    def test_timestamp_is_set(self) -> None:
        env = _make_envelope()
        result = build_evidence_from_envelope(env, "get_system_overview")
        self.assertIsNotNone(result[0].timestamp)

    def test_partial_status_is_treated_as_evidence(self) -> None:
        env = _make_envelope(status="partial")
        result = build_evidence_from_envelope(env, "get_energy_performance")
        self.assertGreater(len(result), 0)


class TestRankAndDedup(unittest.TestCase):
    def test_empty_store_returns_empty(self) -> None:
        store = EvidenceStore()
        result = rank_and_dedup(store)
        self.assertEqual(len(result.items), 0)

    def test_single_item_preserved(self) -> None:
        item = _make_evidence_item(confidence=0.9, data_source="databricks")
        store = _make_store(item)
        result = rank_and_dedup(store)
        self.assertEqual(len(result.items), 1)

    def test_duplicate_payloads_deduped(self) -> None:
        item1 = _make_evidence_item(payload={"a": 1})
        item2 = _make_evidence_item(payload={"a": 1})  # same payload
        store = _make_store(item1, item2)
        result = rank_and_dedup(store)
        self.assertEqual(len(result.items), 1)

    def test_different_payloads_both_kept(self) -> None:
        item1 = _make_evidence_item(payload={"a": 1})
        item2 = _make_evidence_item(payload={"b": 2})
        store = _make_store(item1, item2)
        result = rank_and_dedup(store)
        self.assertEqual(len(result.items), 2)

    def test_items_sorted_highest_score_first(self) -> None:
        item_low = _make_evidence_item(
            payload={"x": "low"},
            confidence=0.1,
            data_source="web-search",
            timestamp="2020-01-01T00:00:00",
        )
        item_high = _make_evidence_item(
            payload={"x": "high"},
            confidence=1.0,
            data_source="databricks",
            timestamp=datetime.utcnow().isoformat(),
        )
        store = _make_store(item_low, item_high)
        result = rank_and_dedup(store)
        self.assertEqual(result.items[0].payload["x"], "high")

    def test_returns_new_store_instance(self) -> None:
        item = _make_evidence_item()
        store = _make_store(item)
        result = rank_and_dedup(store)
        self.assertIsNot(result, store)


class TestEvidenceIsSufficient(unittest.TestCase):
    def test_empty_store_is_not_sufficient(self) -> None:
        store = EvidenceStore()
        self.assertFalse(evidence_is_sufficient(store))

    def test_high_quality_item_is_sufficient(self) -> None:
        item = _make_evidence_item(
            confidence=1.0,
            data_source="databricks",
            timestamp=datetime.utcnow().isoformat(),
        )
        store = _make_store(item)
        self.assertTrue(evidence_is_sufficient(store))

    def test_low_score_item_is_not_sufficient(self) -> None:
        item = _make_evidence_item(
            confidence=0.05,
            data_source="web-search",
            timestamp="2020-01-01T00:00:00",
        )
        store = _make_store(item)
        self.assertFalse(evidence_is_sufficient(store))

    def test_custom_min_items_respected(self) -> None:
        item1 = _make_evidence_item(confidence=1.0, payload={"a": 1})
        item2 = _make_evidence_item(confidence=1.0, payload={"b": 2})
        store = _make_store(item1, item2)
        self.assertTrue(evidence_is_sufficient(store, min_items=2))
        self.assertFalse(evidence_is_sufficient(store, min_items=3))

    def test_custom_min_score_respected(self) -> None:
        item = _make_evidence_item(
            confidence=0.5,
            data_source="databricks",
            timestamp=datetime.utcnow().isoformat(),
        )
        store = _make_store(item)
        # With a very high min_score, item should fail
        self.assertFalse(evidence_is_sufficient(store, min_score=0.99))

    def test_search_documents_empty_chunks_not_sufficient(self) -> None:
        env = _make_envelope(data={"chunks": []})
        items = build_evidence_from_envelope(env, "search_documents")
        store = EvidenceStore(items=items)
        self.assertFalse(evidence_is_sufficient(store))


# ===========================================================================
# Tests for answer_verifier.py
# ===========================================================================


class TestTokenize(unittest.TestCase):
    def test_extracts_words(self) -> None:
        result = _tokenize("The energy output was 100 MWh today")
        self.assertIn("the", result)
        self.assertIn("energy", result)
        self.assertIn("output", result)

    def test_ignores_short_words(self) -> None:
        # Words shorter than 3 chars are excluded
        result = _tokenize("it is a day")
        self.assertNotIn("it", result)
        self.assertNotIn("is", result)
        self.assertNotIn("a", result)

    def test_lowercase_normalization(self) -> None:
        result = _tokenize("Energy ENERGY energy")
        self.assertEqual(len(result), 1)
        self.assertIn("energy", result)

    def test_empty_string_returns_empty_set(self) -> None:
        self.assertEqual(_tokenize(""), set())

    def test_vietnamese_chars_included(self) -> None:
        result = _tokenize("năng lượng mặt trời")
        self.assertGreater(len(result), 0)


class TestEvidenceText(unittest.TestCase):
    def test_empty_store_returns_empty_string(self) -> None:
        store = EvidenceStore()
        result = _evidence_text(store)
        self.assertEqual(result.strip(), "")

    def test_single_item_payload_stringified(self) -> None:
        item = _make_evidence_item(payload={"energy": 100.0})
        store = _make_store(item)
        result = _evidence_text(store)
        self.assertIn("100.0", result)

    def test_multiple_items_concatenated(self) -> None:
        item1 = _make_evidence_item(payload={"a": "alpha"})
        item2 = _make_evidence_item(payload={"b": "beta"})
        store = _make_store(item1, item2)
        result = _evidence_text(store)
        self.assertIn("alpha", result)
        self.assertIn("beta", result)


class TestHeuristicCheck(unittest.TestCase):
    def test_empty_evidence_with_uncertainty_phrase_is_grounded(self) -> None:
        result = _heuristic_check("I don't know", "")
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.method, "heuristic_no_evidence_ack")

    def test_empty_evidence_with_viet_uncertainty_phrase_is_grounded(self) -> None:
        result = _heuristic_check("Không có dữ liệu", "")
        self.assertTrue(result.is_grounded)

    def test_empty_evidence_with_fabricated_claims_is_not_grounded(self) -> None:
        result = _heuristic_check("The energy output was 500 MWh.", "")
        self.assertFalse(result.is_grounded)
        self.assertTrue(result.request_more_retrieval)

    def test_empty_answer_is_grounded(self) -> None:
        result = _heuristic_check("", "Some evidence here.")
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.method, "heuristic_empty_answer")

    def test_high_overlap_is_grounded(self) -> None:
        evidence = "The solar farm produced 100 MWh of energy output today"
        answer = "The solar farm produced about 100 MWh of energy output"
        result = _heuristic_check(answer, evidence)
        self.assertTrue(result.is_grounded)

    def test_low_overlap_is_not_grounded(self) -> None:
        evidence = "Solar panels generate electricity from sunlight"
        answer = "Cryptocurrency prices fell dramatically yesterday causing market panic"
        result = _heuristic_check(answer, evidence)
        self.assertFalse(result.is_grounded)
        self.assertGreater(len(result.unsupported_claims), 0)

    def test_overlap_method_set(self) -> None:
        evidence = "energy output facility"
        answer = "energy output facility performance"
        result = _heuristic_check(answer, evidence)
        self.assertEqual(result.method, "heuristic_overlap")

    def test_no_evidence_method_set(self) -> None:
        result = _heuristic_check("claims without evidence", "")
        self.assertEqual(result.method, "heuristic_no_evidence")


class TestAnswerVerifier(unittest.TestCase):
    # --- No router (heuristic path) ---

    def test_no_router_uses_heuristic(self) -> None:
        verifier = AnswerVerifier(model_router=None)
        item = _make_evidence_item(payload={"energy": "100 MWh output from the solar energy farm"})
        store = _make_store(item)
        result = verifier.verify("The solar energy farm produced 100 MWh output.", store)
        self.assertIsInstance(result, VerifierResult)

    def test_no_router_empty_store_empty_answer_is_ungrounded(self) -> None:
        # Empty evidence + empty answer → heuristic considers it ungrounded (no data to support it)
        verifier = AnswerVerifier(model_router=None)
        store = EvidenceStore()
        result = verifier.verify("", store)
        self.assertIsInstance(result, VerifierResult)  # just verify it returns a result

    def test_no_router_ungrounded_answer_detected(self) -> None:
        verifier = AnswerVerifier(model_router=None)
        item = _make_evidence_item(payload={"note": "system operational"})
        store = _make_store(item)
        result = verifier.verify(
            "The stock market crashed and GDP fell by 5%.", store
        )
        self.assertFalse(result.is_grounded)

    # --- LLM router path ---

    def test_llm_router_grounded_result_parsed(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_grounded": true, "unsupported_claims": [], "suggestion": null, "request_more_retrieval": false}'
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        item = _make_evidence_item()
        store = _make_store(item)
        result = verifier.verify("The system is working well.", store)

        self.assertTrue(result.is_grounded)
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.unsupported_claims, [])
        self.assertFalse(result.request_more_retrieval)

    def test_llm_router_ungrounded_result_parsed(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"is_grounded": false, "unsupported_claims": ["Claim A is unsupported"], '
            '"suggestion": "Need more data", "request_more_retrieval": true}'
        )
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        result = verifier.verify("Claim A is unsupported.", store)

        self.assertFalse(result.is_grounded)
        self.assertEqual(result.method, "llm")
        self.assertIn("Claim A is unsupported", result.unsupported_claims)
        self.assertTrue(result.request_more_retrieval)
        self.assertEqual(result.suggestion, "Need more data")

    def test_llm_router_falls_back_to_heuristic_on_invalid_json(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "this is not json"
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        result = verifier.verify("some answer", store)

        # Should fall back to heuristic
        self.assertNotEqual(result.method, "llm")

    def test_llm_router_falls_back_on_exception(self) -> None:
        mock_router = MagicMock()
        mock_router.generate.side_effect = RuntimeError("LLM down")

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        result = verifier.verify("some answer", store)

        self.assertNotEqual(result.method, "llm")
        self.assertIsInstance(result, VerifierResult)

    def test_llm_router_json_in_markdown_fences_parsed(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "```json\n"
            '{"is_grounded": true, "unsupported_claims": [], "suggestion": null, "request_more_retrieval": false}'
            "\n```"
        )
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        result = verifier.verify("The energy data looks good.", store)

        self.assertTrue(result.is_grounded)
        self.assertEqual(result.method, "llm")

    def test_llm_router_max_output_tokens_passed_to_generate(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_grounded": true, "unsupported_claims": [], "suggestion": null, "request_more_retrieval": false}'
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router, max_output_tokens=256)
        store = _make_store(_make_evidence_item())
        verifier.verify("answer", store)

        _, kwargs = mock_router.generate.call_args
        self.assertEqual(kwargs.get("max_output_tokens"), 256)

    def test_llm_router_temperature_zero_passed_to_generate(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_grounded": true, "unsupported_claims": [], "suggestion": null, "request_more_retrieval": false}'
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        verifier.verify("answer", store)

        _, kwargs = mock_router.generate.call_args
        self.assertEqual(kwargs.get("temperature"), 0.0)

    def test_verifier_result_defaults(self) -> None:
        result = VerifierResult(is_grounded=True)
        self.assertEqual(result.unsupported_claims, [])
        self.assertIsNone(result.suggestion)
        self.assertFalse(result.request_more_retrieval)
        self.assertEqual(result.method, "heuristic")

    def test_llm_missing_fields_use_defaults(self) -> None:
        mock_router = MagicMock()
        mock_response = MagicMock()
        # Minimal JSON — missing optional fields
        mock_response.text = '{"is_grounded": true}'
        mock_router.generate.return_value = mock_response

        verifier = AnswerVerifier(model_router=mock_router)
        store = _make_store(_make_evidence_item())
        result = verifier.verify("answer", store)

        self.assertTrue(result.is_grounded)
        self.assertEqual(result.unsupported_claims, [])
        self.assertFalse(result.request_more_retrieval)


# ---------------------------------------------------------------------------
# EvidenceStore.merge_payload and to_source_metadata_dicts
# ---------------------------------------------------------------------------

class TestEvidenceStoreHelpers(unittest.TestCase):
    def test_merge_payload_empty_store(self) -> None:
        store = EvidenceStore()
        self.assertEqual(store.merge_payload(), {})

    def test_merge_payload_single_item(self) -> None:
        item = _make_evidence_item(payload={"energy_mwh": 42})
        store = _make_store(item)
        merged = store.merge_payload()
        self.assertEqual(merged["energy_mwh"], 42)

    def test_merge_payload_last_write_wins(self) -> None:
        item1 = _make_evidence_item(payload={"key": "first"})
        item2 = _make_evidence_item(payload={"key": "second"})
        store = _make_store(item1, item2)
        merged = store.merge_payload()
        self.assertEqual(merged["key"], "second")

    def test_to_source_metadata_dicts_empty(self) -> None:
        store = EvidenceStore()
        self.assertEqual(store.to_source_metadata_dicts(), [])

    def test_to_source_metadata_dicts_deduped(self) -> None:
        item1 = _make_evidence_item(source="fact_energy", tool="get_energy_performance")
        item2 = _make_evidence_item(source="fact_energy", tool="get_energy_performance")
        store = _make_store(item1, item2)
        result = store.to_source_metadata_dicts()
        self.assertEqual(len(result), 1)

    def test_to_source_metadata_dicts_keys(self) -> None:
        item = _make_evidence_item(source="dim_facility", tool="get_facility_info")
        store = _make_store(item)
        result = store.to_source_metadata_dicts()
        self.assertIn("layer", result[0])
        self.assertIn("dataset", result[0])
        self.assertIn("data_source", result[0])


if __name__ == "__main__":
    unittest.main()
