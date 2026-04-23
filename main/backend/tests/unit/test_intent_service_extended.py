"""Extended unit tests for intent_service.py targeting missing coverage lines.

Targets:
- Lines 314-335: initialize_semantic_router (success + exception path)
- Lines 338-343: _cosine_similarity (zero-vector edge cases)
- Lines 348, 352: detect_intent (empty message, cache hit)
- Lines 358-378: detect_intent semantic routing branch
- Lines 393, 399, 406-407: semantic override / fallback paths
- Lines 414-419: matched_topic with low score (below threshold)
- Line 462: cache eviction at limit
- Lines 498, 511, 513-514: keyword_match bias branches
- All topic keyword detections for every ChatTopic value
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat import ChatTopic
from app.services.solar_ai_chat.intent_service import (
    IntentDetectionResult,
    VietnameseIntentService,
    _INTENT_CACHE_LIMIT,
    normalize_vietnamese_text,
)


# ---------------------------------------------------------------------------
# normalize_vietnamese_text
# ---------------------------------------------------------------------------

class TestNormalizeVietnameseText:
    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_vietnamese_text("  hello  ") == "hello"

    def test_lowercases_input(self) -> None:
        assert normalize_vietnamese_text("HELLO WORLD") == "hello world"

    def test_removes_tone_marks_basic(self) -> None:
        result = normalize_vietnamese_text("Năng lượng")
        assert result == "nang luong"

    def test_removes_complex_vietnamese_diacritics(self) -> None:
        result = normalize_vietnamese_text("Chất lượng dữ liệu")
        assert result == "chat luong du lieu"

    def test_result_is_ascii_only(self) -> None:
        result = normalize_vietnamese_text("Nhiệt độ tối đa hiệu suất")
        assert result.isascii()

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_vietnamese_text("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert normalize_vietnamese_text("   ") == ""

    def test_plain_ascii_unchanged(self) -> None:
        assert normalize_vietnamese_text("pipeline status") == "pipeline status"

    def test_mixed_language_strips_diacritics(self) -> None:
        result = normalize_vietnamese_text("AQI và energy performance")
        assert result.isascii()
        assert "aqi" in result
        assert "energy performance" in result

    def test_special_chars_oe_ligature(self) -> None:
        # 'đ' (U+0111) should be stripped to 'd' after NFD + ASCII filter
        result = normalize_vietnamese_text("độ")
        assert result.isascii()


# ---------------------------------------------------------------------------
# Topic detection — all ChatTopic values
# ---------------------------------------------------------------------------

class TestTopicDetectionAllTopics:
    """Ensure every ChatTopic can be detected via keyword matching."""

    @pytest.fixture(autouse=True)
    def _service(self) -> None:
        self.svc = VietnameseIntentService(semantic_enabled=False)

    # --- DATA_QUALITY_ISSUES ---
    def test_data_quality_via_aqi_keyword(self) -> None:
        r = self.svc.detect_intent("chi so aqi hom nay")
        assert r.topic == ChatTopic.DATA_QUALITY_ISSUES

    def test_data_quality_via_quality_alert_english(self) -> None:
        r = self.svc.detect_intent("data quality alert found today")
        assert r.topic == ChatTopic.DATA_QUALITY_ISSUES

    def test_data_quality_via_low_score_facility(self) -> None:
        r = self.svc.detect_intent("Which facilities have low score data quality")
        assert r.topic == ChatTopic.DATA_QUALITY_ISSUES

    def test_data_quality_via_canh_bao_du_lieu(self) -> None:
        r = self.svc.detect_intent("canh bao du lieu ngay hom nay")
        assert r.topic == ChatTopic.DATA_QUALITY_ISSUES

    # --- FORECAST_72H ---
    def test_forecast_72h_via_72h_keyword(self) -> None:
        r = self.svc.detect_intent("show forecast for next 72 hours")
        assert r.topic == ChatTopic.FORECAST_72H

    def test_forecast_72h_via_ba_ngay(self) -> None:
        r = self.svc.detect_intent("du bao san luong ba ngay tiep theo")
        assert r.topic == ChatTopic.FORECAST_72H

    def test_forecast_72h_via_confidence_interval(self) -> None:
        r = self.svc.detect_intent("what is the 72h confidence interval")
        assert r.topic == ChatTopic.FORECAST_72H

    def test_forecast_72h_via_three_day_forecast(self) -> None:
        r = self.svc.detect_intent("give me the three-day forecast please")
        assert r.topic == ChatTopic.FORECAST_72H

    # --- PIPELINE_STATUS ---
    def test_pipeline_status_via_pipeline_keyword(self) -> None:
        r = self.svc.detect_intent("pipeline trang thai hien tai nhu the nao")
        assert r.topic == ChatTopic.PIPELINE_STATUS

    def test_pipeline_status_via_eta(self) -> None:
        r = self.svc.detect_intent("eta for the current pipeline run")
        assert r.topic == ChatTopic.PIPELINE_STATUS

    def test_pipeline_status_via_canh_bao(self) -> None:
        r = self.svc.detect_intent("co canh bao nao tu pipeline khong")
        assert r.topic == ChatTopic.PIPELINE_STATUS

    # --- ML_MODEL ---
    def test_ml_model_via_model_keyword(self) -> None:
        r = self.svc.detect_intent("model performance metrics hien tai")
        assert r.topic == ChatTopic.ML_MODEL

    def test_ml_model_via_champion_model(self) -> None:
        r = self.svc.detect_intent("what is the champion model being used")
        assert r.topic == ChatTopic.ML_MODEL

    def test_ml_model_via_nrmse(self) -> None:
        r = self.svc.detect_intent("give me nrmse and skill score of the current model")
        assert r.topic == ChatTopic.ML_MODEL

    def test_ml_model_via_r_squared(self) -> None:
        r = self.svc.detect_intent("model r-squared va r2 hien tai")
        assert r.topic == ChatTopic.ML_MODEL

    def test_ml_model_via_fallback(self) -> None:
        r = self.svc.detect_intent("is the system using a fallback model version")
        assert r.topic == ChatTopic.ML_MODEL

    # --- ENERGY_PERFORMANCE ---
    def test_energy_performance_via_performance_keyword(self) -> None:
        r = self.svc.detect_intent("hieu suat nang luong hom nay")
        assert r.topic == ChatTopic.ENERGY_PERFORMANCE

    def test_energy_performance_via_compare_top(self) -> None:
        r = self.svc.detect_intent("compare top 3 facilities by energy output")
        assert r.topic == ChatTopic.ENERGY_PERFORMANCE

    def test_energy_performance_via_peak_hour(self) -> None:
        r = self.svc.detect_intent("peak hour energy production today")
        assert r.topic == ChatTopic.ENERGY_PERFORMANCE

    def test_energy_performance_via_capacity_factor(self) -> None:
        r = self.svc.detect_intent("capacity factor for each facility")
        assert r.topic == ChatTopic.ENERGY_PERFORMANCE

    # --- SYSTEM_OVERVIEW ---
    def test_system_overview_via_tong_quan(self) -> None:
        r = self.svc.detect_intent("tong quan he thong hien tai")
        assert r.topic == ChatTopic.SYSTEM_OVERVIEW

    def test_system_overview_via_english(self) -> None:
        r = self.svc.detect_intent("give me the current system overview")
        assert r.topic == ChatTopic.SYSTEM_OVERVIEW

    def test_system_overview_via_production_output(self) -> None:
        r = self.svc.detect_intent("overall production output quality score")
        assert r.topic == ChatTopic.SYSTEM_OVERVIEW

    # --- FACILITY_INFO ---
    def test_facility_info_via_timezone(self) -> None:
        r = self.svc.detect_intent("timezone of each facility")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_facility_info_via_location(self) -> None:
        r = self.svc.detect_intent("location of the wrsf1 facility")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_facility_info_via_installed_capacity(self) -> None:
        r = self.svc.detect_intent("installed capacity of all facilities")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_facility_info_via_facility_id(self) -> None:
        r = self.svc.detect_intent("WRSF1 facility installed capacity location")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_facility_info_via_list_facilities(self) -> None:
        r = self.svc.detect_intent("list all facilities in the system")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_facility_info_via_darlington_point(self) -> None:
        r = self.svc.detect_intent("where is darlington point located")
        assert r.topic == ChatTopic.FACILITY_INFO

    # --- GENERAL fallback ---
    def test_general_fallback_no_keywords(self) -> None:
        r = self.svc.detect_intent("tell me a funny story about robots")
        assert r.topic == ChatTopic.GENERAL
        assert r.confidence <= 0.5

    def test_general_returns_low_confidence(self) -> None:
        r = self.svc.detect_intent("what is your name")
        assert r.confidence < 0.5


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    @pytest.fixture(autouse=True)
    def _service(self) -> None:
        self.svc = VietnameseIntentService(semantic_enabled=False)

    def test_single_keyword_hit_has_base_confidence(self) -> None:
        r = self.svc.detect_intent("pipeline")
        assert r.confidence >= 0.5

    def test_multiple_keyword_hits_increase_confidence(self) -> None:
        r_single = self.svc.detect_intent("pipeline")
        r_multi = self.svc.detect_intent("pipeline trang thai eta canh bao")
        assert r_multi.confidence >= r_single.confidence

    def test_confidence_capped_at_0_99(self) -> None:
        # Flood with many keywords from the same topic
        r = self.svc.detect_intent(
            "model r2 r-squared nrmse skill score champion model fallback version "
            "gbt model info ml model model version current model model quality"
        )
        assert r.confidence <= 0.99

    def test_keyword_confidence_formula(self) -> None:
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService as VS
        assert VS._keyword_confidence(0) == 0.5
        assert VS._keyword_confidence(1) == 0.65
        assert VS._keyword_confidence(2) == round(min(0.99, 0.5 + 2 * 0.15), 2)
        assert VS._keyword_confidence(100) == 0.99


# ---------------------------------------------------------------------------
# Empty/invalid input
# ---------------------------------------------------------------------------

class TestInvalidInput:
    @pytest.fixture(autouse=True)
    def _service(self) -> None:
        self.svc = VietnameseIntentService(semantic_enabled=False)

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.svc.detect_intent("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.svc.detect_intent("   ")


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestIntentCache:
    def test_cache_hit_returns_same_result(self) -> None:
        svc = VietnameseIntentService(semantic_enabled=False)
        r1 = svc.detect_intent("pipeline trang thai")
        r2 = svc.detect_intent("pipeline trang thai")
        assert r1 == r2
        # Should be in cache (same normalised key)
        assert len(svc._intent_cache) >= 1

    def test_cache_evicts_oldest_when_full(self) -> None:
        svc = VietnameseIntentService(semantic_enabled=False)
        # Fill the cache with distinct keys
        for i in range(_INTENT_CACHE_LIMIT):
            svc.detect_intent(f"pipeline query number {i}")
        # Cache should not exceed the limit
        assert len(svc._intent_cache) <= _INTENT_CACHE_LIMIT
        # One more entry should still succeed (oldest is evicted)
        svc.detect_intent("new pipeline query that was not cached")
        assert len(svc._intent_cache) <= _INTENT_CACHE_LIMIT

    def test_cache_updates_existing_key(self) -> None:
        svc = VietnameseIntentService(semantic_enabled=False)
        msg = "pipeline alerts"
        r1 = svc.detect_intent(msg)
        # Manually populate to simulate the re-cache path
        norm = svc._intent_cache
        assert normalize_vietnamese_text(msg) in norm
        r2 = svc.detect_intent(msg)
        assert r1.topic == r2.topic


# ---------------------------------------------------------------------------
# Semantic routing (initialize_semantic_router)
# ---------------------------------------------------------------------------

class TestSemanticRouter:
    def test_initialize_loads_embeddings(self) -> None:
        mock_client = MagicMock()
        # Return a list of 1-D vectors (length = number of canonical phrases)
        total_phrases = sum(
            len(phrases)
            for phrases in VietnameseIntentService._TOPIC_CANONICAL_PHRASES.values()
        )
        mock_client.embed_batch.return_value = [[0.1, 0.2]] * total_phrases

        svc = VietnameseIntentService(embedding_client=mock_client, semantic_enabled=True)
        svc.initialize_semantic_router()

        assert len(svc._topic_embeddings) > 0
        mock_client.embed_batch.assert_called_once()

    def test_initialize_skipped_when_semantic_disabled(self) -> None:
        mock_client = MagicMock()
        svc = VietnameseIntentService(embedding_client=mock_client, semantic_enabled=False)
        svc.initialize_semantic_router()
        mock_client.embed_batch.assert_not_called()

    def test_initialize_skipped_when_no_client(self) -> None:
        svc = VietnameseIntentService(embedding_client=None, semantic_enabled=True)
        svc.initialize_semantic_router()
        assert svc._topic_embeddings == {}

    def test_initialize_disables_semantic_on_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.embed_batch.side_effect = RuntimeError("embedding API down")
        svc = VietnameseIntentService(embedding_client=mock_client, semantic_enabled=True)
        svc.initialize_semantic_router()
        assert svc._embedding_client is None
        assert svc._topic_embeddings == {}


# ---------------------------------------------------------------------------
# _cosine_similarity edge cases
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    # _cosine_similarity is an instance method — use a no-arg instance.
    _svc = VietnameseIntentService(embedding_client=None, semantic_enabled=False)

    def test_identical_vectors_return_1(self) -> None:
        v = [1.0, 0.0, 0.0]
        sim = self._svc._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-9

    def test_orthogonal_vectors_return_0(self) -> None:
        sim = self._svc._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-9

    def test_zero_vector_a_returns_0(self) -> None:
        sim = self._svc._cosine_similarity([0.0, 0.0], [1.0, 0.5])
        assert sim == 0.0

    def test_zero_vector_b_returns_0(self) -> None:
        sim = self._svc._cosine_similarity([1.0, 0.5], [0.0, 0.0])
        assert sim == 0.0

    def test_both_zero_returns_0(self) -> None:
        sim = self._svc._cosine_similarity([0.0], [0.0])
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Semantic detection path in detect_intent
# ---------------------------------------------------------------------------

class TestDetectIntentSemanticPath:
    def _make_svc_with_semantic(
        self,
        embed_text_vec: list[float],
        topic_embeddings: dict,
        semantic_min_confidence: float = 0.65,
    ) -> VietnameseIntentService:
        mock_client = MagicMock()
        mock_client.embed_text.return_value = embed_text_vec
        svc = VietnameseIntentService(
            embedding_client=mock_client,
            semantic_enabled=True,
            semantic_min_confidence=semantic_min_confidence,
        )
        svc._topic_embeddings = topic_embeddings
        return svc

    def test_semantic_result_used_when_no_keyword_match(self) -> None:
        # The query "unusual phrase xyz" won't hit any keyword
        # but we inject a perfect semantic vector for FORECAST_72H.
        vec = [1.0, 0.0]
        svc = self._make_svc_with_semantic(
            embed_text_vec=vec,
            topic_embeddings={ChatTopic.FORECAST_72H: [[1.0, 0.0]]},
            semantic_min_confidence=0.5,
        )
        r = svc.detect_intent("unusual phrase xyz")
        assert r.topic == ChatTopic.FORECAST_72H

    def test_semantic_below_min_confidence_ignored(self) -> None:
        # Cosine similarity will be low — model vec is orthogonal
        svc = self._make_svc_with_semantic(
            embed_text_vec=[1.0, 0.0],
            topic_embeddings={ChatTopic.FORECAST_72H: [[0.0, 1.0]]},
            semantic_min_confidence=0.9,
        )
        # No keyword match either → should return GENERAL
        r = svc.detect_intent("unusual phrase xyz")
        assert r.topic == ChatTopic.GENERAL

    def test_keyword_wins_over_semantic_when_confidence_high(self) -> None:
        # Force a strong keyword match (score >= 3 → confidence 0.95)
        # and a semantic result for a different topic.
        vec = [1.0, 0.0]
        svc = self._make_svc_with_semantic(
            embed_text_vec=vec,
            topic_embeddings={ChatTopic.SYSTEM_OVERVIEW: [[1.0, 0.0]]},
            semantic_min_confidence=0.5,
        )
        # Many pipeline keywords → high keyword confidence
        r = svc.detect_intent("pipeline eta trang thai canh bao pipeline alert")
        # Keyword confidence >= 0.8, so keyword should win
        assert r.topic == ChatTopic.PIPELINE_STATUS

    def test_semantic_overrides_weak_keyword_different_topic(self) -> None:
        # 1 keyword hit → confidence 0.65 (< 0.8), semantic has higher confidence
        # and a different topic → semantic should win
        vec = [1.0, 0.0]
        svc = self._make_svc_with_semantic(
            embed_text_vec=vec,
            topic_embeddings={ChatTopic.ML_MODEL: [[1.0, 0.0]]},
            semantic_min_confidence=0.5,
            # semantic_keyword_score_threshold=1 default
        )
        # "pipeline" alone → score=1 → confidence=0.65, topic=PIPELINE_STATUS
        # semantic sees ML_MODEL with cosine=1.0 > 0.65+0.08
        r = svc.detect_intent("pipeline")
        # Either ML_MODEL (semantic override) or PIPELINE_STATUS (keyword wins)
        # The code overrides when semantic.confidence >= keyword.confidence + 0.08 and keyword < 0.8
        # cosine=1.0, keyword_conf=0.65 → 1.0 >= 0.73 and 0.65 < 0.8 → semantic wins
        assert r.topic == ChatTopic.ML_MODEL

    def test_semantic_client_failure_falls_back_to_keyword(self) -> None:
        mock_client = MagicMock()
        mock_client.embed_text.side_effect = RuntimeError("embedding down")
        svc = VietnameseIntentService(
            embedding_client=mock_client,
            semantic_enabled=True,
        )
        svc._topic_embeddings = {ChatTopic.FORECAST_72H: [[1.0, 0.0]]}
        # The semantic path should catch the exception and fall back to keyword
        r = svc.detect_intent("72h khoang tin cay 3 ngay")
        assert r.topic == ChatTopic.FORECAST_72H
        # After failure, embedding client should be disabled
        assert svc._embedding_client is None

    def test_no_api_key_raises_value_error(self) -> None:
        """detect_intent on empty normalised string raises ValueError."""
        svc = VietnameseIntentService(semantic_enabled=False)
        with pytest.raises(ValueError):
            svc.detect_intent("   ")


# ---------------------------------------------------------------------------
# Keyword-match bias branches (_keyword_match)
# ---------------------------------------------------------------------------

class TestKeywordMatchBias:
    @pytest.fixture(autouse=True)
    def _service(self) -> None:
        self.svc = VietnameseIntentService(semantic_enabled=False)

    def test_facility_priority_markers_bias_facility_info(self) -> None:
        # "installed capacity" is a facility_priority_marker → facility bias +2
        r = self.svc.detect_intent("installed capacity of each station")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_how_many_stations_biases_facility_info(self) -> None:
        r = self.svc.detect_intent("how many stations are in the system")
        assert r.topic == ChatTopic.FACILITY_INFO

    def test_ml_priority_markers_bias_ml_model(self) -> None:
        # "fallback" and "r-squared" are ml_priority_markers → ml bias +2
        r = self.svc.detect_intent("fallback model with r-squared score")
        assert r.topic == ChatTopic.ML_MODEL

    def test_skill_score_ml_priority_marker(self) -> None:
        r = self.svc.detect_intent("what is the skill score and nrmse")
        assert r.topic == ChatTopic.ML_MODEL

    def test_energy_comparison_bias_fires_for_top_facility(self) -> None:
        r = self.svc.detect_intent("top 5 facilities by energy output comparison")
        assert r.topic == ChatTopic.ENERGY_PERFORMANCE

    def test_energy_comparison_does_not_fire_for_aqi_query(self) -> None:
        # AQI prevents the energy comparison bias
        r = self.svc.detect_intent("chi so aqi aqi thap nhat cac tram")
        assert r.topic == ChatTopic.DATA_QUALITY_ISSUES

    def test_is_energy_comparison_query_with_compare_and_facility(self) -> None:
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService as VS
        assert VS._is_energy_comparison_query("compare top 2 facilities energy")

    def test_is_energy_comparison_query_false_for_capacity_markers(self) -> None:
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService as VS
        assert not VS._is_energy_comparison_query("installed capacity of facilities")

    def test_is_energy_comparison_query_false_for_aqi(self) -> None:
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService as VS
        assert not VS._is_energy_comparison_query("aqi facility ranking top 3")

    def test_liet_ke_biases_facility(self) -> None:
        r = self.svc.detect_intent("liet ke tat ca co so hien tai")
        assert r.topic == ChatTopic.FACILITY_INFO


# ---------------------------------------------------------------------------
# Matched topic with score below semantic threshold
# ---------------------------------------------------------------------------

class TestMatchedTopicBelowThreshold:
    def test_topic_returned_when_below_threshold_no_semantic(self) -> None:
        # semantic_keyword_score_threshold=2 means score=1 doesn't hit fast path
        # but should still return a result via the tail of detect_intent
        svc = VietnameseIntentService(
            semantic_enabled=False,
            semantic_keyword_score_threshold=2,
        )
        # "pipeline" → score=1 < threshold=2 → goes through tail path
        r = svc.detect_intent("pipeline")
        assert r.topic == ChatTopic.PIPELINE_STATUS
        assert r.confidence >= 0.5
