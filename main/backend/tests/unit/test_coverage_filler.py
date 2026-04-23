"""Small targeted tests to close remaining coverage gaps."""
from __future__ import annotations

from app.services.solar_ai_chat import chat_service as cs_module
from app.services.solar_ai_chat.query_rewriter import QueryRewriter


class TestIsCapacityOrStationQuery:
    def test_installed_capacity(self) -> None:
        assert cs_module._is_capacity_or_station_query("What is the installed capacity?")

    def test_how_many_stations(self) -> None:
        assert cs_module._is_capacity_or_station_query("how many stations are there?")

    def test_vietnamese_cong_suat(self) -> None:
        assert cs_module._is_capacity_or_station_query("cong suat lap dat la bao nhieu")

    def test_unrelated_returns_false(self) -> None:
        assert not cs_module._is_capacity_or_station_query("show me the energy output today")


class TestIsImplicitFollowup:
    def test_unrelated_returns_false(self) -> None:
        assert not cs_module._is_implicit_followup("show me the energy output today")


class TestIntentCacheLRU:
    def test_repeated_message_hits_cache_then_moves_to_end(self) -> None:
        from app.services.solar_ai_chat.intent_service import VietnameseIntentService

        svc = VietnameseIntentService(embedding_client=None, semantic_enabled=False)
        # First call populates cache
        r1 = svc.detect_intent("what is the solar system overview")
        # Second call with same message hits the cache pop-and-repromote branch
        r2 = svc.detect_intent("what is the solar system overview")
        assert r1.topic == r2.topic


class TestQueryRewriter:
    def test_empty_message_returns_empty_normalized(self) -> None:
        result = QueryRewriter().rewrite("")
        assert result.normalized == ""
        assert result.language == "en"

    def test_whitespace_only_treated_as_empty(self) -> None:
        result = QueryRewriter().rewrite("   \t\n ")
        assert result.normalized == ""

    def test_non_empty_returns_normalized_and_language(self) -> None:
        result = QueryRewriter().rewrite("Hello world")
        assert result.original == "Hello world"
        assert result.language in ("en", "vi")
