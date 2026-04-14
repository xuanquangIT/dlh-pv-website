"""Tests for SolarAIChatService — web search via agentic web_lookup tool call.

In the new architecture web search is triggered when the LLM requests the
web_lookup tool, not by heuristic text matching.  These tests mock the LLM to
produce a web_lookup function call and verify the service executes the search,
returns results to the LLM, and finalises the response correctly.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.llm_client import LLMToolResult, ToolCallRequest
from app.services.solar_ai_chat.web_search_client import WebSearchResult


class StubWebSearchClient:
    def __init__(self, results: list[WebSearchResult]) -> None:
        self.enabled = True
        self._results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        self.queries.append(query)
        return list(self._results)


def _facility_metrics():
    return (
        {
            "facility_count": 3,
            "facilities": [
                {"facility_name": "Darlington Point", "total_capacity_mw": 324.0, "timezone_name": "Australia/Eastern"},
                {"facility_name": "Avonlie", "total_capacity_mw": 254.1, "timezone_name": "Australia/Eastern"},
                {"facility_name": "Emerald", "total_capacity_mw": 88.0, "timezone_name": "Australia/Eastern"},
            ],
        },
        [{"layer": "Gold", "dataset": "gold.dim_facility", "data_source": "databricks"}],
    )


def _build_service_with_web_search(web_results, llm_side_effect):
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics()
    intent_service = MagicMock()
    # Confidence below pre-fetch threshold (0.6) so the agentic loop runs
    # and the LLM can request web_lookup as expected by these tests.
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO, confidence=0.5
    )
    web_client = StubWebSearchClient(web_results)

    model_router = MagicMock()
    model_router.generate_with_tools.side_effect = llm_side_effect

    return SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=model_router,
        history_repository=None,
        web_search_client=web_client,
    ), web_client


# ---------------------------------------------------------------------------
# LLM-triggered web search
# ---------------------------------------------------------------------------


def test_llm_web_lookup_results_are_passed_back_to_llm() -> None:
    """When LLM autonomously requests web_lookup (not user-triggered), the
    service executes it and returns results so the LLM can synthesise an answer.

    Note: user-triggered web searches ("search internet ...") now use the
    direct web-search path (test_solar_ai_chat_web_search_direct.py).
    This test covers the agentic path where the LLM itself decides to search.
    """
    web_results = [
        WebSearchResult(
            title="Darlington Point Solar Farm - Overview",
            url="https://example.com/darlington-point",
            snippet="Darlington Point Solar Farm is one of the largest PV plants in Australia.",
            score=0.98,
        ),
    ]

    llm_side_effect = [
        # Step 1: LLM autonomously requests web_lookup
        LLMToolResult(
            function_call=ToolCallRequest(
                name="web_lookup",
                arguments={"query": "Darlington Point solar farm capacity NSW"},
            ),
            text=None,
            model_used="gemini-mock",
            fallback_used=False,
        ),
        # Step 2: LLM synthesises answer using web results
        LLMToolResult(
            function_call=None,
            text="Darlington Point Solar Farm là trạm lớn nhất với công suất 324 MW.",
            model_used="gemini-mock",
            fallback_used=False,
        ),
    ]

    service, web_client = _build_service_with_web_search(web_results, llm_side_effect)

    response = service.handle_query(
        SolarChatRequest(
            # No "internet"/"search" keywords — goes through agentic loop, not direct path
            message="Cho tôi biết thêm thông tin về trạm Darlington Point",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert "Darlington Point" in response.answer
    assert response.model_used == "gemini-mock"
    assert response.fallback_used is False
    # Web search was executed
    assert len(web_client.queries) == 1
    assert "Darlington Point" in web_client.queries[0]


def test_web_search_disabled_returns_error_to_llm_but_still_answers() -> None:
    """When web search client is disabled, the service returns an error response
    to the LLM tool call and the LLM should still produce a final answer."""
    service, web_client = _build_service_with_web_search([], [
        LLMToolResult(
            function_call=ToolCallRequest(
                name="web_lookup",
                arguments={"query": "solar farm info"},
            ),
            text=None,
            model_used="gemini-mock",
            fallback_used=False,
        ),
        LLMToolResult(
            function_call=None,
            text="Xin loi, khong co ket qua web.",
            model_used="gemini-mock",
            fallback_used=False,
        ),
    ])
    # Disable web search
    web_client.enabled = False

    response = service.handle_query(
        SolarChatRequest(
            message="Tim thong tin tren internet",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )
    # Should still produce an answer (LLM handles gracefully)
    assert response.answer  # non-empty
    # Web search was NOT executed (disabled)
    assert len(web_client.queries) == 0


def test_no_llm_returns_insufficient_data_response() -> None:
    """With model_router=None the service cannot do agentic reasoning and
    returns the structured unavailability notice."""
    web_client = StubWebSearchClient([])
    service = SolarAIChatService(
        repository=MagicMock(),
        intent_service=MagicMock(),
        model_router=None,
        history_repository=None,
        web_search_client=web_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Tìm thông tin trên internet",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.fallback_used is True
    assert response.model_used == "none"
    assert len(web_client.queries) == 0  # web search never triggered
