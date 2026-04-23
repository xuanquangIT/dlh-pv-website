"""Tests for the direct web-search path in SolarAIChatService.

When a user explicitly requests an internet search ('search internet ...'),
the service must:
1. Call web_search_client directly (NOT via generate_with_tools).
2. Combine web snippets + any pre-fetched system data as evidence.
3. Synthesise via generate() with history context in the prompt.

This bypasses gpt-5-mini's failure to call web_lookup inside generate_with_tools.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import (
    SolarAIChatService,
    _extract_search_query,
    _needs_web_search,
)
from app.services.solar_ai_chat.llm_client import LLMGenerationResult
from app.services.solar_ai_chat.web_search_client import WebSearchResult


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

class StubWebSearchClient:
    def __init__(self, results: list[WebSearchResult], enabled: bool = True) -> None:
        self.enabled = enabled
        self._results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        self.queries.append(query)
        return list(self._results)


def _make_web_result(title: str = "PR Guide", snippet: str = "PR = E_AC / (P_r * H)") -> WebSearchResult:
    return WebSearchResult(title=title, url="https://example.com/pr", snippet=snippet, score=0.9)


def _build_service(
    web_results: list[WebSearchResult],
    answer_text: str = "PR = sản lượng thực / sản lượng lý thuyết × 100%",
    intent_topic: ChatTopic = ChatTopic.ENERGY_PERFORMANCE,
    intent_confidence: float = 0.9,
    repository_data: dict | None = None,
    web_enabled: bool = True,
) -> tuple[SolarAIChatService, StubWebSearchClient, MagicMock]:
    repository = MagicMock()
    if repository_data is not None:
        repository.fetch_topic_metrics.return_value = (
            repository_data,
            [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}],
        )
    else:
        repository.fetch_topic_metrics.return_value = (
            {
                "top_facilities": [
                    {"facility": "Darlington Point", "energy_mwh": 44956.19, "capacity_mw": 324.0},
                    {"facility": "Avonlie", "energy_mwh": 33536.17, "capacity_mw": 254.1},
                ],
                "facility_count": 8,
                "top_performance_ratio_facilities": [
                    {"facility": "Emerald", "performance_ratio_pct": 23.57},
                ],
                "window_days": 30,
            },
            [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}],
        )

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=intent_topic, confidence=intent_confidence
    )

    web_client = StubWebSearchClient(web_results, enabled=web_enabled)

    model_router = MagicMock()
    model_router.generate.return_value = LLMGenerationResult(
        text=answer_text,
        model_used="gpt-5-mini",
        fallback_used=False,
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=model_router,
        history_repository=None,
        web_search_client=web_client,
    )
    return service, web_client, model_router


# ---------------------------------------------------------------------------
# _needs_web_search unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "Search internet xem cách tính PR",
    "search internet PR formula",
    "Tìm kiếm trên web về PR",
    "tim kiem tren web PR",
    "Google search performance ratio",
    "tra cứu công thức performance ratio",
    "Tìm thông tin trực tuyến (online)",
    "tìm trên internet thông tin về 2 trạm này",
])
def test_needs_web_search_returns_true(message: str) -> None:
    assert _needs_web_search(message), f"Expected True for: {message!r}"


@pytest.mark.parametrize("message", [
    "Cách tính performance ratio",
    "2 trạm lớn nhất hệ thống",
    "Tổng quan hệ thống hiện tại",
    "Dữ liệu năng lượng 30 ngày",
])
def test_needs_web_search_returns_false_for_normal_queries(message: str) -> None:
    assert not _needs_web_search(message), f"Expected False for: {message!r}"


# ---------------------------------------------------------------------------
# _extract_search_query unit tests
# ---------------------------------------------------------------------------

def test_extract_search_query_strips_prefix() -> None:
    q = _extract_search_query("search internet cách tính performance ratio", None)
    assert "search internet" not in q.lower()
    assert "performance ratio" in q.lower()


def test_extract_search_query_strips_vi_prefix() -> None:
    q = _extract_search_query("tìm kiếm công thức PR trên mạng", None)
    # Should not still start with "tìm kiếm"
    assert not q.lower().startswith("tìm kiếm")


def test_extract_search_query_enriches_short_query_from_history() -> None:
    """A vague follow-up like 'xem cách tính chính xác' should be enriched
    with the last assistant turn's content."""
    from app.schemas.solar_ai_chat.session import ChatMessage
    from datetime import datetime

    history = [
        ChatMessage(
            id="1", session_id="s1", sender="user",
            content="Cách tính performance ratio",
            timestamp=datetime(2026, 4, 13, 14, 44),
        ),
        ChatMessage(
            id="2", session_id="s1", sender="assistant",
            content="Performance Ratio (PR) là tỷ lệ giữa năng lượng thực tế và lý thuyết.",
            timestamp=datetime(2026, 4, 13, 14, 44, 30),
        ),
    ]
    q = _extract_search_query("Search internet xem cách tính chính xác", history)
    # Should contain context from assistant's last message
    assert "performance ratio" in q.lower() or "PR" in q


def test_extract_search_query_no_history_returns_stripped_message() -> None:
    q = _extract_search_query("search performance ratio formula", None)
    assert q  # non-empty
    assert "search" not in q.lower()


# ---------------------------------------------------------------------------
# Direct web-search integration path
# ---------------------------------------------------------------------------

def test_direct_web_search_calls_client_not_generate_with_tools() -> None:
    """When user says 'search internet ...', web_search_client.search() is
    called directly — generate_with_tools must NOT be called."""
    service, web_client, model_router = _build_service(
        web_results=[_make_web_result()],
    )

    response = service.handle_query(SolarChatRequest(
        message="Search internet cách tính performance ratio",
        role=ChatRole.DATA_ENGINEER,
        session_id=None,
    ))

    assert response.answer  # non-empty
    model_router.generate_with_tools.assert_not_called()
    model_router.generate.assert_called_once()
    assert len(web_client.queries) == 1


def test_direct_web_search_results_in_synthesis_prompt() -> None:
    """Web snippets should reach generate() via the synthesis prompt."""
    web_result = _make_web_result(
        title="How to calculate Performance Ratio",
        snippet="PR = E_AC divided by P_r times H_POA",
    )
    service, web_client, model_router = _build_service(web_results=[web_result])

    service.handle_query(SolarChatRequest(
        message="Search internet cách tính performance ratio",
        role=ChatRole.DATA_ENGINEER,
        session_id=None,
    ))

    # The synthesis prompt passed to generate() must contain the web snippet
    call_args = model_router.generate.call_args
    prompt_text = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
    assert "PR = E_AC" in prompt_text or "How to calculate" in prompt_text


def test_direct_web_search_includes_prefetched_data_in_evidence() -> None:
    """Pre-fetched system data (energy KPIs) should also appear in the evidence
    passed to generate() when web search is requested."""
    service, web_client, model_router = _build_service(
        web_results=[_make_web_result()],
    )

    service.handle_query(SolarChatRequest(
        message="Search internet cách tính performance ratio",
        role=ChatRole.DATA_ANALYST,
        session_id=None,
    ))

    call_args = model_router.generate.call_args
    prompt_text = call_args[0][0] if call_args[0] else ""
    # Darlington Point is in pre-fetched top_facilities
    assert "Darlington Point" in prompt_text


def test_direct_web_search_with_history_context_in_prompt() -> None:
    """Conversation history should be included in the prompt when web search fires."""
    from app.schemas.solar_ai_chat.session import ChatMessage
    from datetime import datetime

    history = [
        ChatMessage(
            id="1", session_id="s1", sender="user",
            content="2 trạm lớn nhất là gì?",
            timestamp=datetime(2026, 4, 13, 14, 30),
        ),
        ChatMessage(
            id="2", session_id="s1", sender="assistant",
            content="Darlington Point (44956 MWh) và Avonlie (33536 MWh).",
            timestamp=datetime(2026, 4, 13, 14, 30, 30),
        ),
    ]

    service, web_client, model_router = _build_service(web_results=[_make_web_result()])

    # Patch history loading to return our fake history
    with patch.object(service, "_load_history", return_value=history):
        service.handle_query(SolarChatRequest(
            message="Tìm thông tin 2 trạm này trên internet và so sánh thêm",
            role=ChatRole.DATA_ENGINEER,
            session_id="s1",
        ))

    call_args = model_router.generate.call_args
    prompt_text = call_args[0][0] if call_args[0] else ""
    # History should be in the prompt so LLM knows "2 trạm này" refers to D.P. + Avonlie
    assert "Darlington Point" in prompt_text
    assert "Avonlie" in prompt_text


def test_direct_web_search_when_client_disabled_answers_from_knowledge() -> None:
    """When web search is disabled, the service must still answer (from history
    and/or knowledge) — not refuse with 'I cannot search the web'."""
    service, web_client, model_router = _build_service(
        web_results=[],
        web_enabled=False,
        answer_text="Performance Ratio là tỷ lệ giữa năng lượng thực tế và lý thuyết.",
    )

    response = service.handle_query(SolarChatRequest(
        message="Search internet cách tính performance ratio",
        role=ChatRole.DATA_ENGINEER,
        session_id=None,
    ))

    assert "Performance Ratio" in response.answer
    # Web client never queried (it's disabled so _execute_web_lookup returns error dict)
    assert len(web_client.queries) == 0
    # But generate() is still called with an evidence prompt
    model_router.generate.assert_called_once()


def test_direct_web_search_vague_followup_query_is_enriched() -> None:
    """'search internet xem cách tính chính xác' after a PR discussion should
    produce a search query mentioning Performance Ratio, not a generic query."""
    from app.schemas.solar_ai_chat.session import ChatMessage
    from datetime import datetime

    history = [
        ChatMessage(
            id="1", session_id="s1", sender="user",
            content="Cách tính performance ratio",
            timestamp=datetime(2026, 4, 13, 14, 44),
        ),
        ChatMessage(
            id="2", session_id="s1", sender="assistant",
            content="Performance Ratio (PR) là tỷ lệ giữa năng lượng thực tế và lý thuyết.",
            timestamp=datetime(2026, 4, 13, 14, 44, 30),
        ),
    ]

    service, web_client, model_router = _build_service(web_results=[_make_web_result()])

    with patch.object(service, "_load_history", return_value=history):
        service.handle_query(SolarChatRequest(
            message="Search internet xem cách tính chính xác",
            role=ChatRole.DATA_ENGINEER,
            session_id="s1",
        ))

    # The search query must be enriched with PR context, not just "cách tính chính xác"
    assert len(web_client.queries) == 1
    query = web_client.queries[0].lower()
    assert "performance ratio" in query or "pr" in query


def test_followup_context_reaches_synthesis_prompt() -> None:
    """After 'largest 2 stations' answer, follow-up asking to search for
    comparison info should include those station names in the synthesis prompt."""
    from app.schemas.solar_ai_chat.session import ChatMessage
    from datetime import datetime

    history = [
        ChatMessage(
            id="1", session_id="s1", sender="user",
            content="2 trạm lớn nhất của hệ thống là gì",
            timestamp=datetime(2026, 4, 13, 14, 30),
        ),
        ChatMessage(
            id="2", session_id="s1", sender="assistant",
            content="Darlington Point (44,956 MWh) và Avonlie (33,536 MWh).",
            timestamp=datetime(2026, 4, 13, 14, 30, 30),
        ),
    ]

    service, web_client, model_router = _build_service(
        web_results=[_make_web_result(
            title="Darlington Point Solar Farm",
            snippet="324 MW solar farm in NSW Australia.",
        )],
    )

    with patch.object(service, "_load_history", return_value=history):
        service.handle_query(SolarChatRequest(
            message="Tìm thông tin 2 trạm này trên internet và so sánh thêm nhiều khía cạnh",
            role=ChatRole.DATA_ENGINEER,
            session_id="s1",
        ))

    call_args = model_router.generate.call_args
    prompt_text = call_args[0][0] if call_args[0] else ""
    # History should put "Darlington Point" and "Avonlie" in context
    assert "Darlington Point" in prompt_text
    assert "Avonlie" in prompt_text
