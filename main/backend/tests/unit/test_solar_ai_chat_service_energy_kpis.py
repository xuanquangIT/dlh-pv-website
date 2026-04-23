"""Tests for SolarAIChatService — energy KPI queries.

With the new agentic architecture, model_router=None means the service
returns build_insufficient_data_response() because it cannot reason without
an LLM.  The actual data-backed answers are produced by the LLM after calling
tools — tested via mock model_router in test_agentic_tool_loop.py.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService


def _make_service(model_router=None):
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = (
        {
            "top_facilities": [
                {"facility": "Darlington Point", "energy_mwh": 48235.56, "capacity_mw": 324.0},
                {"facility": "Avonlie", "energy_mwh": 34845.68, "capacity_mw": 254.1},
            ],
            "bottom_facilities": [
                {"facility": "Emerald", "energy_mwh": 14593.87, "capacity_mw": 88.0},
            ],
            "facility_count": 8,
            "peak_hours": [{"hour": 23, "energy_mwh": 15367.19}],
            "tomorrow_forecast_mwh": 4010.13,
            "window_days": 30,
        },
        [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}],
    )
    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        confidence=0.9,
    )
    return SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=model_router,
        history_repository=None,
        web_search_client=None,
    )


# ---------------------------------------------------------------------------
# No-LLM path: returns structured unavailability notice
# ---------------------------------------------------------------------------


def test_no_llm_returns_insufficient_data_response_for_kpi_question() -> None:
    """With model_router=None the service must return a clear unavailability
    notice rather than a stale hardcoded template."""
    service = _make_service(model_router=None)
    response = service.handle_query(
        SolarChatRequest(
            message="Các chỉ số KPI nào để đánh giá nhà máy điện mặt trời?",
            role=ChatRole.DATA_ANALYST,
            session_id=None,
        )
    )
    assert response.fallback_used is True
    assert response.model_used == "none"
    # Must not fabricate metric values
    assert "48235" not in response.answer
    assert "34845" not in response.answer


def test_no_llm_returns_insufficient_data_response_for_ranking_question() -> None:
    service = _make_service(model_router=None)
    response = service.handle_query(
        SolarChatRequest(
            message="Top facilities and peak hours today",
            role=ChatRole.DATA_ANALYST,
            session_id=None,
        )
    )
    assert response.fallback_used is True
    assert response.model_used == "none"


def test_no_llm_returns_vietnamese_response_for_vietnamese_message() -> None:
    service = _make_service(model_router=None)
    response = service.handle_query(
        SolarChatRequest(
            message="Trạm nào có sản lượng cao nhất?",
            role=ChatRole.DATA_ANALYST,
            session_id=None,
        )
    )
    assert response.fallback_used is True
    # Vietnamese message -> Vietnamese insufficient-data notice
    assert "kha dung" in response.answer.lower() or "khong" in response.answer.lower() or "unavailable" in response.answer.lower()


# ---------------------------------------------------------------------------
# Agentic loop with mock LLM
# ---------------------------------------------------------------------------


def _build_mock_model_router(answer_text: str):
    """Return a mock model_router that produces a final text answer.

    With the intent pre-fetch synthesis bypass, the service now calls
    ``generate()`` instead of ``generate_with_tools()`` when pre-fetched data
    is available.  Both methods are mocked so the test works regardless of
    which path is taken.
    """
    from app.services.solar_ai_chat.llm_client import LLMGenerationResult, LLMToolResult

    mock = MagicMock()
    # Used by the synthesis bypass (pre-fetch path)
    mock.generate.return_value = LLMGenerationResult(
        text=answer_text,
        model_used="gemini-mock",
        fallback_used=False,
    )
    # Used by the agentic loop (fallback path)
    mock.generate_with_tools.return_value = LLMToolResult(
        function_call=None,
        text=answer_text,
        model_used="gemini-mock",
        fallback_used=False,
    )
    return mock


def test_agentic_loop_returns_llm_answer() -> None:
    expected_answer = "Darlington Point là trạm dẫn đầu với 48,235 MWh."
    model_router = _build_mock_model_router(expected_answer)
    service = _make_service(model_router=model_router)

    response = service.handle_query(
        SolarChatRequest(
            message="Trạm nào có sản lượng cao nhất?",
            role=ChatRole.DATA_ANALYST,
            session_id=None,
        )
    )

    assert response.answer == expected_answer
    assert response.model_used == "gemini-mock"
    assert response.fallback_used is False
    # Pre-fetch succeeded → synthesis bypass calls generate(), not generate_with_tools()
    model_router.generate.assert_called_once()


def test_agentic_loop_with_tool_call_then_answer() -> None:
    """Simulate: LLM calls get_energy_performance, then returns text."""
    from app.services.solar_ai_chat.llm_client import LLMToolResult, ToolCallRequest

    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = (
        {"top_facilities": [{"facility": "Darlington Point", "energy_mwh": 48235.56}], "facility_count": 8},
        [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}],
    )
    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.GENERAL, confidence=0.0
    )

    model_router = MagicMock()
    model_router.generate_with_tools.side_effect = [
        # First call: request a tool
        LLMToolResult(
            function_call=ToolCallRequest(name="get_energy_performance", arguments={}),
            text=None,
            model_used="gemini-mock",
            fallback_used=False,
        ),
        # Second call: final answer
        LLMToolResult(
            function_call=None,
            text="Darlington Point dẫn đầu với 48,235 MWh.",
            model_used="gemini-mock",
            fallback_used=False,
        ),
    ]

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=model_router,
        history_repository=None,
        web_search_client=None,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Top facilities hôm nay?",
            role=ChatRole.DATA_ANALYST,
            session_id=None,
        )
    )

    assert "Darlington Point" in response.answer
    assert response.model_used == "gemini-mock"
    assert response.fallback_used is False
    assert model_router.generate_with_tools.call_count == 2
    # Tool was executed
    repository.fetch_topic_metrics.assert_called_once()
