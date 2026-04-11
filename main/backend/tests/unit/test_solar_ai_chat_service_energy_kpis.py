from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService


def _build_metrics_payload() -> tuple[dict, list[dict[str, str]]]:
    return (
        {
            "top_facilities": [
                {"facility": "Darlington Point", "energy_mwh": 48235.56},
                {"facility": "Avonlie", "energy_mwh": 34845.68},
                {"facility": "Emerald", "energy_mwh": 14593.87},
            ],
            "peak_hours": [
                {"hour": 23, "energy_mwh": 15367.19},
                {"hour": 0, "energy_mwh": 15346.26},
                {"hour": 1, "energy_mwh": 15145.88},
            ],
            "tomorrow_forecast_mwh": 4010.13,
        },
        [{"layer": "Gold", "dataset": "gold.fact_energy", "data_source": "databricks"}],
    )


def test_service_returns_metric_explanation_for_energy_kpi_question() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _build_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        confidence=0.9,
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=None,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Các chỉ số thường được dùng để đánh giá một nhà máy điện năng lượng mặt trời có hoạt động tốt không?",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.ENERGY_PERFORMANCE
    assert "Performance Ratio (PR)" in response.answer
    assert "Capacity Factor (CF)" in response.answer
    assert "Top facilities:" not in response.answer


def test_service_keeps_ranking_answer_for_ranking_energy_question() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _build_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        confidence=0.9,
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=None,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Top facilities and peak hours today",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.ENERGY_PERFORMANCE
    assert "Top facilities:" in response.answer
    assert "Peak hours:" in response.answer
