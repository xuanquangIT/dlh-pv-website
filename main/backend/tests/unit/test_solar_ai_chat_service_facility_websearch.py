from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.solar_ai_chat import ChatRole, ChatTopic, SolarChatRequest
from app.services.solar_ai_chat.chat_service import SolarAIChatService
from app.services.solar_ai_chat.web_search_client import WebSearchResult


class StubWebSearchClient:
    def __init__(self, results: list[WebSearchResult]) -> None:
        self.enabled = True
        self._results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[WebSearchResult]:
        self.queries.append(query)
        return list(self._results)


def _facility_metrics_payload() -> tuple[dict, list[dict[str, str]]]:
    return (
        {
            "facility_count": 3,
            "facilities": [
                {
                    "facility_name": "Avonlie",
                    "location_lat": -34.913826,
                    "location_lng": 146.590545,
                    "total_capacity_mw": 254.1,
                    "timezone_name": "Australia/Eastern",
                    "timezone_utc_offset": "UTC+10:00",
                },
                {
                    "facility_name": "Darlington Point",
                    "location_lat": -34.648971,
                    "location_lng": 146.061179,
                    "total_capacity_mw": 324.0,
                    "timezone_name": "Australia/Eastern",
                    "timezone_utc_offset": "UTC+10:00",
                },
                {
                    "facility_name": "Emerald",
                    "location_lat": -23.507996,
                    "location_lng": 148.126404,
                    "total_capacity_mw": 88.0,
                    "timezone_name": "Australia/Eastern",
                    "timezone_utc_offset": "UTC+10:00",
                },
            ],
        },
        [{"layer": "Gold", "dataset": "gold.dim_facility", "data_source": "databricks"}],
    )


def test_explicit_internet_request_for_largest_capacity_station_uses_web_search() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO,
        confidence=0.9,
    )

    web_search_client = StubWebSearchClient(
        [
            WebSearchResult(
                title="Darlington Point Solar Farm - Overview",
                url="https://example.com/darlington-point-overview",
                snippet="Darlington Point Solar Farm is one of the largest utility-scale PV plants in Australia.",
                score=0.98,
            ),
            WebSearchResult(
                title="NSW Planning - Darlington Point details",
                url="https://example.com/darlington-point-planning",
                snippet="Project profile covering location, commissioning, and installed capacity.",
                score=0.86,
            ),
        ]
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=web_search_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message=(
                "Trạm có capacity lớn nhất là trạm nào, thông tin chi tiết trạm đó "
                "tìm thông tin trạm trên internet"
            ),
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.FACILITY_INFO
    assert response.model_used == "web-search-fallback"
    assert response.fallback_used is True
    assert response.key_metrics.get("web_search_used") is True
    assert response.key_metrics.get("web_search_source_count") == 2
    assert "Darlington Point" in response.answer
    assert "Nguồn tham khảo" in response.answer
    assert web_search_client.queries
    assert "Darlington Point solar farm" in web_search_client.queries[0]


def test_explicit_internet_request_for_smallest_capacity_station_uses_web_search() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO,
        confidence=0.9,
    )

    web_search_client = StubWebSearchClient(
        [
            WebSearchResult(
                title="Emerald Solar Farm profile",
                url="https://example.com/emerald-solar-farm",
                snippet="Emerald Solar Farm in Queensland includes utility-scale photovoltaic generation assets.",
                score=0.93,
            ),
            WebSearchResult(
                title="Queensland energy project note",
                url="https://example.com/qld-energy-emerald",
                snippet="Project note describing Emerald solar generation, site location and installed capacity.",
                score=0.84,
            ),
        ]
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=web_search_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Trạm có capacty nhỏ nhất, tìm thông tin trạm đó trên internet",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.FACILITY_INFO
    assert response.model_used == "web-search-fallback"
    assert "capacity nhỏ nhất" in response.answer
    assert "Emerald" in response.answer
    assert web_search_client.queries
    assert "Emerald solar farm" in web_search_client.queries[0]


def test_irrelevant_facility_web_results_are_rejected() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO,
        confidence=0.9,
    )

    web_search_client = StubWebSearchClient(
        [
            WebSearchResult(
                title="Trạm vũ trụ Quốc tế - Wikipedia tiếng Việt",
                url="https://vi.wikipedia.org/wiki/Tr%E1%BA%A1m_v%C5%A9_tr%E1%BB%A5_Qu%E1%BB%91c_t%E1%BA%BF",
                snippet="Trạm vũ trụ Quốc tế là trạm nghiên cứu trong quỹ đạo thấp của Trái Đất.",
                score=0.99,
            ),
            WebSearchResult(
                title="Trạm sạc xe điện SolarEV",
                url="https://example.com/solarev",
                snippet="Mạng lưới trạm sạc xe điện tại Việt Nam.",
                score=0.88,
            ),
        ]
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=web_search_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Trạm có capacity lớn nhất là trạm nào, tìm thông tin trạm đó trên internet",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.FACILITY_INFO
    assert response.model_used == "deterministic-summary"
    assert "web_search_used" not in response.key_metrics
    assert "Trạm vũ trụ Quốc tế" not in response.answer
    assert "SolarEV" not in response.answer


def test_facility_query_without_internet_request_keeps_deterministic_summary() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO,
        confidence=0.9,
    )

    web_search_client = StubWebSearchClient(
        [
            WebSearchResult(
                title="Darlington Point Solar Farm - Overview",
                url="https://example.com/darlington-point-overview",
                snippet="Utility-scale PV station profile.",
                score=0.98,
            )
        ]
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=web_search_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Trạm có capacity lớn nhất là trạm nào?",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.topic == ChatTopic.FACILITY_INFO
    assert response.model_used == "deterministic-summary"
    assert "web_search_used" not in response.key_metrics
    assert web_search_client.queries == []


def test_facility_web_search_sanitizes_noisy_snippets() -> None:
    repository = MagicMock()
    repository.fetch_topic_metrics.return_value = _facility_metrics_payload()

    intent_service = MagicMock()
    intent_service.detect_intent.return_value = SimpleNamespace(
        topic=ChatTopic.FACILITY_INFO,
        confidence=0.9,
    )

    web_search_client = StubWebSearchClient(
        [
            WebSearchResult(
                title="Darlington Point public profile",
                url="https://example.com/darlington-point-profile",
                snippet=(
                    "Wikipedia The Free Encyclopedia | Country Australia | Location NSW "
                    "| Coordinates 34.6S 146.0E | Capacity 324 MW | Map | Contents"
                ),
                score=0.95,
            )
        ]
    )

    service = SolarAIChatService(
        repository=repository,
        intent_service=intent_service,
        model_router=None,
        history_repository=None,
        web_search_client=web_search_client,
    )

    response = service.handle_query(
        SolarChatRequest(
            message="Trạm lớn nhất là trạm nào, tìm thông tin trạm đó trên internet",
            role=ChatRole.DATA_ENGINEER,
            session_id=None,
        )
    )

    assert response.model_used == "web-search-fallback"
    assert "Tóm tắt: Nguồn này cung cấp hồ sơ dự án" in response.answer
    assert "The Free Encyclopedia" not in response.answer
    assert "|" not in response.answer
