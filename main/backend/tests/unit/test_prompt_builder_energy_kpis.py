import pytest

from app.schemas.solar_ai_chat import ChatTopic
from app.services.solar_ai_chat.prompt_builder import build_fallback_summary


ENERGY_METRICS_SAMPLE = {
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
}


@pytest.mark.parametrize(
    "message",
    [
        "Các chỉ số thường được dùng để đánh giá một nhà máy điện năng lượng mặt trời có hoạt động tốt không?",
        "Những tiêu chí nào để đánh giá hiệu quả vận hành của nhà máy điện mặt trời?",
        "Cho mình hỏi KPI nào dùng để assess solar plant health?",
        "Chi so nao de danh gia nha may solar hoat dong tot khong",
        "What metrics should I use to evaluate if a solar power plant is performing well?",
        "Which indicators are commonly used to assess solar plant performance?",
    ],
)
def test_metric_evaluation_queries_return_kpi_explanation(message: str) -> None:
    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=ENERGY_METRICS_SAMPLE,
        sources=[],
        user_message=message,
    )

    assert "Performance Ratio (PR)" in summary
    assert "Capacity Factor (CF)" in summary
    assert "Specific Yield" in summary
    assert "Top facilities:" not in summary
    assert "Peak hours:" not in summary


def test_metric_evaluation_query_with_ranking_marker_keeps_data_ranking_path() -> None:
    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=ENERGY_METRICS_SAMPLE,
        sources=[],
        user_message="Top metrics to evaluate plant performance today",
    )

    assert "Top facilities:" in summary
    assert "Performance Ratio (PR)" not in summary


def test_regular_energy_performance_question_still_returns_energy_summary() -> None:
    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=ENERGY_METRICS_SAMPLE,
        sources=[],
        user_message="Top facilities and peak hours today",
    )

    assert "Top facilities:" in summary
    assert "Peak hours:" in summary
    assert "Tomorrow's forecast" in summary


def test_top_two_facility_comparison_query_returns_comparison_summary() -> None:
    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=ENERGY_METRICS_SAMPLE,
        sources=[],
        user_message="So sánh 2 facilities lớn nhất mà hệ thống đang theo dõi",
    )

    assert "So sánh Top 2 Facilities" in summary
    assert "Darlington Point" in summary
    assert "Avonlie" in summary
    assert "Chênh lệch tuyệt đối" in summary
    assert "không phải capacity lắp đặt" in summary
    assert "Tỷ trọng trong Top 2" in summary
