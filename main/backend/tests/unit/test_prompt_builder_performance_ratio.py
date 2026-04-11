from app.schemas.solar_ai_chat import ChatTopic
from app.services.solar_ai_chat.prompt_builder import build_fallback_summary


def test_performance_ratio_definition_query_prefers_concept_answer() -> None:
    metrics = {
        "top_performance_ratio_facilities": [
            {"facility": "A", "performance_ratio_pct": 88.1},
            {"facility": "B", "performance_ratio_pct": 86.4},
        ]
    }

    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=metrics,
        sources=[],
        user_message="Performance Ratio la gi?",
    )

    assert "Performance Ratio (PR)" in summary
    assert "capacity_factor_pct" in summary
    assert "Top trạm" not in summary


def test_performance_ratio_ranking_query_returns_top_facilities() -> None:
    metrics = {
        "top_performance_ratio_facilities": [
            {"facility": "A", "performance_ratio_pct": 88.1},
            {"facility": "B", "performance_ratio_pct": 86.4},
        ]
    }

    summary = build_fallback_summary(
        topic=ChatTopic.ENERGY_PERFORMANCE,
        metrics=metrics,
        sources=[],
        user_message="Top performance ratio stations",
    )

    assert "Top trạm theo performance ratio" in summary
    assert "A (88.1%)" in summary
