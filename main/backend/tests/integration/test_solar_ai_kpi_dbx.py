import pytest
from app.core.settings import get_solar_chat_settings
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat.enums import ChatTopic

@pytest.fixture(scope="module")
def repository():
    settings = get_solar_chat_settings()
    return SolarChatRepository(settings)

def test_system_overview_databricks_30_days(repository: SolarChatRepository):
    # Test Default 30 days
    metrics, sources = repository.fetch_topic_metrics(ChatTopic.SYSTEM_OVERVIEW, {"timeframe_days": 30})
    
    # Verify it used Databricks (not fallback CSV)
    assert any(s.get("data_source") == "databricks" for s in sources), "Should fetch from Databricks, not CSV fallback"
    
    assert "production_output_mwh" in metrics
    assert "window_days" in metrics
    assert metrics["window_days"] == 30
    assert metrics["production_output_mwh"] > 0

def test_system_overview_databricks_today(repository: SolarChatRepository):
    # Test Today (timeframe_days = 0)
    metrics, sources = repository.fetch_topic_metrics(ChatTopic.SYSTEM_OVERVIEW, {"timeframe_days": 0})
    
    assert any(s.get("data_source") == "databricks" for s in sources)
    assert metrics["window_days"] == 0
    # Today's output might be 0 if no data has arrived yet, so we just check it exists
    assert "production_output_mwh" in metrics

def test_energy_performance_databricks_1_day(repository: SolarChatRepository):
    # Test 1 day lookback
    metrics, sources = repository.fetch_topic_metrics(ChatTopic.ENERGY_PERFORMANCE, {"timeframe_days": 1})
    
    assert any(s.get("data_source") == "databricks" for s in sources)
    assert metrics["window_days"] == 1
    assert "top_facilities" in metrics
    assert "bottom_facilities" in metrics

    # The sum of top and bottom facilities should not be empty
    assert len(metrics["top_facilities"]) > 0
