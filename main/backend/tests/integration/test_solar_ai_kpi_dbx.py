import pytest
from logging import getLogger
from app.core.settings import get_solar_chat_settings
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat.enums import ChatTopic

logger = getLogger(__name__)

@pytest.fixture(scope="module")
def repository():
    settings = get_solar_chat_settings()
    repo = SolarChatRepository(settings)
    return repo

def _spy_on_queries(repo: SolarChatRepository, topic: ChatTopic, arguments: dict):
    captured_queries = []
    original_execute_query = repo._execute_query
    
    def mock_execute_query(sql: str):
        captured_queries.append(sql)
        # Call the real implementation so the test functions normally
        return original_execute_query(sql)
        
    repo._execute_query = mock_execute_query
    try:
        metrics, sources = repo.fetch_topic_metrics(topic, arguments)
    finally:
        repo._execute_query = original_execute_query
        
    return metrics, sources, captured_queries

def test_system_overview_databricks_30_days(repository: SolarChatRepository):
    # Test Default 30 days
    metrics, sources, queries = _spy_on_queries(repository, ChatTopic.SYSTEM_OVERVIEW, {"timeframe_days": 30})
    
    print("\n\n--- CAPTURED SQL QUERIES: SYSTEM OVERVIEW (30 DAYS) ---")
    for q in queries:
        print(f"\n> {q}")
        
    # Verify it used Databricks (not fallback CSV)
    assert any(s.get("data_source") == "databricks" for s in sources), "Should fetch from Databricks, not CSV fallback"
    
    assert "production_output_mwh" in metrics
    assert "window_days" in metrics
    assert metrics["window_days"] == 30
    assert metrics["production_output_mwh"] > 0

def test_system_overview_databricks_today(repository: SolarChatRepository):
    # Test Today (timeframe_days = 0)
    metrics, sources, queries = _spy_on_queries(repository, ChatTopic.SYSTEM_OVERVIEW, {"timeframe_days": 0})
    
    print("\n\n--- CAPTURED SQL QUERIES: SYSTEM OVERVIEW (0 DAYS / TODAY) ---")
    for q in queries:
        print(f"\n> {q}")
        
    assert any(s.get("data_source") == "databricks" for s in sources)
    assert metrics["window_days"] == 0
    # Today's output might be 0 if no data has arrived yet, so we just check it exists
    assert "production_output_mwh" in metrics

def test_energy_performance_databricks_1_day(repository: SolarChatRepository):
    # Test 1 day lookback
    metrics, sources, queries = _spy_on_queries(repository, ChatTopic.ENERGY_PERFORMANCE, {"timeframe_days": 1})
    
    print("\n\n--- CAPTURED SQL QUERIES: ENERGY PERFORMANCE (1 DAY) ---")
    for q in queries:
        print(f"\n> {q}")
        
    assert any(s.get("data_source") == "databricks" for s in sources)
    assert metrics["window_days"] == 1
    assert "top_facilities" in metrics
    assert "bottom_facilities" in metrics

    # The sum of top and bottom facilities should not be empty
    assert len(metrics["top_facilities"]) > 0
