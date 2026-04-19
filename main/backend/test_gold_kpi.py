import os
from types import SimpleNamespace
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository

def run():
    settings = SimpleNamespace(
        databricks_host=os.getenv("DATABRICKS_HOST", "mock"),
        databricks_token=os.getenv("DATABRICKS_TOKEN", "mock"),
        databricks_catalog="pv",
        uc_catalog="pv",
        analytics_lookback_days=30
    )
    repo = SolarChatRepository(settings)
    
    data, sources = repo.fetch_gold_kpi({
        'table_name': 'system_kpi', 
        'anchor_date': '2026-03-04', 
        'station_name': 'ALL', 
        'limit': 100
    })
    print("ROWS:", data.get('rows'))

if __name__ == '__main__':
    run()
