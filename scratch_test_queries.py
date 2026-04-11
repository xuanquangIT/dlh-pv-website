import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main', 'backend')))

from app.services.databricks_service import get_daily_forecast, get_model_monitoring_metrics, get_registry_models

print("--- Forecast Daily ---")
try:
    f = get_daily_forecast()
    print(len(f), f[:2])
except Exception as e:
    print(f"Error: {e}")

print("--- Model Monitoring ---")
try:
    m = get_model_monitoring_metrics()
    print(len(m), m[:2])
except Exception as e:
    print(f"Error: {e}")

print("--- Registry Models ---")
try:
    r = get_registry_models()
    print(len(r), r[:2])
except Exception as e:
    print(f"Error: {e}")
