from app.services.databricks_service import get_job_info

try:
    print(get_job_info(28718564084384))
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
