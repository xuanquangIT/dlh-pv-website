from enum import Enum


class ChatRole(str, Enum):
    DATA_ENGINEER = "data_engineer"
    ML_ENGINEER = "ml_engineer"
    DATA_ANALYST = "data_analyst"
    ADMIN = "admin"


class ChatTopic(str, Enum):
    GENERAL = "general"
    SYSTEM_OVERVIEW = "system_overview"
    ENERGY_PERFORMANCE = "energy_performance"
    ML_MODEL = "ml_model"
    PIPELINE_STATUS = "pipeline_status"
    # 2026-04 cutover: ML now produces D+1, D+3, D+5, D+7 horizon-specific
    # forecasts (see pv.gold.daily_forecast_d{1,3,5,7}). Old "FORECAST_72H"
    # name kept as a deprecated alias so legacy chat_messages.topic rows
    # written before the cutover still deserialize.
    FORECAST_7D = "forecast_7d"
    FORECAST_72H = "forecast_72h"  # deprecated — pre-2026-04 alias for FORECAST_7D
    DATA_QUALITY_ISSUES = "data_quality_issues"
    FACILITY_INFO = "facility_info"
