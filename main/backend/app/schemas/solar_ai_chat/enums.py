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
    FORECAST_72H = "forecast_72h"
    DATA_QUALITY_ISSUES = "data_quality_issues"
