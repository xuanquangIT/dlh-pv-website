"""Centralized RBAC permission maps for the Solar AI Chat module.

Single source of truth for both topic-level (chat_service.py) and
tool-level (tool_executor.py) access control. Tool permissions are
derived automatically from topic permissions via TOOL_NAME_TO_TOPIC,
so adding a new tool only requires updating one data structure.
"""
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.tools import TOOL_NAME_TO_TOPIC

# Topic-level permissions (used by SolarAIChatService for regex / RAG path).
# Note: extreme metric tools (aqi, energy, weather) are tool-only and not
# represented as ChatTopic members, so they are handled separately below.
ROLE_TOPIC_PERMISSIONS: dict[ChatRole, set[ChatTopic]] = {
    ChatRole.DATA_ENGINEER: {
        ChatTopic.GENERAL,
        ChatTopic.SYSTEM_OVERVIEW,
        ChatTopic.ENERGY_PERFORMANCE,
        ChatTopic.PIPELINE_STATUS,
        ChatTopic.FORECAST_72H,
        ChatTopic.DATA_QUALITY_ISSUES,
        ChatTopic.FACILITY_INFO,
    },
    ChatRole.ML_ENGINEER: {
        ChatTopic.GENERAL,
        ChatTopic.SYSTEM_OVERVIEW,
        ChatTopic.ENERGY_PERFORMANCE,
        ChatTopic.ML_MODEL,
        ChatTopic.FORECAST_72H,
        ChatTopic.FACILITY_INFO,
    },
    ChatRole.DATA_ANALYST: {
        ChatTopic.GENERAL,
        ChatTopic.SYSTEM_OVERVIEW,
        ChatTopic.ENERGY_PERFORMANCE,
        ChatTopic.ML_MODEL,
        ChatTopic.FORECAST_72H,
        ChatTopic.DATA_QUALITY_ISSUES,
        ChatTopic.FACILITY_INFO,
    },
    ChatRole.ADMIN: {
        ChatTopic.GENERAL,
        ChatTopic.SYSTEM_OVERVIEW,
        ChatTopic.ENERGY_PERFORMANCE,
        ChatTopic.ML_MODEL,
        ChatTopic.PIPELINE_STATUS,
        ChatTopic.FORECAST_72H,
        ChatTopic.DATA_QUALITY_ISSUES,
        ChatTopic.FACILITY_INFO,
    },
}

# Extra tool-level grants that have no ChatTopic equivalent (extreme metrics).
# These are appended to each role's derived tool permissions below.
_EXTRA_TOOL_GRANTS: dict[ChatRole, set[str]] = {
    ChatRole.DATA_ENGINEER: {"get_extreme_aqi", "get_extreme_energy", "get_extreme_weather", "query_gold_kpi"},
    ChatRole.ML_ENGINEER: {"get_extreme_energy", "get_extreme_weather", "query_gold_kpi"},
    ChatRole.DATA_ANALYST: {"get_extreme_aqi", "get_extreme_energy", "get_extreme_weather", "query_gold_kpi"},
    ChatRole.ADMIN: {"get_extreme_aqi", "get_extreme_energy", "get_extreme_weather", "query_gold_kpi"},
}


def _build_tool_permissions(
    role_topics: dict[ChatRole, set[ChatTopic]],
    tool_topic_map: dict[str, str],
    extra_grants: dict[ChatRole, set[str]],
) -> dict[ChatRole, set[str]]:
    """Derive tool permissions from topic permissions plus explicit extra grants."""
    result: dict[ChatRole, set[str]] = {}
    for role, topics in role_topics.items():
        topic_values = {t.value for t in topics}
        allowed_tools: set[str] = {
            name for name, topic_val in tool_topic_map.items()
            if topic_val in topic_values
        }
        allowed_tools |= extra_grants.get(role, set())
        if allowed_tools:
            allowed_tools.add("search_documents")
        result[role] = allowed_tools
    return result


ROLE_TOOL_PERMISSIONS: dict[ChatRole, set[str]] = _build_tool_permissions(
    ROLE_TOPIC_PERMISSIONS,
    TOOL_NAME_TO_TOPIC,
    _EXTRA_TOOL_GRANTS,
)
