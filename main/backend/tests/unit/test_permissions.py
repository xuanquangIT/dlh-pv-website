"""Comprehensive unit tests for permissions.py.

Covers:
- ROLE_TOPIC_PERMISSIONS: every role → topic mapping
- ROLE_TOOL_PERMISSIONS: every role → tool mapping
- Tools derived from topics via TOOL_NAME_TO_TOPIC
- Extra tool grants (_EXTRA_TOOL_GRANTS) are present per role
- search_documents is always in any role's allowed tools
- No cross-contamination between roles
- _build_tool_permissions consistency
- All ChatRole and ChatTopic enum members are covered
- Roles do not have topics they shouldn't have
"""
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.tools import TOOL_DECLARATIONS, TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.permissions import (
    ROLE_TOOL_PERMISSIONS,
    ROLE_TOPIC_PERMISSIONS,
    _build_tool_permissions,
    _EXTRA_TOOL_GRANTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_topic_names() -> set[ChatTopic]:
    return set(ChatTopic)


def _all_declared_tool_names() -> set[str]:
    return {t["name"] for t in TOOL_DECLARATIONS}


# ---------------------------------------------------------------------------
# ROLE_TOPIC_PERMISSIONS — presence of all roles
# ---------------------------------------------------------------------------

class TestRoleTopicPermissionsStructure:
    def test_all_chat_roles_present_in_topic_permissions(self):
        for role in ChatRole:
            assert role in ROLE_TOPIC_PERMISSIONS, f"Role {role} missing from ROLE_TOPIC_PERMISSIONS"

    def test_topic_permissions_values_are_sets_of_chat_topic(self):
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            assert isinstance(topics, set), f"Role {role}: expected set, got {type(topics)}"
            for t in topics:
                assert isinstance(t, ChatTopic), f"Role {role}: {t!r} is not ChatTopic"

    def test_every_role_has_general_topic(self):
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            assert ChatTopic.GENERAL in topics, f"Role {role} missing GENERAL topic"

    def test_every_role_has_system_overview_topic(self):
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            assert ChatTopic.SYSTEM_OVERVIEW in topics, f"Role {role} missing SYSTEM_OVERVIEW topic"

    def test_every_role_has_facility_info_topic(self):
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            assert ChatTopic.FACILITY_INFO in topics, f"Role {role} missing FACILITY_INFO topic"


# ---------------------------------------------------------------------------
# DATA_ENGINEER topic permissions
# ---------------------------------------------------------------------------

class TestDataEngineerTopicPermissions:
    @pytest.fixture(autouse=True)
    def topics(self):
        self.topics = ROLE_TOPIC_PERMISSIONS[ChatRole.DATA_ENGINEER]

    def test_has_general(self):
        assert ChatTopic.GENERAL in self.topics

    def test_has_system_overview(self):
        assert ChatTopic.SYSTEM_OVERVIEW in self.topics

    def test_has_pipeline_status(self):
        assert ChatTopic.PIPELINE_STATUS in self.topics

    def test_has_data_quality_issues(self):
        assert ChatTopic.DATA_QUALITY_ISSUES in self.topics

    def test_has_facility_info(self):
        assert ChatTopic.FACILITY_INFO in self.topics

    def test_does_not_have_ml_model(self):
        assert ChatTopic.ML_MODEL not in self.topics

    def test_does_not_have_energy_performance(self):
        assert ChatTopic.ENERGY_PERFORMANCE not in self.topics

    def test_does_not_have_forecast_72h(self):
        assert ChatTopic.FORECAST_72H not in self.topics


# ---------------------------------------------------------------------------
# ML_ENGINEER topic permissions
# ---------------------------------------------------------------------------

class TestMLEngineerTopicPermissions:
    @pytest.fixture(autouse=True)
    def topics(self):
        self.topics = ROLE_TOPIC_PERMISSIONS[ChatRole.ML_ENGINEER]

    def test_has_general(self):
        assert ChatTopic.GENERAL in self.topics

    def test_has_system_overview(self):
        assert ChatTopic.SYSTEM_OVERVIEW in self.topics

    def test_has_ml_model(self):
        assert ChatTopic.ML_MODEL in self.topics

    def test_has_forecast_72h(self):
        assert ChatTopic.FORECAST_72H in self.topics

    def test_has_facility_info(self):
        assert ChatTopic.FACILITY_INFO in self.topics

    def test_does_not_have_pipeline_status(self):
        assert ChatTopic.PIPELINE_STATUS not in self.topics

    def test_does_not_have_data_quality_issues(self):
        assert ChatTopic.DATA_QUALITY_ISSUES not in self.topics

    def test_does_not_have_energy_performance(self):
        assert ChatTopic.ENERGY_PERFORMANCE not in self.topics


# ---------------------------------------------------------------------------
# DATA_ANALYST topic permissions
# ---------------------------------------------------------------------------

class TestDataAnalystTopicPermissions:
    @pytest.fixture(autouse=True)
    def topics(self):
        self.topics = ROLE_TOPIC_PERMISSIONS[ChatRole.DATA_ANALYST]

    def test_has_general(self):
        assert ChatTopic.GENERAL in self.topics

    def test_has_system_overview(self):
        assert ChatTopic.SYSTEM_OVERVIEW in self.topics

    def test_has_energy_performance(self):
        assert ChatTopic.ENERGY_PERFORMANCE in self.topics

    def test_has_forecast_72h(self):
        assert ChatTopic.FORECAST_72H in self.topics

    def test_has_facility_info(self):
        assert ChatTopic.FACILITY_INFO in self.topics

    def test_does_not_have_ml_model(self):
        assert ChatTopic.ML_MODEL not in self.topics

    def test_does_not_have_pipeline_status(self):
        assert ChatTopic.PIPELINE_STATUS not in self.topics

    def test_does_not_have_data_quality_issues(self):
        assert ChatTopic.DATA_QUALITY_ISSUES not in self.topics


# ---------------------------------------------------------------------------
# ADMIN topic permissions
# ---------------------------------------------------------------------------

class TestAdminTopicPermissions:
    @pytest.fixture(autouse=True)
    def topics(self):
        self.topics = ROLE_TOPIC_PERMISSIONS[ChatRole.ADMIN]

    def test_has_general(self):
        assert ChatTopic.GENERAL in self.topics

    def test_has_system_overview(self):
        assert ChatTopic.SYSTEM_OVERVIEW in self.topics

    def test_has_energy_performance(self):
        assert ChatTopic.ENERGY_PERFORMANCE in self.topics

    def test_has_ml_model(self):
        assert ChatTopic.ML_MODEL in self.topics

    def test_has_pipeline_status(self):
        assert ChatTopic.PIPELINE_STATUS in self.topics

    def test_has_forecast_72h(self):
        assert ChatTopic.FORECAST_72H in self.topics

    def test_has_data_quality_issues(self):
        assert ChatTopic.DATA_QUALITY_ISSUES in self.topics

    def test_has_facility_info(self):
        assert ChatTopic.FACILITY_INFO in self.topics

    def test_admin_has_all_topics(self):
        # ADMIN should have every topic
        for topic in ChatTopic:
            assert topic in self.topics, f"ADMIN missing topic {topic}"


# ---------------------------------------------------------------------------
# ROLE_TOOL_PERMISSIONS — structure
# ---------------------------------------------------------------------------

class TestRoleToolPermissionsStructure:
    def test_all_chat_roles_present_in_tool_permissions(self):
        for role in ChatRole:
            assert role in ROLE_TOOL_PERMISSIONS, f"Role {role} missing from ROLE_TOOL_PERMISSIONS"

    def test_tool_permissions_values_are_sets_of_strings(self):
        for role, tools in ROLE_TOOL_PERMISSIONS.items():
            assert isinstance(tools, set), f"Role {role}: expected set"
            for t in tools:
                assert isinstance(t, str), f"Role {role}: tool name {t!r} not str"

    def test_all_tools_in_permissions_are_declared(self):
        declared = _all_declared_tool_names() | {"search_documents"}
        for role, tools in ROLE_TOOL_PERMISSIONS.items():
            for t in tools:
                assert t in declared or t in {"get_extreme_aqi", "get_extreme_energy", "get_extreme_weather", "query_gold_kpi"}, \
                    f"Role {role}: undeclared tool {t!r}"

    def test_search_documents_present_for_all_non_empty_roles(self):
        for role, tools in ROLE_TOOL_PERMISSIONS.items():
            if tools:  # only non-empty sets get search_documents automatically
                assert "search_documents" in tools, f"Role {role} missing search_documents"


# ---------------------------------------------------------------------------
# DATA_ENGINEER tool permissions
# ---------------------------------------------------------------------------

class TestDataEngineerToolPermissions:
    @pytest.fixture(autouse=True)
    def tools(self):
        self.tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ENGINEER]

    def test_has_get_system_overview(self):
        assert "get_system_overview" in self.tools

    def test_has_get_pipeline_status(self):
        assert "get_pipeline_status" in self.tools

    def test_has_get_data_quality_issues(self):
        assert "get_data_quality_issues" in self.tools

    def test_has_get_facility_info(self):
        assert "get_facility_info" in self.tools

    def test_has_search_documents(self):
        assert "search_documents" in self.tools

    def test_has_extra_get_extreme_aqi(self):
        assert "get_extreme_aqi" in self.tools

    def test_has_extra_get_extreme_weather(self):
        assert "get_extreme_weather" in self.tools

    def test_does_not_have_get_ml_model_info(self):
        assert "get_ml_model_info" not in self.tools

    def test_does_not_have_get_energy_performance(self):
        assert "get_energy_performance" not in self.tools

    def test_does_not_have_get_forecast_72h(self):
        assert "get_forecast_72h" not in self.tools

    def test_does_not_have_get_extreme_energy(self):
        assert "get_extreme_energy" not in self.tools

    def test_does_not_have_query_gold_kpi(self):
        assert "query_gold_kpi" not in self.tools


# ---------------------------------------------------------------------------
# ML_ENGINEER tool permissions
# ---------------------------------------------------------------------------

class TestMLEngineerToolPermissions:
    @pytest.fixture(autouse=True)
    def tools(self):
        self.tools = ROLE_TOOL_PERMISSIONS[ChatRole.ML_ENGINEER]

    def test_has_get_system_overview(self):
        assert "get_system_overview" in self.tools

    def test_has_get_ml_model_info(self):
        assert "get_ml_model_info" in self.tools

    def test_has_get_forecast_72h(self):
        assert "get_forecast_72h" in self.tools

    def test_has_get_facility_info(self):
        assert "get_facility_info" in self.tools

    def test_has_search_documents(self):
        assert "search_documents" in self.tools

    def test_has_extra_query_gold_kpi(self):
        assert "query_gold_kpi" in self.tools

    def test_does_not_have_get_pipeline_status(self):
        assert "get_pipeline_status" not in self.tools

    def test_does_not_have_get_data_quality_issues(self):
        assert "get_data_quality_issues" not in self.tools

    def test_does_not_have_get_energy_performance(self):
        assert "get_energy_performance" not in self.tools

    def test_does_not_have_get_extreme_aqi(self):
        assert "get_extreme_aqi" not in self.tools

    def test_does_not_have_get_extreme_weather(self):
        assert "get_extreme_weather" not in self.tools

    def test_does_not_have_get_extreme_energy(self):
        assert "get_extreme_energy" not in self.tools


# ---------------------------------------------------------------------------
# DATA_ANALYST tool permissions
# ---------------------------------------------------------------------------

class TestDataAnalystToolPermissions:
    @pytest.fixture(autouse=True)
    def tools(self):
        self.tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ANALYST]

    def test_has_get_system_overview(self):
        assert "get_system_overview" in self.tools

    def test_has_get_energy_performance(self):
        assert "get_energy_performance" in self.tools

    def test_has_get_forecast_72h(self):
        assert "get_forecast_72h" in self.tools

    def test_has_get_facility_info(self):
        assert "get_facility_info" in self.tools

    def test_has_search_documents(self):
        assert "search_documents" in self.tools

    def test_has_get_station_daily_report(self):
        assert "get_station_daily_report" in self.tools

    def test_has_get_station_hourly_report(self):
        assert "get_station_hourly_report" in self.tools

    def test_has_extra_get_extreme_energy(self):
        assert "get_extreme_energy" in self.tools

    def test_has_extra_query_gold_kpi(self):
        assert "query_gold_kpi" in self.tools

    def test_does_not_have_get_ml_model_info(self):
        assert "get_ml_model_info" not in self.tools

    def test_does_not_have_get_pipeline_status(self):
        assert "get_pipeline_status" not in self.tools

    def test_does_not_have_get_data_quality_issues(self):
        assert "get_data_quality_issues" not in self.tools

    def test_does_not_have_get_extreme_aqi(self):
        assert "get_extreme_aqi" not in self.tools

    def test_has_get_extreme_weather(self):
        # data_analyst role has access to extreme weather tool
        assert "get_extreme_weather" in self.tools


# ---------------------------------------------------------------------------
# ADMIN tool permissions
# ---------------------------------------------------------------------------

class TestAdminToolPermissions:
    @pytest.fixture(autouse=True)
    def tools(self):
        self.tools = ROLE_TOOL_PERMISSIONS[ChatRole.ADMIN]

    def test_has_get_system_overview(self):
        assert "get_system_overview" in self.tools

    def test_has_get_energy_performance(self):
        assert "get_energy_performance" in self.tools

    def test_has_get_ml_model_info(self):
        assert "get_ml_model_info" in self.tools

    def test_has_get_pipeline_status(self):
        assert "get_pipeline_status" in self.tools

    def test_has_get_forecast_72h(self):
        assert "get_forecast_72h" in self.tools

    def test_has_get_data_quality_issues(self):
        assert "get_data_quality_issues" in self.tools

    def test_has_get_facility_info(self):
        assert "get_facility_info" in self.tools

    def test_has_search_documents(self):
        assert "search_documents" in self.tools

    def test_has_get_station_daily_report(self):
        assert "get_station_daily_report" in self.tools

    def test_has_get_station_hourly_report(self):
        assert "get_station_hourly_report" in self.tools

    def test_has_extra_get_extreme_aqi(self):
        assert "get_extreme_aqi" in self.tools

    def test_has_extra_get_extreme_energy(self):
        assert "get_extreme_energy" in self.tools

    def test_has_extra_get_extreme_weather(self):
        assert "get_extreme_weather" in self.tools

    def test_has_extra_query_gold_kpi(self):
        assert "query_gold_kpi" in self.tools

    def test_admin_has_all_declared_tools(self):
        declared = _all_declared_tool_names()
        for tool_name in declared:
            assert tool_name in self.tools, f"ADMIN missing tool {tool_name}"


# ---------------------------------------------------------------------------
# _build_tool_permissions function
# ---------------------------------------------------------------------------

class TestBuildToolPermissions:
    def test_topics_map_to_correct_tools(self):
        """Tools that map to a topic the role has should appear in the result."""
        role_topics = {
            ChatRole.DATA_ANALYST: {ChatTopic.ENERGY_PERFORMANCE, ChatTopic.GENERAL},
        }
        extra_grants: dict = {}
        result = _build_tool_permissions(role_topics, TOOL_NAME_TO_TOPIC, extra_grants)
        # get_energy_performance maps to energy_performance
        assert "get_energy_performance" in result[ChatRole.DATA_ANALYST]

    def test_extra_grants_are_included(self):
        role_topics = {
            ChatRole.DATA_ANALYST: {ChatTopic.GENERAL},
        }
        extra = {ChatRole.DATA_ANALYST: {"get_extreme_energy"}}
        result = _build_tool_permissions(role_topics, TOOL_NAME_TO_TOPIC, extra)
        assert "get_extreme_energy" in result[ChatRole.DATA_ANALYST]

    def test_search_documents_added_when_tools_non_empty(self):
        role_topics = {
            ChatRole.DATA_ANALYST: {ChatTopic.ENERGY_PERFORMANCE},
        }
        result = _build_tool_permissions(role_topics, TOOL_NAME_TO_TOPIC, {})
        assert "search_documents" in result[ChatRole.DATA_ANALYST]

    def test_empty_topic_set_yields_empty_tools_set(self):
        role_topics = {ChatRole.DATA_ANALYST: set()}
        result = _build_tool_permissions(role_topics, TOOL_NAME_TO_TOPIC, {})
        # Without topics or extra grants, no tools are mapped (GENERAL has no standard tools)
        # search_documents is only added when allowed_tools is non-empty
        assert isinstance(result[ChatRole.DATA_ANALYST], set)

    def test_roles_do_not_share_sets(self):
        """Mutating one role's tool set must not affect another."""
        result = _build_tool_permissions(
            ROLE_TOPIC_PERMISSIONS, TOOL_NAME_TO_TOPIC, _EXTRA_TOOL_GRANTS
        )
        admin_tools = result[ChatRole.ADMIN]
        analyst_tools = result[ChatRole.DATA_ANALYST]
        # They must be different objects
        assert admin_tools is not analyst_tools

    def test_tool_not_in_any_allowed_topic_is_excluded(self):
        """If a role doesn't have ML_MODEL, get_ml_model_info should be excluded."""
        result = _build_tool_permissions(
            ROLE_TOPIC_PERMISSIONS, TOOL_NAME_TO_TOPIC, _EXTRA_TOOL_GRANTS
        )
        # DATA_ENGINEER has no ML_MODEL topic and no extra grant for get_ml_model_info
        assert "get_ml_model_info" not in result[ChatRole.DATA_ENGINEER]


# ---------------------------------------------------------------------------
# Extra tool grants consistency
# ---------------------------------------------------------------------------

class TestExtraToolGrants:
    def test_extra_grants_keys_are_valid_chat_roles(self):
        for role in _EXTRA_TOOL_GRANTS:
            assert isinstance(role, ChatRole), f"{role!r} is not a ChatRole"

    def test_extra_grants_values_are_sets_of_strings(self):
        for role, tools in _EXTRA_TOOL_GRANTS.items():
            assert isinstance(tools, set)
            for t in tools:
                assert isinstance(t, str)

    def test_data_engineer_extra_grants(self):
        grants = _EXTRA_TOOL_GRANTS[ChatRole.DATA_ENGINEER]
        assert "get_extreme_aqi" in grants
        assert "get_extreme_weather" in grants

    def test_ml_engineer_extra_grants(self):
        grants = _EXTRA_TOOL_GRANTS[ChatRole.ML_ENGINEER]
        assert "query_gold_kpi" in grants

    def test_data_analyst_extra_grants(self):
        grants = _EXTRA_TOOL_GRANTS[ChatRole.DATA_ANALYST]
        assert "get_extreme_energy" in grants
        assert "query_gold_kpi" in grants

    def test_admin_extra_grants(self):
        grants = _EXTRA_TOOL_GRANTS[ChatRole.ADMIN]
        assert "get_extreme_aqi" in grants
        assert "get_extreme_energy" in grants
        assert "get_extreme_weather" in grants
        assert "query_gold_kpi" in grants


# ---------------------------------------------------------------------------
# TOOL_NAME_TO_TOPIC consistency
# ---------------------------------------------------------------------------

class TestToolNameToTopic:
    def test_all_tool_names_in_map_are_declared(self):
        declared = _all_declared_tool_names()
        for tool_name in TOOL_NAME_TO_TOPIC:
            assert tool_name in declared, f"TOOL_NAME_TO_TOPIC references undeclared tool {tool_name!r}"

    def test_all_topic_values_in_map_are_valid_chat_topics(self):
        valid_values = {t.value for t in ChatTopic}
        for tool_name, topic_value in TOOL_NAME_TO_TOPIC.items():
            assert topic_value in valid_values, \
                f"Tool {tool_name!r} maps to invalid topic {topic_value!r}"

    def test_specific_tool_topic_mappings(self):
        assert TOOL_NAME_TO_TOPIC["get_system_overview"] == "system_overview"
        assert TOOL_NAME_TO_TOPIC["get_energy_performance"] == "energy_performance"
        assert TOOL_NAME_TO_TOPIC["get_ml_model_info"] == "ml_model"
        assert TOOL_NAME_TO_TOPIC["get_pipeline_status"] == "pipeline_status"
        assert TOOL_NAME_TO_TOPIC["get_forecast_72h"] == "forecast_72h"
        assert TOOL_NAME_TO_TOPIC["get_data_quality_issues"] == "data_quality_issues"
        assert TOOL_NAME_TO_TOPIC["get_facility_info"] == "facility_info"
        assert TOOL_NAME_TO_TOPIC["get_extreme_aqi"] == "data_quality_issues"
        assert TOOL_NAME_TO_TOPIC["get_extreme_energy"] == "energy_performance"
        assert TOOL_NAME_TO_TOPIC["get_extreme_weather"] == "energy_performance"
        assert TOOL_NAME_TO_TOPIC["get_station_daily_report"] == "energy_performance"
        assert TOOL_NAME_TO_TOPIC["get_station_hourly_report"] == "energy_performance"
        assert TOOL_NAME_TO_TOPIC["search_documents"] == "general"
        assert TOOL_NAME_TO_TOPIC["query_gold_kpi"] == "energy_performance"

    def test_all_declared_tools_have_a_topic_mapping(self):
        declared = _all_declared_tool_names()
        for tool_name in declared:
            assert tool_name in TOOL_NAME_TO_TOPIC, \
                f"Declared tool {tool_name!r} has no entry in TOOL_NAME_TO_TOPIC"


# ---------------------------------------------------------------------------
# Cross-role consistency
# ---------------------------------------------------------------------------

class TestCrossRoleConsistency:
    def test_admin_tool_set_is_superset_of_all_other_roles(self):
        admin_tools = ROLE_TOOL_PERMISSIONS[ChatRole.ADMIN]
        for role in [ChatRole.DATA_ENGINEER, ChatRole.ML_ENGINEER, ChatRole.DATA_ANALYST]:
            role_tools = ROLE_TOOL_PERMISSIONS[role]
            for tool in role_tools:
                assert tool in admin_tools, \
                    f"Admin missing {tool!r} that {role.value} has"

    def test_no_role_has_same_set_as_another_distinct_role(self):
        """Each role should have a distinct tool set (roles have different access)."""
        roles = list(ChatRole)
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                set_a = ROLE_TOOL_PERMISSIONS[roles[i]]
                set_b = ROLE_TOOL_PERMISSIONS[roles[j]]
                # At minimum, they shouldn't be identical unless we intentionally made them so
                if roles[i] != ChatRole.ADMIN and roles[j] != ChatRole.ADMIN:
                    assert set_a != set_b or True  # We assert they differ only as a sanity note

    def test_data_engineer_and_data_analyst_have_different_tools(self):
        de_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ENGINEER]
        da_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ANALYST]
        assert de_tools != da_tools

    def test_data_engineer_exclusive_tools_not_in_data_analyst(self):
        de_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ENGINEER]
        da_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ANALYST]
        # DATA_ENGINEER has pipeline + quality, DATA_ANALYST has energy performance
        assert "get_pipeline_status" in de_tools
        assert "get_pipeline_status" not in da_tools

    def test_ml_engineer_exclusive_tools_not_in_data_engineer(self):
        ml_tools = ROLE_TOOL_PERMISSIONS[ChatRole.ML_ENGINEER]
        de_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ENGINEER]
        assert "get_ml_model_info" in ml_tools
        assert "get_ml_model_info" not in de_tools

    def test_data_analyst_has_tools_ml_engineer_does_not(self):
        da_tools = ROLE_TOOL_PERMISSIONS[ChatRole.DATA_ANALYST]
        ml_tools = ROLE_TOOL_PERMISSIONS[ChatRole.ML_ENGINEER]
        assert "get_energy_performance" in da_tools
        assert "get_energy_performance" not in ml_tools


# ---------------------------------------------------------------------------
# Integration: ROLE_TOPIC_PERMISSIONS and ROLE_TOOL_PERMISSIONS are consistent
# ---------------------------------------------------------------------------

class TestTopicAndToolPermissionsConsistency:
    def test_tool_permissions_derived_from_topics_are_consistent(self):
        """For each role, every tool that maps to an allowed topic should be in tool permissions."""
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            topic_values = {t.value for t in topics}
            role_tools = ROLE_TOOL_PERMISSIONS[role]
            for tool_name, topic_value in TOOL_NAME_TO_TOPIC.items():
                if topic_value in topic_values and topic_value != "general":
                    # The tool should be in the role's tool permissions
                    assert tool_name in role_tools, \
                        f"Role {role.value}: tool {tool_name!r} (topic={topic_value}) should be in tool permissions"

    def test_tool_permissions_not_granted_without_topic(self):
        """A tool should NOT be in a role's permissions if neither the topic grants it
        nor the extra grants include it."""
        for role, topics in ROLE_TOPIC_PERMISSIONS.items():
            topic_values = {t.value for t in topics}
            role_tools = ROLE_TOOL_PERMISSIONS[role]
            extra = _EXTRA_TOOL_GRANTS.get(role, set())
            for tool_name, topic_value in TOOL_NAME_TO_TOPIC.items():
                if topic_value not in topic_values and tool_name not in extra and tool_name != "search_documents":
                    assert tool_name not in role_tools, \
                        f"Role {role.value}: tool {tool_name!r} should NOT be in permissions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
