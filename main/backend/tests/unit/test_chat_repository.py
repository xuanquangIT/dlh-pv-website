"""Unit tests for SolarChatRepository (chat_repository.py).

Tests cover:
- SolarChatRepository construction and MRO
- fetch_topic_metrics dispatch for every ChatTopic
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat.enums import ChatTopic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> SolarChatSettings:
    s = SolarChatSettings()
    s.databricks_host = overrides.get("databricks_host", "https://adb-123.azuredatabricks.net")
    s.databricks_token = overrides.get("databricks_token", "dapi-test-token")
    s.databricks_sql_http_path = overrides.get(
        "databricks_sql_http_path", "/sql/1.0/warehouses/abc123"
    )
    s.uc_catalog = overrides.get("uc_catalog", "pv")
    s.uc_silver_schema = overrides.get("uc_silver_schema", "silver")
    s.uc_gold_schema = overrides.get("uc_gold_schema", "gold")
    return s


def _make_repo(**overrides) -> SolarChatRepository:
    return SolarChatRepository(settings=_make_settings(**overrides))


DUMMY_METRICS: dict = {"key": "value"}
DUMMY_SOURCES: list = [{"layer": "Gold", "dataset": "test", "data_source": "databricks"}]


# ---------------------------------------------------------------------------
# Construction / inheritance
# ---------------------------------------------------------------------------

class TestSolarChatRepositoryConstruction:
    def test_instantiation_succeeds(self):
        repo = _make_repo()
        assert isinstance(repo, SolarChatRepository)

    def test_inherits_from_base_components(self):
        from app.repositories.solar_ai_chat.extreme_repository import ExtremeRepository
        from app.repositories.solar_ai_chat.kpi_repository import KpiRepository
        from app.repositories.solar_ai_chat.report_repository import ReportRepository
        from app.repositories.solar_ai_chat.topic_repository import TopicRepository

        repo = _make_repo()
        assert isinstance(repo, ExtremeRepository)
        assert isinstance(repo, TopicRepository)
        assert isinstance(repo, ReportRepository)
        assert isinstance(repo, KpiRepository)

    def test_catalog_set_from_settings(self):
        repo = _make_repo(uc_catalog="my_catalog")
        assert repo._catalog == "my_catalog"

    def test_silver_schema_set_from_settings(self):
        repo = _make_repo(uc_silver_schema="silver_test")
        assert repo._silver_schema == "silver_test"

    def test_gold_schema_set_from_settings(self):
        repo = _make_repo(uc_gold_schema="gold_test")
        assert repo._gold_schema == "gold_test"


# ---------------------------------------------------------------------------
# fetch_topic_metrics
# ---------------------------------------------------------------------------

class TestFetchTopicMetricsDispatch:
    """Verify that fetch_topic_metrics delegates to the correct handler."""

    def _make_patched_repo(self) -> SolarChatRepository:
        return _make_repo()

    def test_general_topic_calls_general_greeting(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_general_greeting", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.GENERAL)
        mock_fn.assert_called_once()
        assert metrics == DUMMY_METRICS
        assert sources == DUMMY_SOURCES

    def test_system_overview_topic(self):
        repo = self._make_patched_repo()
        args = {"some": "arg"}
        with patch.object(repo, "_system_overview", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.SYSTEM_OVERVIEW, arguments=args)
        mock_fn.assert_called_once_with(args)
        assert metrics == DUMMY_METRICS

    def test_energy_performance_topic(self):
        repo = self._make_patched_repo()
        args = {"facility": "WRSF1"}
        with patch.object(repo, "_energy_performance", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.ENERGY_PERFORMANCE, arguments=args)
        mock_fn.assert_called_once_with(args)
        assert metrics == DUMMY_METRICS

    def test_ml_model_topic(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_ml_model", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.ML_MODEL)
        mock_fn.assert_called_once()

    def test_pipeline_status_topic(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_pipeline_status", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.PIPELINE_STATUS)
        mock_fn.assert_called_once()

    def test_forecast_72h_topic(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_forecast_72h", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.FORECAST_72H)
        mock_fn.assert_called_once()

    def test_data_quality_issues_topic(self):
        repo = self._make_patched_repo()
        args = {"days": 7}
        with patch.object(repo, "_data_quality_issues", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.DATA_QUALITY_ISSUES, arguments=args)
        mock_fn.assert_called_once_with(args)
        assert metrics == DUMMY_METRICS

    def test_facility_info_topic(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_facility_info", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            metrics, sources = repo.fetch_topic_metrics(ChatTopic.FACILITY_INFO)
        mock_fn.assert_called_once()

    def test_general_does_not_call_handlers_dict(self):
        """GENERAL topic short-circuits before the handlers dict lookup."""
        repo = self._make_patched_repo()
        with patch.object(repo, "_general_greeting", return_value=(DUMMY_METRICS, DUMMY_SOURCES)):
            with patch.object(repo, "_system_overview") as mock_overview:
                repo.fetch_topic_metrics(ChatTopic.GENERAL)
        mock_overview.assert_not_called()

    def test_fetch_topic_metrics_no_arguments_defaults_to_none(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_system_overview", return_value=(DUMMY_METRICS, DUMMY_SOURCES)) as mock_fn:
            repo.fetch_topic_metrics(ChatTopic.SYSTEM_OVERVIEW)
        # arguments=None is passed through to lambda
        mock_fn.assert_called_once_with(None)

    def test_fetch_topic_metrics_returns_tuple(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_general_greeting", return_value=({"a": 1}, [{"layer": "X"}])):
            result = repo.fetch_topic_metrics(ChatTopic.GENERAL)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_handler_exception_propagates(self):
        repo = self._make_patched_repo()
        with patch.object(repo, "_ml_model", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                repo.fetch_topic_metrics(ChatTopic.ML_MODEL)
