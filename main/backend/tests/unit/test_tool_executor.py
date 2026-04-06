from unittest.mock import MagicMock

import pytest

from app.repositories.solar_ai_chat.chat_repository import SolarChatRepository
from app.schemas.solar_ai_chat.enums import ChatRole, ChatTopic
from app.schemas.solar_ai_chat.tools import TOOL_NAME_TO_TOPIC
from app.services.solar_ai_chat.tool_executor import ToolExecutor


@pytest.fixture()
def mock_repository() -> MagicMock:
    repo = MagicMock(spec=SolarChatRepository)
    repo.fetch_topic_metrics.return_value = (
        {"facility_count": 8},
        [{"layer": "Gold", "dataset": "lh_gold_dim_facility", "data_source": "trino"}],
    )
    repo.fetch_extreme_aqi.return_value = (
        {"extreme_metric": "aqi", "query_type": "highest", "highest_aqi_value": 120},
        [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality", "data_source": "trino"}],
    )
    repo.fetch_extreme_energy.return_value = (
        {"extreme_metric": "energy", "query_type": "highest", "highest_energy_mwh": 55.5},
        [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy", "data_source": "trino"}],
    )
    repo.fetch_extreme_weather.return_value = (
        {"extreme_metric": "weather", "query_type": "highest", "highest_weather_value": 38.2},
        [{"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather", "data_source": "trino"}],
    )
    repo.fetch_station_daily_report.return_value = (
        {
            "report_date": "2025-05-05",
            "metrics_requested": ["aqi_value", "energy_mwh", "shortwave_radiation"],
            "stations": [
                {"facility": "Avonlie", "energy_mwh": 120.5, "shortwave_radiation": 450.2, "aqi_value": 15.3},
                {"facility": "Bomen", "energy_mwh": 95.1, "shortwave_radiation": 410.8, "aqi_value": 12.1},
            ],
            "station_count": 2,
        },
        [
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_energy", "data_source": "trino"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_weather", "data_source": "trino"},
            {"layer": "Silver", "dataset": "lh_silver_clean_hourly_air_quality", "data_source": "trino"},
        ],
    )
    return repo


@pytest.fixture()
def executor(mock_repository: MagicMock) -> ToolExecutor:
    return ToolExecutor(mock_repository)


class TestToolDispatch:
    @pytest.mark.parametrize("tool_name,expected_topic", [
        ("get_system_overview", ChatTopic.SYSTEM_OVERVIEW),
        ("get_energy_performance", ChatTopic.ENERGY_PERFORMANCE),
        ("get_ml_model_info", ChatTopic.ML_MODEL),
        ("get_pipeline_status", ChatTopic.PIPELINE_STATUS),
        ("get_forecast_72h", ChatTopic.FORECAST_72H),
        ("get_data_quality_issues", ChatTopic.DATA_QUALITY_ISSUES),
    ])
    def test_topic_tools_call_fetch_topic_metrics(
        self, executor: ToolExecutor, mock_repository: MagicMock,
        tool_name: str, expected_topic: ChatTopic,
    ) -> None:
        metrics, sources = executor.execute(tool_name, {}, ChatRole.ADMIN)
        mock_repository.fetch_topic_metrics.assert_called_once_with(expected_topic)
        assert "facility_count" in metrics

    def test_extreme_aqi_dispatch(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"query_type": "highest", "timeframe": "day"}
        metrics, _ = executor.execute("get_extreme_aqi", args, ChatRole.ADMIN)
        mock_repository.fetch_extreme_aqi.assert_called_once()
        assert metrics["extreme_metric"] == "aqi"

    def test_extreme_energy_dispatch(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"query_type": "highest", "timeframe": "week"}
        metrics, _ = executor.execute("get_extreme_energy", args, ChatRole.ADMIN)
        mock_repository.fetch_extreme_energy.assert_called_once()
        assert metrics["extreme_metric"] == "energy"

    def test_extreme_weather_dispatch(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"query_type": "lowest", "timeframe": "day", "weather_metric": "temperature_2m"}
        metrics, _ = executor.execute("get_extreme_weather", args, ChatRole.ADMIN)
        mock_repository.fetch_extreme_weather.assert_called_once()
        call_kwargs = mock_repository.fetch_extreme_weather.call_args[1]
        assert call_kwargs["weather_metric"] == "temperature_2m"
        assert call_kwargs["weather_metric_label"] == "temperature"
        assert call_kwargs["weather_unit"] == "C"

    def test_unknown_tool_raises(self, executor: ToolExecutor) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            executor.execute("nonexistent_tool", {}, ChatRole.ADMIN)

    def test_anchor_date_parsed(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"query_type": "highest", "timeframe": "day", "anchor_date": "2025-05-20"}
        executor.execute("get_extreme_aqi", args, ChatRole.ADMIN)
        from datetime import date
        call_kwargs = mock_repository.fetch_extreme_aqi.call_args[1]
        assert call_kwargs["anchor_date"] == date(2025, 5, 20)

    def test_invalid_anchor_date_passes_none(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"query_type": "highest", "timeframe": "day", "anchor_date": "not-a-date"}
        executor.execute("get_extreme_aqi", args, ChatRole.ADMIN)
        call_kwargs = mock_repository.fetch_extreme_aqi.call_args[1]
        assert call_kwargs["anchor_date"] is None


class TestToolRBAC:
    def test_admin_can_access_all_topic_tools(self, executor: ToolExecutor) -> None:
        topic_tools = [
            "get_system_overview", "get_energy_performance", "get_ml_model_info",
            "get_pipeline_status", "get_forecast_72h", "get_data_quality_issues",
        ]
        for tool_name in topic_tools:
            executor.execute(tool_name, {}, ChatRole.ADMIN)

    def test_ml_engineer_denied_extreme_aqi(self, executor: ToolExecutor) -> None:
        with pytest.raises(PermissionError):
            executor.execute("get_extreme_aqi", {"query_type": "highest", "timeframe": "day"}, ChatRole.ML_ENGINEER)

    def test_data_engineer_denied_ml_model(self, executor: ToolExecutor) -> None:
        with pytest.raises(PermissionError):
            executor.execute("get_ml_model_info", {}, ChatRole.DATA_ENGINEER)

    def test_ml_engineer_denied_pipeline_status(self, executor: ToolExecutor) -> None:
        with pytest.raises(PermissionError):
            executor.execute("get_pipeline_status", {}, ChatRole.ML_ENGINEER)

    def test_data_engineer_allowed_energy(self, executor: ToolExecutor) -> None:
        executor.execute("get_extreme_energy", {"query_type": "highest", "timeframe": "day"}, ChatRole.DATA_ENGINEER)


class TestStationDailyReport:
    def test_dispatch_calls_repository(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"anchor_date": "2025-05-05", "metrics": ["energy_mwh", "aqi_value"]}
        metrics, sources = executor.execute("get_station_daily_report", args, ChatRole.DATA_ENGINEER)
        mock_repository.fetch_station_daily_report.assert_called_once()
        assert metrics["station_count"] == 2
        assert len(sources) == 3

    def test_dispatch_without_anchor_date_uses_today(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        from datetime import date
        executor.execute("get_station_daily_report", {}, ChatRole.ADMIN)
        call_kwargs = mock_repository.fetch_station_daily_report.call_args[1]
        assert call_kwargs["anchor_date"] == date.today()

    def test_all_roles_can_access(self, executor: ToolExecutor) -> None:
        args = {"anchor_date": "2025-05-05"}
        for role in ChatRole:
            executor.execute("get_station_daily_report", args, role)

    def test_metrics_filter_passed(self, executor: ToolExecutor, mock_repository: MagicMock) -> None:
        args = {"anchor_date": "2025-05-05", "metrics": ["shortwave_radiation"]}
        executor.execute("get_station_daily_report", args, ChatRole.DATA_ENGINEER)
        call_kwargs = mock_repository.fetch_station_daily_report.call_args[1]
        assert call_kwargs["metrics"] == ["shortwave_radiation"]
