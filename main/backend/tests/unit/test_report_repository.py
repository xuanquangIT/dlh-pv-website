from datetime import date
from unittest.mock import patch

import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.report_repository import ReportRepository


def _make_repo() -> ReportRepository:
    settings = SolarChatSettings()
    return ReportRepository(settings=settings)


def test_station_daily_report_no_data_includes_range_and_reason(caplog: pytest.LogCaptureFixture) -> None:
    repo = _make_repo()

    with patch.object(repo, "_station_daily_report_trino", return_value=[]), patch.object(
        repo,
        "_station_report_date_range_trino",
        return_value=("2026-01-01", "2026-04-06"),
    ):
        with caplog.at_level("INFO"):
            metrics, sources = repo.fetch_station_daily_report(
                anchor_date=date(2024, 3, 1),
                metrics=["energy_mwh", "aqi_value"],
            )

    assert metrics["report_date"] == "2024-03-01"
    assert metrics["station_count"] == 0
    assert metrics["has_data"] is False
    assert metrics["no_data_reason"] == "không có dữ liệu cho ngày 2024-03-01"
    assert metrics["available_date_min"] == "2026-01-01"
    assert metrics["available_date_max"] == "2026-04-06"
    assert len(sources) == 2
    assert "station_daily_report_no_data" in caplog.text


def test_station_daily_report_with_data_has_no_reason() -> None:
    repo = _make_repo()
    sample_station = [{"facility": "Avonlie", "energy_mwh": 100.0}]

    with patch.object(repo, "_station_daily_report_trino", return_value=sample_station), patch.object(
        repo,
        "_station_report_date_range_trino",
        return_value=("2026-01-01", "2026-04-06"),
    ):
        metrics, _ = repo.fetch_station_daily_report(
            anchor_date=date(2026, 3, 1),
            metrics=["energy_mwh"],
        )

    assert metrics["station_count"] == 1
    assert metrics["has_data"] is True
    assert metrics["no_data_reason"] is None
