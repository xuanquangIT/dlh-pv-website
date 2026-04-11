from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.settings import get_solar_chat_settings


@dataclass(frozen=True)
class BilingualPromptCase:
    case_id: str
    description: str
    vi_prompt: str
    en_prompt: str
    expected_topic: str
    parity_paths: tuple[str, ...] = ()
    databricks_validator: str | None = None


@dataclass(frozen=True)
class ConversationTurnCase:
    vi_prompt: str
    en_prompt: str
    expected_topic: str | None = None
    parity_paths: tuple[str, ...] = ()
    databricks_validator: str | None = None
    requires_anchor_station: bool = False


@dataclass(frozen=True)
class LongConversationCase:
    case_id: str
    description: str
    turns: tuple[ConversationTurnCase, ...]


@dataclass(frozen=True)
class AssertionResult:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


SINGLE_TURN_CASES: tuple[BilingualPromptCase, ...] = (
    BilingualPromptCase(
        case_id="facility_timezone_parity",
        description="Facility timezone response must be bilingual-consistent and Databricks-verified.",
        vi_prompt="Múi giờ của các trạm hiện tại là gì?",
        en_prompt="What are the current time zones of all stations?",
        expected_topic="facility_info",
        parity_paths=(
            "facility_count",
            "facilities[0].timezone_name",
            "facilities[0].timezone_utc_offset",
        ),
        databricks_validator="facility_info",
    ),
    BilingualPromptCase(
        case_id="system_overview_consistency",
        description="System overview metrics must match Databricks and remain stable across Vietnamese/English prompts.",
        vi_prompt="Cho tôi tổng quan hệ thống hiện tại gồm sản lượng, R-squared và chất lượng dữ liệu.",
        en_prompt="Give me the current system overview including production output, R-squared, and data quality.",
        expected_topic="system_overview",
        parity_paths=(
            "production_output_mwh",
            "r_squared",
            "data_quality_score",
            "facility_count",
        ),
        databricks_validator="system_overview",
    ),
    BilingualPromptCase(
        case_id="energy_performance_consistency",
        description="Top energy facilities and tomorrow forecast must align across languages and Databricks.",
        vi_prompt="Top trạm năng lượng tốt nhất và dự báo ngày mai là gì?",
        en_prompt="What are the top energy facilities and tomorrow forecast?",
        expected_topic="energy_performance",
        parity_paths=(
            "top_facilities[0].facility",
            "top_facilities[0].energy_mwh",
            "tomorrow_forecast_mwh",
        ),
        databricks_validator="energy_performance",
    ),
    BilingualPromptCase(
        case_id="ml_model_transparency",
        description="Model quality response must expose consistent ML metrics in both languages.",
        vi_prompt="Mô hình dự báo hiện tại có tốt không, R-squared bao nhiêu so với bản trước?",
        en_prompt="How good is the current forecast model and what is the R-squared delta versus the previous one?",
        expected_topic="ml_model",
        parity_paths=(
            "model_name",
            "model_version",
            "comparison.current_r_squared",
            "comparison.previous_r_squared",
            "comparison.delta_r_squared",
            "is_fallback_model",
        ),
        databricks_validator="ml_model",
    ),
    BilingualPromptCase(
        case_id="forecast_72h_alignment",
        description="72-hour forecast should return aligned daily values in Vietnamese and English.",
        vi_prompt="Cho tôi dự báo sản lượng trong 72 giờ tới.",
        en_prompt="Show me the expected energy production for the next 72 hours.",
        expected_topic="forecast_72h",
        parity_paths=(
            "daily_forecast[0].date",
            "daily_forecast[0].expected_mwh",
            "daily_forecast[1].expected_mwh",
            "daily_forecast[2].expected_mwh",
        ),
        databricks_validator="forecast_72h",
    ),
    BilingualPromptCase(
        case_id="data_quality_alerts",
        description="Data quality issue detection should be language-invariant and Databricks-verified.",
        vi_prompt="Có trạm nào chất lượng dữ liệu thấp hoặc có cảnh báo không?",
        en_prompt="Are there any low-score facilities or data quality alerts?",
        expected_topic="data_quality_issues",
        parity_paths=(
            "low_score_facilities",
        ),
        databricks_validator="data_quality_issues",
    ),
)


LONG_CONVERSATION_CASE = LongConversationCase(
    case_id="long_context_bilingual",
    description=(
        "Multi-turn context stress test with cross-topic switching, pronoun/coreference usage, "
        "and bilingual content consistency checks."
    ),
    turns=(
        ConversationTurnCase(
            vi_prompt="Trạm nào có công suất lớn nhất hiện tại?",
            en_prompt="Which station currently has the largest capacity?",
            expected_topic="facility_info",
            parity_paths=("facility_count",),
            databricks_validator="facility_info",
        ),
        ConversationTurnCase(
            vi_prompt="Múi giờ của trạm đó là gì?",
            en_prompt="What is that station's timezone?",
            expected_topic="facility_info",
            parity_paths=("facility_count",),
            requires_anchor_station=True,
        ),
        ConversationTurnCase(
            vi_prompt="Nhắc lại công suất của chính trạm đó.",
            en_prompt="Repeat the capacity of that exact station.",
            expected_topic="facility_info",
            parity_paths=("facility_count",),
            requires_anchor_station=True,
        ),
        ConversationTurnCase(
            vi_prompt="Bây giờ cho tôi tổng quan hệ thống hiện tại.",
            en_prompt="Now give me the current system overview.",
            expected_topic="system_overview",
            parity_paths=("production_output_mwh", "facility_count", "r_squared"),
            databricks_validator="system_overview",
        ),
        ConversationTurnCase(
            vi_prompt="Mô hình dự báo hiện tại tốt không, cho tôi R-squared và delta.",
            en_prompt="Is the current forecasting model good? Give me the R-squared and delta.",
            expected_topic="ml_model",
            parity_paths=("comparison.current_r_squared", "comparison.delta_r_squared"),
            databricks_validator="ml_model",
        ),
        ConversationTurnCase(
            vi_prompt="Cho tôi dự báo 72 giờ tới.",
            en_prompt="Show me the 72-hour forecast.",
            expected_topic="forecast_72h",
            parity_paths=("daily_forecast[0].expected_mwh",),
            databricks_validator="forecast_72h",
        ),
        ConversationTurnCase(
            vi_prompt="Có cảnh báo chất lượng dữ liệu nào đáng chú ý không?",
            en_prompt="Are there any notable data quality alerts?",
            expected_topic="data_quality_issues",
            parity_paths=("low_score_facilities",),
            databricks_validator="data_quality_issues",
        ),
        ConversationTurnCase(
            vi_prompt="Tóm tắt lại số cơ sở và mô hình tốt nhất ở trên.",
            en_prompt="Summarize the facility count and best model from above.",
            expected_topic=None,
            parity_paths=(),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run long-horizon bilingual Solar AI Chat accuracy tests with context stress and Databricks cross-check."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8003", help="Backend base URL.")
    parser.add_argument("--username", default="admin", help="Login username.")
    parser.add_argument("--password", default="admin123", help="Login password.")
    parser.add_argument("--role", default="admin", help="Role value for new sessions.")
    parser.add_argument("--timeout-seconds", type=float, default=240.0, help="HTTP request timeout.")
    parser.add_argument(
        "--output-dir",
        default="test_reports/solar_chat_accuracy",
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--max-single-cases",
        type=int,
        default=0,
        help="If >0, run only the first N single-turn cases.",
    )
    parser.add_argument(
        "--skip-databricks",
        action="store_true",
        help="Skip Databricks cross-validation queries.",
    )
    parser.add_argument(
        "--skip-long-conversation",
        action="store_true",
        help="Skip long multi-turn context scenario.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with non-zero code when any assertion fails.",
    )
    parser.add_argument(
        "--print-answer-preview",
        action="store_true",
        help="Print assistant answer previews while running.",
    )
    return parser.parse_args()


def _build_timeout(total_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=total_timeout)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short(text: Any, limit: int = 240) -> str:
    raw = str(text or "")
    compact = " ".join(raw.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_string(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value.keys())}
    return value


def _extract_path(payload: dict[str, Any], path: str) -> tuple[Any | None, str | None]:
    current: Any = payload
    for raw_segment in path.split("."):
        if not raw_segment:
            return None, "empty path segment"

        segment = raw_segment
        index: int | None = None
        if "[" in raw_segment and raw_segment.endswith("]"):
            left, right = raw_segment[:-1].split("[", 1)
            segment = left
            if not right.isdigit():
                return None, f"invalid list index in '{raw_segment}'"
            index = int(right)

        if not isinstance(current, dict):
            return None, f"segment '{segment}' expected object but found {type(current).__name__}"
        if segment not in current:
            return None, f"segment '{segment}' not found"

        current = current.get(segment)
        if index is not None:
            if not isinstance(current, list):
                return None, f"segment '{segment}' is not a list"
            if index >= len(current):
                return None, f"segment '{segment}' index {index} out of range"
            current = current[index]

    return current, None


def _assertion(name: str, passed: bool, detail: str) -> AssertionResult:
    return AssertionResult(name=name, passed=passed, detail=detail)


def _assert_equal(name: str, actual: Any, expected: Any) -> AssertionResult:
    passed = _normalize_value(actual) == _normalize_value(expected)
    return _assertion(name, passed, f"actual={actual!r}, expected={expected!r}")


def _assert_close(name: str, actual: Any, expected: Any, tolerance: float) -> AssertionResult:
    actual_f = _safe_float(actual, default=math.nan)
    expected_f = _safe_float(expected, default=math.nan)
    if math.isnan(actual_f) or math.isnan(expected_f):
        return _assertion(
            name,
            False,
            f"non-numeric comparison actual={actual!r}, expected={expected!r}",
        )
    passed = abs(actual_f - expected_f) <= tolerance
    return _assertion(
        name,
        passed,
        f"actual={round(actual_f, 6)}, expected={round(expected_f, 6)}, tolerance={tolerance}",
    )


def _prefix_assertions(prefix: str, assertions: list[AssertionResult]) -> list[AssertionResult]:
    return [
        AssertionResult(name=f"{prefix}.{item.name}", passed=item.passed, detail=item.detail)
        for item in assertions
    ]


def login(client: httpx.Client, username: str, password: str) -> None:
    response = client.post(
        "/auth/login",
        data={
            "username": username,
            "password": password,
            "next": "/dashboard",
        },
        follow_redirects=False,
    )
    if response.status_code not in (302, 303):
        raise RuntimeError(f"Login failed: status={response.status_code}, body={response.text[:320]}")
    if not client.cookies:
        raise RuntimeError("Login succeeded but no cookie was set.")


def create_session(client: httpx.Client, role: str, title: str) -> str:
    response = client.post("/solar-ai-chat/sessions", json={"role": role, "title": title})
    if response.status_code >= 400:
        raise RuntimeError(
            f"Create session failed: status={response.status_code}, body={response.text[:320]}"
        )

    payload = response.json()
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise RuntimeError("Create session response missing session_id.")
    return session_id


def query_chat(
    client: httpx.Client,
    role: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        "/solar-ai-chat/query",
        json={
            "role": role,
            "session_id": session_id,
            "message": message,
        },
    )
    roundtrip_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Query failed: status={response.status_code}, body={response.text[:420]}"
        )

    payload = response.json()
    payload["_roundtrip_ms"] = roundtrip_ms
    return payload


class DatabricksVerifier:
    def __init__(self, disabled: bool = False) -> None:
        self._disabled = disabled
        self._settings = get_solar_chat_settings()
        self._host = (self._settings.databricks_host or "").strip()
        self._token = (self._settings.databricks_token or "").strip()
        self._http_path = (self._settings.resolved_databricks_http_path or "").strip()
        self.lookback_days = max(1, int(getattr(self._settings, "analytics_lookback_days", 30)))

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def configured(self) -> bool:
        if self._disabled:
            return False
        return bool(self._host and self._token and self._http_path)

    def query(self, sql: str) -> list[dict[str, Any]]:
        if self._disabled:
            raise RuntimeError("Databricks verifier is disabled.")
        if not self.configured:
            raise RuntimeError("Databricks settings are not configured.")

        from databricks import sql as databricks_sql

        parsed = urlparse(self._host)
        server_hostname = parsed.netloc if parsed.scheme else self._host
        if not server_hostname:
            raise RuntimeError("Invalid DATABRICKS_HOST setting.")

        conn = databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=self._http_path,
            access_token=self._token,
            catalog=(self._settings.uc_catalog or "pv").strip().lower() or "pv",
            schema=(self._settings.uc_silver_schema or "silver").strip().lower() or "silver",
        )
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()


def _derive_timezone_from_coordinates(latitude: float, longitude: float) -> tuple[str, str]:
    if -45.0 <= latitude <= -10.0 and 112.0 <= longitude <= 154.5:
        if longitude < 129.0:
            return "Australia/Western", "UTC+08:00"
        if longitude < 141.0:
            return "Australia/Central", "UTC+09:30"
        return "Australia/Eastern", "UTC+10:00"

    approx_hours = float(round(longitude / 15.0))
    sign = "+" if approx_hours >= 0 else "-"
    abs_hours = abs(approx_hours)
    whole_hours = int(abs_hours)
    minutes = int(round((abs_hours - whole_hours) * 60))
    return "UTC (approx)", f"UTC{sign}{whole_hours:02d}:{minutes:02d}"


def validate_facility_info(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    rows = verifier.query(
        "SELECT facility_name, location_lat, location_lng, total_capacity_mw"
        " FROM gold.dim_facility"
        " WHERE is_current = true"
        " ORDER BY facility_name"
    )

    assertions: list[AssertionResult] = []
    facilities = metrics.get("facilities", []) if isinstance(metrics, dict) else []
    actual_count = _safe_int(metrics.get("facility_count", len(facilities)))
    expected_count = len(rows)
    assertions.append(_assert_equal("facility_count", actual_count, expected_count))

    if rows and facilities:
        expected_top = max(rows, key=lambda row: _safe_float(row.get("total_capacity_mw")))
        actual_top = max(
            facilities,
            key=lambda row: _safe_float(row.get("total_capacity_mw")),
        )

        assertions.append(
            _assert_equal(
                "top_station_name",
                _normalize_string(actual_top.get("facility_name")),
                _normalize_string(expected_top.get("facility_name")),
            )
        )
        assertions.append(
            _assert_close(
                "top_station_capacity_mw",
                actual_top.get("total_capacity_mw"),
                expected_top.get("total_capacity_mw"),
                tolerance=0.01,
            )
        )

        expected_tz_name, expected_tz_offset = _derive_timezone_from_coordinates(
            _safe_float(expected_top.get("location_lat")),
            _safe_float(expected_top.get("location_lng")),
        )
        assertions.append(
            _assert_equal("top_station_timezone_name", actual_top.get("timezone_name"), expected_tz_name)
        )
        assertions.append(
            _assert_equal(
                "top_station_timezone_offset",
                actual_top.get("timezone_utc_offset"),
                expected_tz_offset,
            )
        )

    return assertions


def validate_system_overview(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    lookback_days = verifier.lookback_days
    agg = verifier.query(
        "SELECT COALESCE(SUM(energy_mwh), 0) AS total_mwh,"
        "       COALESCE(AVG(completeness_pct), 0) AS avg_quality"
        " FROM gold.fact_energy"
        f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
    )

    fac = verifier.query(
        "SELECT COUNT(*) AS cnt"
        " FROM gold.dim_facility"
        " WHERE is_current = true"
    )

    r_rows = verifier.query(
        "SELECT r2"
        " FROM gold.model_monitoring_daily"
        " WHERE facility_id = 'ALL' AND r2 IS NOT NULL"
        " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
        " LIMIT 1"
    )

    if not r_rows:
        r_rows = verifier.query(
            "SELECT AVG(r2) AS r2"
            " FROM gold.model_monitoring_daily"
            " WHERE facility_id <> 'ALL'"
            "   AND r2 IS NOT NULL"
            "   AND CAST(eval_date AS DATE) = ("
            "       SELECT MAX(CAST(eval_date AS DATE))"
            "       FROM gold.model_monitoring_daily"
            "       WHERE facility_id <> 'ALL'"
            "   )"
        )

    expected_total_mwh = round(_safe_float(agg[0].get("total_mwh") if agg else 0.0), 2)
    expected_quality = round(_safe_float(agg[0].get("avg_quality") if agg else 0.0), 2)
    expected_facility_count = _safe_int(fac[0].get("cnt") if fac else 0)
    expected_r2 = round(_safe_float(r_rows[0].get("r2") if r_rows else 0.0), 4)

    assertions: list[AssertionResult] = [
        _assert_close("production_output_mwh", metrics.get("production_output_mwh"), expected_total_mwh, 0.05),
        _assert_close("data_quality_score", metrics.get("data_quality_score"), expected_quality, 0.05),
        _assert_equal("facility_count", _safe_int(metrics.get("facility_count")), expected_facility_count),
        _assert_close("r_squared", metrics.get("r_squared"), expected_r2, 0.0001),
        _assert_equal("window_days", _safe_int(metrics.get("window_days")), lookback_days),
    ]
    return assertions


def validate_energy_performance(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    lookback_days = verifier.lookback_days
    top_rows = verifier.query(
        "SELECT COALESCE(d.facility_name, f.facility_id) AS facility,"
        "       SUM(f.energy_mwh) AS total_mwh"
        " FROM gold.fact_energy f"
        " LEFT JOIN gold.dim_facility d"
        "   ON f.facility_id = d.facility_id AND d.is_current = true"
        f" WHERE f.date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
        " GROUP BY COALESCE(d.facility_name, f.facility_id)"
        " ORDER BY total_mwh DESC LIMIT 1"
    )

    tomorrow_rows = verifier.query(
        "SELECT SUM(predicted_energy_mwh_daily) AS tomorrow_mwh"
        " FROM gold.forecast_daily"
        " WHERE CAST(forecast_date AS DATE) = date_add(current_date(), 1)"
    )

    assertions: list[AssertionResult] = []
    top_facilities = metrics.get("top_facilities", []) if isinstance(metrics, dict) else []

    if top_rows and top_facilities:
        expected_top = top_rows[0]
        actual_top = top_facilities[0]
        assertions.append(
            _assert_equal(
                "top_facility_name",
                _normalize_string(actual_top.get("facility")),
                _normalize_string(expected_top.get("facility")),
            )
        )
        assertions.append(
            _assert_close(
                "top_facility_energy_mwh",
                actual_top.get("energy_mwh"),
                round(_safe_float(expected_top.get("total_mwh")), 2),
                tolerance=0.1,
            )
        )

    tomorrow_value = tomorrow_rows[0].get("tomorrow_mwh") if tomorrow_rows else None
    if tomorrow_value is not None:
        assertions.append(
            _assert_close(
                "tomorrow_forecast_mwh",
                metrics.get("tomorrow_forecast_mwh"),
                round(_safe_float(tomorrow_value), 2),
                tolerance=0.1,
            )
        )

    assertions.append(_assert_equal("window_days", _safe_int(metrics.get("window_days")), lookback_days))
    return assertions


def validate_ml_model(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    latest_rows = verifier.query(
        "SELECT model_name, model_version, approach, eval_date,"
        "       r2, skill_score, nrmse_pct"
        " FROM gold.model_monitoring_daily"
        " WHERE facility_id = 'ALL'"
        " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
        " LIMIT 1"
    )

    if not latest_rows:
        return [_assertion("latest_row_exists", False, "gold.model_monitoring_daily has no facility_id='ALL' rows")]

    latest = latest_rows[0]
    comparison = metrics.get("comparison", {}) if isinstance(metrics, dict) else {}
    assertions: list[AssertionResult] = [
        _assert_equal("model_name", metrics.get("model_name"), latest.get("model_name")),
        _assert_equal("model_version", metrics.get("model_version"), latest.get("model_version")),
        _assert_close("current_r_squared", comparison.get("current_r_squared"), latest.get("r2"), tolerance=0.0001),
        _assert_close("skill_score", comparison.get("skill_score"), latest.get("skill_score"), tolerance=0.0001),
        _assert_close("nrmse_pct", comparison.get("nrmse_pct"), latest.get("nrmse_pct"), tolerance=0.001),
    ]

    is_fallback = bool(metrics.get("is_fallback_model", False))
    if is_fallback:
        non_fallback_rows = verifier.query(
            "SELECT model_name, model_version, r2"
            " FROM gold.model_monitoring_daily"
            " WHERE facility_id = 'ALL'"
            "   AND r2 IS NOT NULL"
            "   AND LOWER(model_version) NOT LIKE 'fallback%'"
            "   AND LOWER(approach) NOT LIKE 'fallback%'"
            " ORDER BY CAST(eval_date AS DATE) DESC, generated_at_utc DESC"
            " LIMIT 1"
        )
        if non_fallback_rows:
            expected_non = non_fallback_rows[0]
            expected_label = f"{expected_non.get('model_name')}:{expected_non.get('model_version')}"
            assertions.append(
                _assert_close(
                    "latest_non_fallback_r_squared",
                    comparison.get("latest_non_fallback_r_squared"),
                    expected_non.get("r2"),
                    tolerance=0.0001,
                )
            )
            assertions.append(
                _assert_equal(
                    "latest_non_fallback_model",
                    comparison.get("latest_non_fallback_model"),
                    expected_label,
                )
            )

    return assertions


def validate_forecast_72h(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    rows = verifier.query(
        "SELECT CAST(forecast_date AS DATE) AS day,"
        "       SUM(predicted_energy_mwh_daily) AS expected_mwh"
        " FROM gold.forecast_daily"
        " WHERE CAST(forecast_date AS DATE) BETWEEN current_date() AND date_add(current_date(), 2)"
        " GROUP BY CAST(forecast_date AS DATE)"
        " ORDER BY day ASC"
    )

    if len(rows) < 3:
        latest_rows = verifier.query(
            "SELECT CAST(forecast_date AS DATE) AS day,"
            "       SUM(predicted_energy_mwh_daily) AS expected_mwh"
            " FROM gold.forecast_daily"
            " GROUP BY CAST(forecast_date AS DATE)"
            " ORDER BY day DESC LIMIT 3"
        )
        rows = list(reversed(latest_rows))

    actual_daily = metrics.get("daily_forecast", []) if isinstance(metrics, dict) else []
    assertions: list[AssertionResult] = [
        _assert_equal("daily_forecast_count", len(actual_daily), min(3, len(rows))),
    ]

    compare_len = min(len(rows), len(actual_daily), 3)
    for index in range(compare_len):
        expected_row = rows[index]
        actual_row = actual_daily[index]
        expected_day = str(expected_row.get("day"))
        expected_mwh = round(_safe_float(expected_row.get("expected_mwh")), 2)
        assertions.append(
            _assert_equal(f"daily_forecast[{index}].date", actual_row.get("date"), expected_day)
        )
        assertions.append(
            _assert_close(
                f"daily_forecast[{index}].expected_mwh",
                actual_row.get("expected_mwh"),
                expected_mwh,
                tolerance=0.1,
            )
        )

    return assertions


def validate_data_quality_issues(metrics: dict[str, Any], verifier: DatabricksVerifier) -> list[AssertionResult]:
    lookback_days = verifier.lookback_days
    rows = verifier.query(
        "SELECT COALESCE(facility_name, facility_id) AS facility,"
        "       AVG(completeness_pct) AS avg_score"
        " FROM silver.energy_readings"
        f" WHERE date_hour >= current_timestamp() - INTERVAL {lookback_days} DAYS"
        " GROUP BY COALESCE(facility_name, facility_id)"
        " HAVING AVG(completeness_pct) < 95"
        " ORDER BY avg_score ASC LIMIT 5"
    )

    actual_rows = metrics.get("low_score_facilities", []) if isinstance(metrics, dict) else []
    assertions: list[AssertionResult] = [
        _assert_equal("low_score_facilities_count", len(actual_rows), len(rows)),
    ]

    if rows and actual_rows:
        expected_first = rows[0]
        actual_first = actual_rows[0]
        assertions.append(
            _assert_equal(
                "first_low_score_facility",
                _normalize_string(actual_first.get("facility")),
                _normalize_string(expected_first.get("facility")),
            )
        )
        assertions.append(
            _assert_close(
                "first_low_score_quality_score",
                actual_first.get("quality_score"),
                round(_safe_float(expected_first.get("avg_score")), 2),
                tolerance=0.1,
            )
        )
    if not rows:
        summary_text = _normalize_string(metrics.get("summary", ""))
        assertions.append(
            _assertion(
                "no_issues_summary",
                "all facilities have quality score" in summary_text,
                f"summary={metrics.get('summary', '')!r}",
            )
        )

    return assertions


DB_VALIDATORS: dict[str, Callable[[dict[str, Any], DatabricksVerifier], list[AssertionResult]]] = {
    "facility_info": validate_facility_info,
    "system_overview": validate_system_overview,
    "energy_performance": validate_energy_performance,
    "ml_model": validate_ml_model,
    "forecast_72h": validate_forecast_72h,
    "data_quality_issues": validate_data_quality_issues,
}


def run_databricks_validator(
    validator_name: str | None,
    metrics: dict[str, Any],
    verifier: DatabricksVerifier,
) -> list[AssertionResult]:
    if not validator_name:
        return []

    if verifier.disabled:
        return [_assertion(f"{validator_name}.skipped", True, "Skipped by --skip-databricks")]

    if not verifier.configured:
        return [
            _assertion(
                f"{validator_name}.configured",
                False,
                "Databricks settings are missing. Provide DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_SQL_HTTP_PATH/WAREHOUSE_ID.",
            )
        ]

    validator = DB_VALIDATORS.get(validator_name)
    if validator is None:
        return [_assertion(f"{validator_name}.available", False, "Validator is not implemented.")]

    try:
        return validator(metrics, verifier)
    except Exception as exc:
        return [_assertion(f"{validator_name}.query", False, f"Databricks validation error: {exc}")]


def compare_parity_paths(
    vi_metrics: dict[str, Any],
    en_metrics: dict[str, Any],
    paths: tuple[str, ...],
) -> list[AssertionResult]:
    assertions: list[AssertionResult] = []
    for path in paths:
        vi_value, vi_error = _extract_path(vi_metrics, path)
        en_value, en_error = _extract_path(en_metrics, path)
        if vi_error:
            assertions.append(_assertion(f"parity.{path}", False, f"vi_missing={vi_error}"))
            continue
        if en_error:
            assertions.append(_assertion(f"parity.{path}", False, f"en_missing={en_error}"))
            continue

        passed = _normalize_value(vi_value) == _normalize_value(en_value)
        assertions.append(
            _assertion(
                f"parity.{path}",
                passed,
                f"vi={vi_value!r}, en={en_value!r}",
            )
        )
    return assertions


def _top_station_name(metrics: dict[str, Any]) -> str:
    facilities = metrics.get("facilities", []) if isinstance(metrics, dict) else []
    if not facilities:
        return ""
    top = max(facilities, key=lambda row: _safe_float(row.get("total_capacity_mw")))
    return str(top.get("facility_name", "")).strip()


def run_single_turn_case(
    client: httpx.Client,
    role: str,
    case: BilingualPromptCase,
    verifier: DatabricksVerifier,
    print_answer_preview: bool,
) -> dict[str, Any]:
    case_assertions: list[AssertionResult] = []
    vi_session = create_session(client, role, f"suite-{case.case_id}-vi")
    en_session = create_session(client, role, f"suite-{case.case_id}-en")

    vi_response = query_chat(client, role, vi_session, case.vi_prompt)
    en_response = query_chat(client, role, en_session, case.en_prompt)

    if print_answer_preview:
        print(f"  vi_answer={_short(vi_response.get('answer', ''), 180)}")
        print(f"  en_answer={_short(en_response.get('answer', ''), 180)}")

    case_assertions.append(
        _assert_equal("vi.topic", vi_response.get("topic"), case.expected_topic)
    )
    case_assertions.append(
        _assert_equal("en.topic", en_response.get("topic"), case.expected_topic)
    )
    case_assertions.append(
        _assert_equal("parity.topic", vi_response.get("topic"), en_response.get("topic"))
    )

    vi_metrics = vi_response.get("key_metrics") if isinstance(vi_response.get("key_metrics"), dict) else {}
    en_metrics = en_response.get("key_metrics") if isinstance(en_response.get("key_metrics"), dict) else {}
    case_assertions.extend(compare_parity_paths(vi_metrics, en_metrics, case.parity_paths))

    case_assertions.extend(
        _prefix_assertions(
            "vi.db",
            run_databricks_validator(case.databricks_validator, vi_metrics, verifier),
        )
    )
    case_assertions.extend(
        _prefix_assertions(
            "en.db",
            run_databricks_validator(case.databricks_validator, en_metrics, verifier),
        )
    )

    passed = all(item.passed for item in case_assertions)

    return {
        "case_id": case.case_id,
        "case_type": "single_turn",
        "description": case.description,
        "passed": passed,
        "assertions": [item.as_dict() for item in case_assertions],
        "vi": {
            "prompt": case.vi_prompt,
            "topic": vi_response.get("topic"),
            "latency_ms": vi_response.get("latency_ms"),
            "roundtrip_ms": vi_response.get("_roundtrip_ms"),
            "fallback_used": bool(vi_response.get("fallback_used", False)),
            "warning_message": str(vi_response.get("warning_message", "") or ""),
            "answer_preview": _short(vi_response.get("answer", ""), 320),
            "key_metrics": vi_metrics,
        },
        "en": {
            "prompt": case.en_prompt,
            "topic": en_response.get("topic"),
            "latency_ms": en_response.get("latency_ms"),
            "roundtrip_ms": en_response.get("_roundtrip_ms"),
            "fallback_used": bool(en_response.get("fallback_used", False)),
            "warning_message": str(en_response.get("warning_message", "") or ""),
            "answer_preview": _short(en_response.get("answer", ""), 320),
            "key_metrics": en_metrics,
        },
    }


def run_long_conversation_case(
    client: httpx.Client,
    role: str,
    case: LongConversationCase,
    verifier: DatabricksVerifier,
    print_answer_preview: bool,
) -> dict[str, Any]:
    case_assertions: list[AssertionResult] = []
    turn_results: list[dict[str, Any]] = []

    vi_session = create_session(client, role, f"suite-{case.case_id}-vi")
    en_session = create_session(client, role, f"suite-{case.case_id}-en")

    anchor_station_vi = ""
    anchor_station_en = ""

    for index, turn in enumerate(case.turns, start=1):
        vi_response = query_chat(client, role, vi_session, turn.vi_prompt)
        en_response = query_chat(client, role, en_session, turn.en_prompt)

        vi_metrics = vi_response.get("key_metrics") if isinstance(vi_response.get("key_metrics"), dict) else {}
        en_metrics = en_response.get("key_metrics") if isinstance(en_response.get("key_metrics"), dict) else {}

        if index == 1:
            anchor_station_vi = _top_station_name(vi_metrics)
            anchor_station_en = _top_station_name(en_metrics)

        if turn.expected_topic:
            case_assertions.append(
                _assert_equal(f"turn{index}.vi.topic", vi_response.get("topic"), turn.expected_topic)
            )
            case_assertions.append(
                _assert_equal(f"turn{index}.en.topic", en_response.get("topic"), turn.expected_topic)
            )
        case_assertions.append(
            _assert_equal(f"turn{index}.parity.topic", vi_response.get("topic"), en_response.get("topic"))
        )

        case_assertions.extend(
            _prefix_assertions(
                f"turn{index}",
                compare_parity_paths(vi_metrics, en_metrics, turn.parity_paths),
            )
        )

        case_assertions.extend(
            _prefix_assertions(
                f"turn{index}.vi.db",
                run_databricks_validator(turn.databricks_validator, vi_metrics, verifier),
            )
        )
        case_assertions.extend(
            _prefix_assertions(
                f"turn{index}.en.db",
                run_databricks_validator(turn.databricks_validator, en_metrics, verifier),
            )
        )

        if turn.requires_anchor_station:
            vi_answer = _normalize_string(vi_response.get("answer", ""))
            en_answer = _normalize_string(en_response.get("answer", ""))

            if anchor_station_vi:
                case_assertions.append(
                    _assertion(
                        f"turn{index}.vi.context_anchor",
                        _normalize_string(anchor_station_vi) in vi_answer,
                        f"expected station reference '{anchor_station_vi}' in Vietnamese answer",
                    )
                )
            else:
                case_assertions.append(
                    _assertion(
                        f"turn{index}.vi.context_anchor",
                        False,
                        "Anchor station is empty in Vietnamese session.",
                    )
                )

            if anchor_station_en:
                case_assertions.append(
                    _assertion(
                        f"turn{index}.en.context_anchor",
                        _normalize_string(anchor_station_en) in en_answer,
                        f"expected station reference '{anchor_station_en}' in English answer",
                    )
                )
            else:
                case_assertions.append(
                    _assertion(
                        f"turn{index}.en.context_anchor",
                        False,
                        "Anchor station is empty in English session.",
                    )
                )

        if print_answer_preview:
            print(f"  turn={index} vi_answer={_short(vi_response.get('answer', ''), 140)}")
            print(f"  turn={index} en_answer={_short(en_response.get('answer', ''), 140)}")

        turn_results.append(
            {
                "turn_index": index,
                "vi": {
                    "prompt": turn.vi_prompt,
                    "topic": vi_response.get("topic"),
                    "latency_ms": vi_response.get("latency_ms"),
                    "roundtrip_ms": vi_response.get("_roundtrip_ms"),
                    "warning_message": str(vi_response.get("warning_message", "") or ""),
                    "fallback_used": bool(vi_response.get("fallback_used", False)),
                    "answer_preview": _short(vi_response.get("answer", ""), 240),
                    "key_metrics": vi_metrics,
                },
                "en": {
                    "prompt": turn.en_prompt,
                    "topic": en_response.get("topic"),
                    "latency_ms": en_response.get("latency_ms"),
                    "roundtrip_ms": en_response.get("_roundtrip_ms"),
                    "warning_message": str(en_response.get("warning_message", "") or ""),
                    "fallback_used": bool(en_response.get("fallback_used", False)),
                    "answer_preview": _short(en_response.get("answer", ""), 240),
                    "key_metrics": en_metrics,
                },
            }
        )

    passed = all(item.passed for item in case_assertions)
    return {
        "case_id": case.case_id,
        "case_type": "long_conversation",
        "description": case.description,
        "passed": passed,
        "assertions": [item.as_dict() for item in case_assertions],
        "turns": turn_results,
        "anchors": {
            "vi": anchor_station_vi,
            "en": anchor_station_en,
        },
    }


def _case_assertion_counts(case_result: dict[str, Any]) -> tuple[int, int]:
    assertions = case_result.get("assertions", [])
    total = len(assertions)
    passed = sum(1 for item in assertions if bool(item.get("passed", False)))
    return passed, total


def build_markdown_report(
    report_data: dict[str, Any],
) -> str:
    lines: list[str] = []
    run_info = report_data.get("run_info", {})
    cases: list[dict[str, Any]] = report_data.get("cases", [])

    lines.append("# Solar AI Chat Long-Horizon Accuracy Report")
    lines.append("")
    lines.append(f"- Generated UTC: {run_info.get('generated_utc', '')}")
    lines.append(f"- Base URL: {run_info.get('base_url', '')}")
    lines.append(f"- Databricks validation enabled: {run_info.get('databricks_enabled', False)}")
    lines.append(f"- Single-turn cases: {run_info.get('single_turn_cases', 0)}")
    lines.append(f"- Long conversation cases: {run_info.get('long_conversation_cases', 0)}")
    lines.append("")

    summary = report_data.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Case pass: {summary.get('passed_cases', 0)}/{summary.get('total_cases', 0)}")
    lines.append(
        f"- Assertion pass: {summary.get('passed_assertions', 0)}/{summary.get('total_assertions', 0)}"
    )
    lines.append(f"- Average latency (ms): {summary.get('avg_latency_ms', 0)}")
    lines.append("")

    lines.append("## Case Matrix")
    lines.append("")
    lines.append("| Case ID | Type | Result | Assertions (pass/total) |")
    lines.append("|---|---|---|---|")
    for case in cases:
        passed, total = _case_assertion_counts(case)
        result_text = "PASS" if case.get("passed") else "FAIL"
        lines.append(
            f"| {case.get('case_id')} | {case.get('case_type')} | {result_text} | {passed}/{total} |"
        )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for case in cases:
        passed, total = _case_assertion_counts(case)
        result_text = "PASS" if case.get("passed") else "FAIL"
        lines.append(f"### {case.get('case_id')} ({result_text})")
        lines.append("")
        lines.append(f"- Type: {case.get('case_type')}")
        lines.append(f"- Description: {case.get('description')}")
        lines.append(f"- Assertions: {passed}/{total}")

        failed_assertions = [item for item in case.get("assertions", []) if not item.get("passed", False)]
        if failed_assertions:
            lines.append("- Failed assertions:")
            for item in failed_assertions:
                lines.append(f"  - {item.get('name')}: {item.get('detail')}")
        else:
            lines.append("- Failed assertions: none")

        if case.get("case_type") == "single_turn":
            vi = case.get("vi", {})
            en = case.get("en", {})
            lines.append(f"- VI prompt: {vi.get('prompt', '')}")
            lines.append(f"- EN prompt: {en.get('prompt', '')}")
            lines.append(f"- VI topic: {vi.get('topic', '')}; EN topic: {en.get('topic', '')}")
            lines.append(
                f"- VI latency/roundtrip: {vi.get('latency_ms', -1)}/{vi.get('roundtrip_ms', -1)} ms"
            )
            lines.append(
                f"- EN latency/roundtrip: {en.get('latency_ms', -1)}/{en.get('roundtrip_ms', -1)} ms"
            )
            lines.append(f"- VI warning: {vi.get('warning_message', '') or 'none'}")
            lines.append(f"- EN warning: {en.get('warning_message', '') or 'none'}")
            lines.append(f"- VI answer preview: {vi.get('answer_preview', '')}")
            lines.append(f"- EN answer preview: {en.get('answer_preview', '')}")

        if case.get("case_type") == "long_conversation":
            anchors = case.get("anchors", {})
            lines.append(f"- Anchor station (VI): {anchors.get('vi', '') or 'n/a'}")
            lines.append(f"- Anchor station (EN): {anchors.get('en', '') or 'n/a'}")
            turns = case.get("turns", [])
            lines.append(f"- Turn count: {len(turns)}")
            for turn in turns:
                turn_index = turn.get("turn_index")
                vi = turn.get("vi", {})
                en = turn.get("en", {})
                lines.append(
                    f"  - Turn {turn_index}: VI topic={vi.get('topic', '')}, EN topic={en.get('topic', '')}, "
                    f"VI latency={vi.get('latency_ms', -1)} ms, EN latency={en.get('latency_ms', -1)} ms"
                )
                lines.append(f"    VI prompt: {vi.get('prompt', '')}")
                lines.append(f"    EN prompt: {en.get('prompt', '')}")
                lines.append(f"    VI answer preview: {vi.get('answer_preview', '')}")
                lines.append(f"    EN answer preview: {en.get('answer_preview', '')}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_reports(report_data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"solar_chat_accuracy_{stamp}.json"
    md_path = output_dir / f"solar_chat_accuracy_{stamp}.md"
    latest_json = output_dir / "solar_chat_accuracy_latest.json"
    latest_md = output_dir / "solar_chat_accuracy_latest.md"

    markdown_text = build_markdown_report(report_data)

    json_payload = json.dumps(report_data, ensure_ascii=False, indent=2)
    json_path.write_text(json_payload, encoding="utf-8")
    md_path.write_text(markdown_text, encoding="utf-8")
    latest_json.write_text(json_payload, encoding="utf-8")
    latest_md.write_text(markdown_text, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def _collect_latency(values: list[dict[str, Any]]) -> list[int]:
    latencies: list[int] = []
    for case in values:
        if case.get("case_type") == "single_turn":
            vi_latency = _safe_int(case.get("vi", {}).get("latency_ms"), -1)
            en_latency = _safe_int(case.get("en", {}).get("latency_ms"), -1)
            if vi_latency >= 0:
                latencies.append(vi_latency)
            if en_latency >= 0:
                latencies.append(en_latency)
        elif case.get("case_type") == "long_conversation":
            for turn in case.get("turns", []):
                vi_latency = _safe_int(turn.get("vi", {}).get("latency_ms"), -1)
                en_latency = _safe_int(turn.get("en", {}).get("latency_ms"), -1)
                if vi_latency >= 0:
                    latencies.append(vi_latency)
                if en_latency >= 0:
                    latencies.append(en_latency)
    return latencies


def main() -> int:
    args = parse_args()

    single_cases = list(SINGLE_TURN_CASES)
    if args.max_single_cases and args.max_single_cases > 0:
        single_cases = single_cases[: args.max_single_cases]

    long_cases: list[LongConversationCase] = []
    if not args.skip_long_conversation:
        long_cases.append(LONG_CONVERSATION_CASE)

    verifier = DatabricksVerifier(disabled=bool(args.skip_databricks))

    if not args.skip_databricks and not verifier.configured:
        print("[ERROR] Databricks validation requested but Databricks settings are missing.")
        print("        Required: DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_SQL_HTTP_PATH or DATABRICKS_WAREHOUSE_ID.")
        return 2

    timeout = _build_timeout(args.timeout_seconds)
    base_url = args.base_url.rstrip("/")
    all_case_results: list[dict[str, Any]] = []

    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            login(client, args.username, args.password)

            print(f"[INFO] base_url={base_url}")
            print(f"[INFO] databricks_validation={'off' if args.skip_databricks else 'on'}")
            print(f"[INFO] single_turn_cases={len(single_cases)}")
            print(f"[INFO] long_conversation_cases={len(long_cases)}")

            for index, case in enumerate(single_cases, start=1):
                print(f"[RUN] single_turn {index}/{len(single_cases)} case_id={case.case_id}")
                try:
                    case_result = run_single_turn_case(
                        client=client,
                        role=args.role,
                        case=case,
                        verifier=verifier,
                        print_answer_preview=bool(args.print_answer_preview),
                    )
                except Exception as exc:
                    case_result = {
                        "case_id": case.case_id,
                        "case_type": "single_turn",
                        "description": case.description,
                        "passed": False,
                        "assertions": [
                            _assertion("runtime", False, str(exc)).as_dict(),
                        ],
                    }
                all_case_results.append(case_result)
                passed, total = _case_assertion_counts(case_result)
                print(f"[DONE] case_id={case.case_id} passed={case_result.get('passed')} assertions={passed}/{total}")

            for index, case in enumerate(long_cases, start=1):
                print(f"[RUN] long_conversation {index}/{len(long_cases)} case_id={case.case_id}")
                try:
                    case_result = run_long_conversation_case(
                        client=client,
                        role=args.role,
                        case=case,
                        verifier=verifier,
                        print_answer_preview=bool(args.print_answer_preview),
                    )
                except Exception as exc:
                    case_result = {
                        "case_id": case.case_id,
                        "case_type": "long_conversation",
                        "description": case.description,
                        "passed": False,
                        "assertions": [
                            _assertion("runtime", False, str(exc)).as_dict(),
                        ],
                    }
                all_case_results.append(case_result)
                passed, total = _case_assertion_counts(case_result)
                print(f"[DONE] case_id={case.case_id} passed={case_result.get('passed')} assertions={passed}/{total}")

    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 2

    total_cases = len(all_case_results)
    passed_cases = sum(1 for case in all_case_results if bool(case.get("passed", False)))
    total_assertions = sum(len(case.get("assertions", [])) for case in all_case_results)
    passed_assertions = sum(
        1
        for case in all_case_results
        for item in case.get("assertions", [])
        if bool(item.get("passed", False))
    )

    latencies = _collect_latency(all_case_results)
    avg_latency = round(statistics.mean(latencies), 2) if latencies else 0.0

    report_data = {
        "run_info": {
            "generated_utc": _utc_now_iso(),
            "base_url": base_url,
            "databricks_enabled": not args.skip_databricks,
            "single_turn_cases": len(single_cases),
            "long_conversation_cases": len(long_cases),
            "role": args.role,
        },
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "total_assertions": total_assertions,
            "passed_assertions": passed_assertions,
            "failed_assertions": total_assertions - passed_assertions,
            "avg_latency_ms": avg_latency,
        },
        "cases": all_case_results,
    }

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BACKEND_ROOT / output_dir

    report_paths = save_reports(report_data, output_dir)

    print("[REPORT]")
    print(f"  markdown={report_paths['markdown']}")
    print(f"  json={report_paths['json']}")
    print(f"  latest_markdown={report_paths['latest_markdown']}")
    print(f"  latest_json={report_paths['latest_json']}")
    print(
        "[SUMMARY] "
        f"cases={passed_cases}/{total_cases} "
        f"assertions={passed_assertions}/{total_assertions} "
        f"avg_latency_ms={avg_latency}"
    )

    if args.strict_exit and (passed_cases != total_cases or passed_assertions != total_assertions):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
