"""CLI tool to validate the Solar AI Chat Data Visualization pipeline (Feature Group D).

It has two modes:

1. Offline mode (default): exercises ``ChartSpecBuilder`` directly against a
   set of fixture metric payloads mimicking real tool outputs (station daily
   report, hourly report, extreme energy, ML model metrics, RAG chunks). It
   asserts that:

     * A DataTable is built for list-of-dicts metrics with correct columns.
     * A Chart is built with the appropriate chart_type (line for time series,
       bar for categorical comparison).
     * KPI cards are extracted for scalar numeric metrics.

2. Live mode (``--live``): posts a message to the real ``/solar-ai-chat/query``
   and ``/solar-ai-chat/stream`` endpoints, then asserts the response / done
   event contains a matching visualization payload.

Usage examples::

    # Offline fixtures (no backend required)
    python scripts/solar_chat_viz_cli.py

    # Live check against a running backend
    python scripts/solar_chat_viz_cli.py --live \\
        --message "Station daily report for Avonlie"

    # Dump JSON output to inspect structure
    python scripts/solar_chat_viz_cli.py --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure we can import the app when invoked directly from /main/backend/scripts
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.solar_ai_chat.chart_service import ChartSpecBuilder  # noqa: E402
from app.services.solar_ai_chat.chat_service import _should_visualize  # noqa: E402
from app.repositories.solar_ai_chat.topic_repository import (  # noqa: E402
    _apply_energy_focus,
    _normalize_focus,
)
from app.repositories.solar_ai_chat.kpi_repository import KpiRepository  # noqa: E402
from app.services.solar_ai_chat.tool_executor import _is_correlation_query  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture payloads (shapes chosen to match real tool outputs)
# ---------------------------------------------------------------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "station_daily_report": {
        "topic": "station_report",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": ["total_energy_mwh", "station_count"],
        "metrics": {
            "report_date": "2026-04-10",
            "station_count": 4,
            "total_energy_mwh": 812.45,
            "has_data": True,
            "stations": [
                {"facility": "Avonlie", "energy_mwh": 221.5, "capacity_factor_pct": 18.4},
                {"facility": "Bomen", "energy_mwh": 198.3, "capacity_factor_pct": 15.7},
                {"facility": "Darlington Point", "energy_mwh": 205.8, "capacity_factor_pct": 16.9},
                {"facility": "Finley", "energy_mwh": 186.9, "capacity_factor_pct": 14.1},
            ],
        },
    },
    "station_hourly_report": {
        "topic": "station_report",
        "expect_table": True,
        "expect_chart": "line",
        "expect_kpi_keys": ["total_energy_mwh", "row_count"],
        "metrics": {
            "report_date": "2026-04-10",
            "row_count": 24,
            "total_energy_mwh": 112.3,
            "hourly_rows": [
                {"hour": h, "facility": "Avonlie",
                 "energy_mwh": round(max(0.0, 8 * ((1.0 - abs(12 - h) / 12)) + 0.5), 3),
                 "capacity_factor_pct": round(max(0.0, 20 - abs(12 - h)), 2)}
                for h in range(24)
            ],
        },
    },
    "ml_model_info": {
        "topic": "ml_model_info",
        "expect_table": False,
        "expect_chart": None,
        "expect_kpi_keys": ["r2", "rmse", "mae"],
        "metrics": {
            "champion_model": "GBT",
            "r2": 0.89,
            "rmse": 0.34,
            "mae": 0.21,
            "nrmse_pct": 7.1,
            "skill_score": 0.58,
        },
    },
    "rag_chunks": {
        "topic": "search_documents",
        "expect_table": True,
        "expect_chart": None,
        "expect_kpi_keys": ["total_results"],
        "metrics": {
            "total_results": 3,
            "chunks": [
                {"content": "Inverter maintenance guide...", "source_file": "manual_abb_100.pdf",
                 "doc_type": "equipment_manual", "score": 0.91},
                {"content": "Incident report 2025-11-03 ...", "source_file": "incident_20251103.pdf",
                 "doc_type": "incident_report", "score": 0.85},
                {"content": "Champion model changelog ...", "source_file": "changelog_v4.md",
                 "doc_type": "model_changelog", "score": 0.78},
            ],
        },
    },
    # --- New fixtures covering the energy_performance + chart heuristics fixes --
    "energy_focus_overview_stripped": {
        # Simulates get_energy_performance(focus='overview') AFTER _apply_energy_focus.
        # all_facilities must contain ONLY energy_mwh; no capacity columns.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": ["facility_count"],
        "metrics": {
            "facility_count": 8,
            "window_days": 30,
            "tomorrow_forecast_mwh": 2684.8,
            "all_facilities": [
                {"facility_id": "DARLSF", "facility": "Darlington Point", "energy_mwh": 39252.79},
                {"facility_id": "AVLSF",  "facility": "Avonlie",          "energy_mwh": 37668.17},
                {"facility_id": "BOMENSF","facility": "Bomen",            "energy_mwh": 15560.58},
                {"facility_id": "EMERASF","facility": "Emerald",          "energy_mwh": 15055.97},
                {"facility_id": "FINLEYSF","facility": "Finley",          "energy_mwh": 10031.37},
                {"facility_id": "YATSF1", "facility": "Yatpool",          "energy_mwh":  9640.69},
                {"facility_id": "LIMOSF2","facility": "Limondale 2",      "energy_mwh":  5127.67},
                {"facility_id": "WRSF1",  "facility": "White Rock Solar Farm", "energy_mwh": 2755.95},
            ],
        },
    },
    "energy_focus_capacity_stripped": {
        # Simulates focus='capacity' — facility lists have only capacity_factor_pct.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": ["facility_count"],
        "metrics": {
            "facility_count": 8,
            "window_days": 30,
            "all_facilities": [
                {"facility_id": "EMERASF", "facility": "Emerald",     "capacity_factor_pct": 46.2},
                {"facility_id": "AVLSF",   "facility": "Avonlie",     "capacity_factor_pct": 41.0},
                {"facility_id": "LIMOSF2", "facility": "Limondale 2", "capacity_factor_pct": 37.7},
                {"facility_id": "BOMENSF", "facility": "Bomen",       "capacity_factor_pct": 35.34},
                {"facility_id": "WRSF1",   "facility": "White Rock Solar Farm", "capacity_factor_pct": 33.95},
                {"facility_id": "DARLSF",  "facility": "Darlington Point",       "capacity_factor_pct": 33.05},
                {"facility_id": "YATSF1",  "facility": "Yatpool",     "capacity_factor_pct": 28.63},
                {"facility_id": "FINLEYSF","facility": "Finley",      "capacity_factor_pct": 17.08},
            ],
        },
    },
    "grouped_bar_same_unit_mwh": {
        # actual + forecast both in MWh → grouped bar with 2 series on same axis.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "expect_chart_traces": 2,
        "metrics": {
            "forecast_accuracy": [
                {"facility": "Avonlie",          "actual_mwh": 37668.0, "forecast_mwh": 41000.0},
                {"facility": "Bomen",            "actual_mwh": 15560.0, "forecast_mwh": 16000.0},
                {"facility": "Emerald",          "actual_mwh": 15055.0, "forecast_mwh": 15100.0},
                {"facility": "Finley",           "actual_mwh": 10031.0, "forecast_mwh": 10500.0},
            ],
        },
    },
    "grouped_bar_mixed_unit_fallback": {
        # energy_mwh (MWh) + capacity_factor_pct (%) → mixed units → chart builder
        # MUST fall back to single bar (energy only), no dual-axis, no yaxis2.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "expect_chart_traces": 1,
        "expect_no_yaxis2": True,
        "metrics": {
            "mix": [
                {"facility": "Avonlie",          "energy_mwh": 37668.0, "capacity_factor_pct": 41.0},
                {"facility": "Bomen",            "energy_mwh": 15560.0, "capacity_factor_pct": 35.3},
                {"facility": "Emerald",          "energy_mwh": 15055.0, "capacity_factor_pct": 46.2},
                {"facility": "Finley",           "energy_mwh": 10031.0, "capacity_factor_pct": 17.1},
            ],
        },
    },
    "horizontal_bar_long_labels": {
        # Category labels > 14 chars → bar orientation must be "h".
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "expect_horizontal": True,
        "metrics": {
            "facilities": [
                {"facility": "Darlington Point Solar Farm", "energy_mwh": 39252.0},
                {"facility": "White Rock Solar Farm",       "energy_mwh":  2755.0},
                {"facility": "Limondale 2 Solar Farm",      "energy_mwh":  5127.0},
                {"facility": "Avonlie Solar Park",          "energy_mwh": 37668.0},
            ],
        },
    },
    "top_n_truncation_15": {
        # 20 rows input → chart must show exactly 15 bars; title gets "Top 15" prefix.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "expect_chart_x_len": 15,
        "expect_title_contains": "Top 15",
        "metrics": {
            "facilities": [
                {"facility": f"Facility-{i:02d}", "energy_mwh": float(100 - i * 3)}
                for i in range(20)
            ],
        },
    },
    "scatter_two_numerics": {
        # 2 numeric columns, no time, no category → scatter plot.
        "topic": "correlation",
        "expect_table": True,
        "expect_chart": "scatter",
        "expect_kpi_keys": [],
        "metrics": {
            "points": [
                {"temperature_2m": 20 + i * 0.5, "energy_mwh": 100 + i * 4}
                for i in range(12)
            ],
        },
    },
    "histogram_single_numeric": {
        # 1 numeric, 15 rows → histogram.
        "topic": "distribution",
        "expect_table": True,
        "expect_chart": "histogram",
        "expect_kpi_keys": [],
        "metrics": {
            "aqi_samples": [{"aqi_value": 30 + i * 2} for i in range(15)],
        },
    },
    "no_chart_when_all_strings": {
        # Only string columns → no chart.
        "topic": "general",
        "expect_table": True,
        "expect_chart": None,
        "expect_kpi_keys": [],
        "metrics": {
            "facilities": [
                {"facility": "A", "status": "ok"},
                {"facility": "B", "status": "warn"},
            ],
        },
    },
    "no_chart_when_text_heavy": {
        # Long content column → no chart.
        "topic": "search_documents",
        "expect_table": True,
        "expect_chart": None,
        "expect_kpi_keys": [],
        "metrics": {
            "chunks": [
                {"content": "x" * 200, "score": 0.9, "source_file": "a.pdf"},
                {"content": "y" * 200, "score": 0.8, "source_file": "b.pdf"},
            ],
        },
    },
    "empty_metrics": {
        "topic": "general",
        "expect_table": False,
        "expect_chart": None,
        "expect_kpi_keys": [],
        "metrics": {},
    },
    # --- Broader coverage: other real tool output shapes -------------------
    "forecast_72h": {
        "topic": "forecast_72h",
        "expect_table": True,
        "expect_chart": "line",
        "expect_kpi_keys": [],
        "metrics": {
            "daily_forecast": [
                {"date": "2026-04-21", "expected_mwh": 2720.0, "confidence_low": 2450.0, "confidence_high": 2990.0},
                {"date": "2026-04-22", "expected_mwh": 2684.0, "confidence_low": 2410.0, "confidence_high": 2960.0},
                {"date": "2026-04-23", "expected_mwh": 2650.0, "confidence_low": 2380.0, "confidence_high": 2920.0},
            ],
        },
    },
    "data_quality_issues": {
        "topic": "data_quality",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "metrics": {
            "low_score_facilities": [
                {"facility": "Finley",      "quality_score": 0.62, "likely_causes": "High null rate on shortwave_radiation"},
                {"facility": "Yatpool",     "quality_score": 0.71, "likely_causes": "Delayed ingestion"},
                {"facility": "Limondale 2", "quality_score": 0.74, "likely_causes": "Out-of-range AQI values"},
            ],
        },
    },
    "facility_info": {
        # Bar chart IS produced here because facility + numeric columns fit the
        # grouped-bar heuristic. It's not particularly useful (lat/long bars),
        # but it matches actual behaviour. Primary display is still the table.
        "topic": "facility_info",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "metrics": {
            "facilities": [
                {"facility_name": "Darlington Point", "latitude": -34.57, "longitude": 146.0, "capacity_mw": 275, "state": "NSW"},
                {"facility_name": "Avonlie",          "latitude": -35.12, "longitude": 146.2, "capacity_mw": 190, "state": "NSW"},
            ],
        },
    },
    "extreme_energy": {
        # Single-scalar metric output from get_extreme_* tools — no list, so no
        # DataTable. KPI cards surface the scalar fields instead.
        "topic": "energy_performance",
        "expect_table": False,
        "expect_chart": None,
        "expect_kpi_keys": [],
        "metrics": {
            "facility": "Avonlie",
            "value": 112.2,
            "unit": "MWh",
            "metric": "energy_mwh",
            "recorded_at": "2026-04-18 14:00:00",
        },
    },
    "mart_energy_daily_pr_correlation": {
        # Mimics query_gold_kpi('energy') output — per-facility-per-day rows
        # from gold.mart_energy_daily which has the REAL performance_ratio_pct
        # plus avg_temperature_c. This is the authoritative source for
        # "PR vs temperature" correlation.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "scatter",
        "expect_kpi_keys": [],
        "expect_title_contains": "Performance Ratio vs Avg Temperature",
        "metrics": {
            "rows": [
                {"facility": "Emerald",          "energy_mwh_daily": 450.71, "performance_ratio_pct": 78.2, "weighted_capacity_factor_pct": 42.68, "avg_temperature_c": 24.74, "avg_cloud_cover_pct": 23.94, "daily_insolation_kwh_m2": 5.1},
                {"facility": "Avonlie",          "energy_mwh_daily": 196.68, "performance_ratio_pct": 72.5, "weighted_capacity_factor_pct": 19.35, "avg_temperature_c": 12.11, "avg_cloud_cover_pct":  6.33, "daily_insolation_kwh_m2": 4.2},
                {"facility": "Bomen",            "energy_mwh_daily": 280.45, "performance_ratio_pct": 80.1, "weighted_capacity_factor_pct": 33.11, "avg_temperature_c": 11.60, "avg_cloud_cover_pct":  0.80, "daily_insolation_kwh_m2": 4.8},
                {"facility": "Finley",           "energy_mwh_daily":  47.18, "performance_ratio_pct": 65.3, "weighted_capacity_factor_pct":  9.69, "avg_temperature_c": 14.60, "avg_cloud_cover_pct":  5.33, "daily_insolation_kwh_m2": 3.9},
                {"facility": "White Rock Solar Farm", "energy_mwh_daily": 137.87, "performance_ratio_pct": 82.4, "weighted_capacity_factor_pct": 52.03, "avg_temperature_c": 15.35, "avg_cloud_cover_pct": 1.71, "daily_insolation_kwh_m2": 4.5},
                {"facility": "Yatpool",          "energy_mwh_daily":  52.52, "performance_ratio_pct": 68.9, "weighted_capacity_factor_pct": 17.34, "avg_temperature_c": 13.87, "avg_cloud_cover_pct": 95.36, "daily_insolation_kwh_m2": 2.1},
            ],
        },
    },
    "weather_impact_raw": {
        # Mimics the ACTUAL query_gold_kpi(weather_impact) output — column
        # names taken from the real Gold mart export (short form: avg_energy,
        # avg_temperature, rad_energy_ratio). Includes metadata columns that
        # must be hidden from the table AND excluded from the chart.
        "topic": "energy_performance",
        "expect_table": True,
        "expect_chart": "scatter",
        "expect_kpi_keys": [],
        "expect_title_contains": "vs Avg Temperature",
        "expect_table_hides": ["batch_id", "created_at", "updated_at", "region",
                                "weather_condition_description",
                                "observation_hours", "daytime_observation_hours"],
        "metrics": {
            "rows": [
                {"facility_id": "AVLSF", "facility": "Avonlie", "region": "NSW1",
                 "weather_condition_description": "Cloud=CLEAR | Temp=MILD",
                 "observation_hours": 9, "daytime_observation_hours": 4,
                 "avg_energy": 49.17, "total_energy": 196.68,
                 "avg_capacity_factor": 19.35,
                 "avg_temperature": 12.11, "avg_humidity": None, "avg_cloud_cover": 6.33,
                 "rad_energy_ratio": 363.32,
                 "batch_id": "abc", "created_at": "2026-04-19", "updated_at": "2026-04-19"},
                {"facility_id": "BOMENSF", "facility": "Bomen", "region": "NSW1",
                 "weather_condition_description": "Cloud=CLEAR | Temp=MILD",
                 "observation_hours": 15, "daytime_observation_hours": 7,
                 "avg_energy": 40.07, "total_energy": 280.45,
                 "avg_capacity_factor": 33.11,
                 "avg_temperature": 11.60, "avg_humidity": None, "avg_cloud_cover": 0.80,
                 "rad_energy_ratio": 265.33,
                 "batch_id": "abc", "created_at": "2026-04-19", "updated_at": "2026-04-19"},
                {"facility_id": "EMERASF", "facility": "Emerald", "region": "QLD1",
                 "weather_condition_description": "Cloud=CLEAR | Temp=MILD",
                 "observation_hours": 19, "daytime_observation_hours": 12,
                 "avg_energy": 37.55, "total_energy": 450.71,
                 "avg_capacity_factor": 42.68,
                 "avg_temperature": 24.74, "avg_humidity": None, "avg_cloud_cover": 23.94,
                 "rad_energy_ratio": 124.73,
                 "batch_id": "abc", "created_at": "2026-04-19", "updated_at": "2026-04-19"},
                {"facility_id": "FINLEYSF", "facility": "Finley", "region": "NSW1",
                 "weather_condition_description": "Cloud=CLOUDY | Temp=MILD",
                 "observation_hours": 6, "daytime_observation_hours": 3,
                 "avg_energy": 15.73, "total_energy": 47.18,
                 "avg_capacity_factor": 9.69,
                 "avg_temperature": 14.60, "avg_humidity": None, "avg_cloud_cover": 5.33,
                 "rad_energy_ratio": 146.98,
                 "batch_id": "abc", "created_at": "2026-04-19", "updated_at": "2026-04-19"},
                {"facility_id": "WRSF1", "facility": "White Rock Solar Farm", "region": "NSW1",
                 "weather_condition_description": "Cloud=CLEAR | Temp=MILD",
                 "observation_hours": 21, "daytime_observation_hours": 12,
                 "avg_energy": 11.45, "total_energy": 137.87,
                 "avg_capacity_factor": 52.03,
                 "avg_temperature": 15.35, "avg_humidity": None, "avg_cloud_cover": 1.71,
                 "rad_energy_ratio": 43.02,
                 "batch_id": "abc", "created_at": "2026-04-19", "updated_at": "2026-04-19"},
            ],
        },
    },
    "pipeline_status": {
        "topic": "pipeline_status",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": [],
        "metrics": {
            "stage_progress": [
                {"stage": "Bronze", "percent_complete": 100.0},
                {"stage": "Silver", "percent_complete": 95.0},
                {"stage": "Gold",   "percent_complete": 88.0},
                {"stage": "Serving","percent_complete": 72.0},
            ],
            "eta_minutes": 12,
            "alerts": [
                {"pipeline_name": "bronze_weather_daily", "quality_flag": "warn", "issue": "stale source"},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Offline validation
# ---------------------------------------------------------------------------


def _check(label: str, cond: bool, message: str) -> bool:
    tag = "OK  " if cond else "FAIL"
    line = f"  [{tag}] {label}: {message}"
    try:
        print(line)
    except UnicodeEncodeError:
        # Windows cp1252 can't handle Vietnamese diacritics etc.
        print(line.encode("ascii", "replace").decode("ascii"))
    return cond


def _print_chart_detail(chart: Any, indent: str = "    ") -> None:
    if chart is None:
        print(f"{indent}(no chart)")
        return
    spec = chart.plotly_spec or {}
    traces = spec.get("data") or []
    layout = spec.get("layout") or {}
    print(f"{indent}chart_type  = {chart.chart_type}")
    print(f"{indent}title       = {chart.title}")
    print(f"{indent}trace_count = {len(traces)}")
    for i, t in enumerate(traces):
        name = t.get("name")
        ttype = t.get("type")
        orient = t.get("orientation", "v")
        yaxis = t.get("yaxis", "y")
        xs = t.get("x") or []
        ys = t.get("y") or []
        x_sample = xs[:3] if isinstance(xs, list) else None
        y_sample = ys[:3] if isinstance(ys, list) else None
        print(
            f"{indent}  trace[{i}] type={ttype} name={name} orient={orient} "
            f"yaxis={yaxis} x_len={len(xs) if isinstance(xs, list) else 'n/a'} "
            f"y_len={len(ys) if isinstance(ys, list) else 'n/a'} "
            f"x_sample={x_sample} y_sample={y_sample}"
        )
    print(f"{indent}yaxis.title  = {layout.get('yaxis', {}).get('title')}")
    if "yaxis2" in layout:
        print(f"{indent}yaxis2       = {layout['yaxis2']}")


def _print_table_detail(table: Any, indent: str = "    ") -> None:
    if table is None:
        print(f"{indent}(no table)")
        return
    cols = [f"{c.label}({c.unit})" if c.unit else c.label for c in table.columns]
    print(f"{indent}title    = {table.title}")
    print(f"{indent}columns  = {cols}")
    print(f"{indent}rows     = {table.row_count}")
    if table.rows:
        print(f"{indent}row[0]   = {table.rows[0]}")


def _print_kpi_detail(kpi: Any, indent: str = "    ") -> None:
    if kpi is None:
        print(f"{indent}(no kpi)")
        return
    for card in kpi.cards:
        print(f"{indent}{card.label}: {card.value} {card.unit or ''}")


def _summarize_payload(
    table: Any, chart: Any, kpi: Any, *, table_row_sample: int = 3
) -> dict[str, Any]:
    """Compact JSON-friendly summary of the 3 output payloads.

    Intentionally small: just the shape (title, columns, types, counts,
    sample values) — enough to eyeball whether a test case is correct
    without drowning in full plotly_spec trees.
    """
    summary: dict[str, Any] = {"kpi_cards": None, "chart": None, "data_table": None}

    if kpi is not None:
        summary["kpi_cards"] = [
            {
                "label": c.label,
                "value": c.value,
                "unit": c.unit,
                "format": getattr(c, "format", None),
            }
            for c in kpi.cards
        ]

    if chart is not None:
        spec = chart.plotly_spec or {}
        layout = spec.get("layout") or {}
        traces = spec.get("data") or []
        summary["chart"] = {
            "chart_type": chart.chart_type,
            "title": chart.title,
            "source_metric_key": getattr(chart, "source_metric_key", None),
            "trace_count": len(traces),
            "traces": [
                {
                    "name": t.get("name"),
                    "type": t.get("type"),
                    "orientation": t.get("orientation", "v"),
                    "yaxis": t.get("yaxis", "y"),
                    "x_len": len(t.get("x", [])) if isinstance(t.get("x"), list) else None,
                    "y_len": len(t.get("y", [])) if isinstance(t.get("y"), list) else None,
                    "x_sample": (t.get("x") or [])[:3] if isinstance(t.get("x"), list) else None,
                    "y_sample": (t.get("y") or [])[:3] if isinstance(t.get("y"), list) else None,
                }
                for t in traces
            ],
            "layout": {
                "xaxis_title": (layout.get("xaxis") or {}).get("title"),
                "yaxis_title": (layout.get("yaxis") or {}).get("title"),
                "yaxis2_title": (layout.get("yaxis2") or {}).get("title") if "yaxis2" in layout else None,
                "barmode": layout.get("barmode"),
                "has_dual_axis": "yaxis2" in layout,
            },
        }

    if table is not None:
        summary["data_table"] = {
            "title": table.title,
            "row_count": table.row_count,
            "columns": [
                {"key": c.key, "label": c.label, "unit": c.unit, "type": c.type}
                for c in table.columns
            ],
            "rows_sample": table.rows[:table_row_sample] if table.rows else [],
        }

    return summary


def validate_fixtures(
    dump_path: Path | None,
    verbose: bool = False,
    summary_path: Path | None = None,
) -> bool:
    builder = ChartSpecBuilder()
    all_ok = True
    dump: dict[str, Any] = {}
    summaries: dict[str, Any] = {}

    for name, fixture in FIXTURES.items():
        print(f"\n-> Fixture: {name}")
        table, chart, kpi = builder.build(fixture["metrics"], topic=fixture["topic"])
        summaries[name] = _summarize_payload(table, chart, kpi)

        if verbose:
            print("  --- detail ---")
            _print_table_detail(table)
            _print_chart_detail(chart)
            _print_kpi_detail(kpi)

        dump[name] = {
            "data_table": table.model_dump() if table else None,
            "chart": chart.model_dump() if chart else None,
            "kpi_cards": kpi.model_dump() if kpi else None,
        }

        expect_table = fixture["expect_table"]
        ok = _check(
            "data_table present",
            (table is not None) == expect_table,
            f"expected={expect_table} got={table is not None}",
        )
        all_ok = all_ok and ok

        if table and expect_table:
            ok = _check(
                "data_table columns match keys",
                all(col.key in table.rows[0] for col in table.columns) if table.rows else True,
                f"columns={[c.key for c in table.columns]}",
            )
            all_ok = all_ok and ok
            ok = _check(
                "row_count matches rows length",
                table.row_count == len(table.rows),
                f"row_count={table.row_count} rows={len(table.rows)}",
            )
            all_ok = all_ok and ok

        expect_chart = fixture["expect_chart"]
        if expect_chart is None:
            ok = _check("chart absent", chart is None, f"got={chart.chart_type if chart else None}")
            all_ok = all_ok and ok
        else:
            ok = _check(
                f"chart_type == {expect_chart}",
                chart is not None and chart.chart_type == expect_chart,
                f"got={chart.chart_type if chart else None}",
            )
            all_ok = all_ok and ok
            if chart:
                spec = chart.plotly_spec
                has_data = isinstance(spec, dict) and isinstance(spec.get("data"), list) and len(spec["data"]) >= 1
                ok = _check("plotly_spec has data traces", has_data, f"spec keys={list(spec.keys())}")
                all_ok = all_ok and ok
                first_trace = spec["data"][0] if has_data else {}
                if expect_chart == "histogram":
                    # Histograms only have `x`; `y` is derived by Plotly from bin counts.
                    ok = _check(
                        "first trace has x array",
                        isinstance(first_trace.get("x"), list),
                        f"x_len={len(first_trace.get('x', []))}",
                    )
                else:
                    ok = _check(
                        "first trace has x and y arrays",
                        isinstance(first_trace.get("x"), list) and isinstance(first_trace.get("y"), list),
                        f"x_len={len(first_trace.get('x', []))} y_len={len(first_trace.get('y', []))}",
                    )
                all_ok = all_ok and ok

        # Extra chart-level assertions for new fixtures
        expect_traces = fixture.get("expect_chart_traces")
        if expect_traces is not None and chart is not None:
            traces = chart.plotly_spec.get("data") or []
            ok = _check(
                "trace count matches",
                len(traces) == expect_traces,
                f"expected={expect_traces} got={len(traces)}",
            )
            all_ok = all_ok and ok

        if fixture.get("expect_no_yaxis2") and chart is not None:
            layout = chart.plotly_spec.get("layout") or {}
            ok = _check(
                "no yaxis2 in layout",
                "yaxis2" not in layout,
                f"yaxis2 keys={list(layout.keys())}",
            )
            all_ok = all_ok and ok

        if fixture.get("expect_horizontal") and chart is not None:
            first_trace = (chart.plotly_spec.get("data") or [{}])[0]
            ok = _check(
                "first trace orientation == h",
                first_trace.get("orientation") == "h",
                f"got={first_trace.get('orientation')}",
            )
            all_ok = all_ok and ok

        expect_x_len = fixture.get("expect_chart_x_len")
        if expect_x_len is not None and chart is not None:
            first_trace = (chart.plotly_spec.get("data") or [{}])[0]
            x_or_y = first_trace.get("x") or first_trace.get("y") or []
            ok = _check(
                "chart x/y length matches",
                len(x_or_y) == expect_x_len,
                f"expected={expect_x_len} got={len(x_or_y)}",
            )
            all_ok = all_ok and ok

        expect_title_contains = fixture.get("expect_title_contains")
        if expect_title_contains and chart is not None:
            ok = _check(
                f"title contains '{expect_title_contains}'",
                expect_title_contains in chart.title,
                f"title={chart.title}",
            )
            all_ok = all_ok and ok

        expect_table_hides = fixture.get("expect_table_hides") or []
        if expect_table_hides and table is not None:
            present_keys = {c.key for c in table.columns}
            leaked = [k for k in expect_table_hides if k in present_keys]
            ok = _check(
                "metadata columns hidden from table",
                len(leaked) == 0,
                f"leaked={leaked}",
            )
            all_ok = all_ok and ok

        expected_kpis = set(fixture["expect_kpi_keys"])
        if expected_kpis:
            kpi_labels = {c.label for c in kpi.cards} if kpi else set()
            got_kpi_keys = set()
            for key in expected_kpis:
                # Match by label derived from key
                from app.services.solar_ai_chat.chart_service import _label_for as label_for  # type: ignore
                label, _, _ = label_for(key)
                if label in kpi_labels:
                    got_kpi_keys.add(key)
            ok = _check(
                "expected KPI keys surface",
                expected_kpis.issubset(got_kpi_keys),
                f"expected={sorted(expected_kpis)} got={sorted(got_kpi_keys)}",
            )
            all_ok = all_ok and ok

    if dump_path:
        dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        print(f"\nDumped fixture outputs to: {dump_path}")

    if summary_path:
        summary_path.write_text(
            json.dumps(summaries, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Dumped compact summaries to: {summary_path}")

    return all_ok


# ---------------------------------------------------------------------------
# _apply_energy_focus unit tests
# ---------------------------------------------------------------------------


def _raw_energy_metrics() -> dict[str, Any]:
    """Payload as produced by _energy_performance_databricks BEFORE focus filter."""
    facilities = [
        {"facility_id": "DARLSF", "facility": "Darlington Point", "energy_mwh": 39252.79, "capacity_factor_pct": 33.05},
        {"facility_id": "AVLSF",  "facility": "Avonlie",          "energy_mwh": 37668.17, "capacity_factor_pct": 41.0},
        {"facility_id": "BOMENSF","facility": "Bomen",            "energy_mwh": 15560.58, "capacity_factor_pct": 35.34},
        {"facility_id": "EMERASF","facility": "Emerald",          "energy_mwh": 15055.97, "capacity_factor_pct": 46.2},
        {"facility_id": "FINLEYSF","facility": "Finley",          "energy_mwh": 10031.37, "capacity_factor_pct": 17.08},
        {"facility_id": "YATSF1", "facility": "Yatpool",          "energy_mwh":  9640.69, "capacity_factor_pct": 28.63},
        {"facility_id": "LIMOSF2","facility": "Limondale 2",      "energy_mwh":  5127.67, "capacity_factor_pct": 37.7},
        {"facility_id": "WRSF1",  "facility": "White Rock Solar Farm", "energy_mwh": 2755.95, "capacity_factor_pct": 33.95},
    ]
    return {
        "all_facilities": facilities,
        "top_facilities": facilities[:3],
        "bottom_facilities": facilities[-3:],
        "facility_count": 8,
        "peak_hours": [{"hour": 11, "energy_mwh": 14439.52}],
        "top_performance_ratio_facilities": [
            {"facility": "Emerald",     "performance_ratio_pct": 46.2},
            {"facility": "Avonlie",     "performance_ratio_pct": 41.0},
            {"facility": "Limondale 2", "performance_ratio_pct": 37.7},
        ],
        "tomorrow_forecast_mwh": 2684.84,
        "window_days": 30,
    }


def validate_energy_focus() -> bool:
    print("\n== _apply_energy_focus unit tests ==")
    all_ok = True

    # Case 1: overview (default) — capacity_factor + performance_ratio must be stripped
    print("\n-> focus='overview'")
    out = _apply_energy_focus(_raw_energy_metrics(), "overview")
    first_row = out["all_facilities"][0]
    ok = _check(
        "no capacity_factor_pct in all_facilities",
        "capacity_factor_pct" not in first_row,
        f"row_keys={sorted(first_row.keys())}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "no top_performance_ratio_facilities",
        "top_performance_ratio_facilities" not in out,
        f"present={'top_performance_ratio_facilities' in out}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "energy_mwh preserved",
        "energy_mwh" in first_row,
        f"row_keys={sorted(first_row.keys())}",
    )
    all_ok = all_ok and ok

    # Case 2: focus='capacity' — energy_mwh must be stripped, capacity kept
    print("\n-> focus='capacity'")
    out = _apply_energy_focus(_raw_energy_metrics(), "capacity")
    first_row = out["all_facilities"][0]
    ok = _check(
        "no energy_mwh in all_facilities",
        "energy_mwh" not in first_row,
        f"row_keys={sorted(first_row.keys())}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "capacity_factor_pct preserved",
        "capacity_factor_pct" in first_row,
        f"row_keys={sorted(first_row.keys())}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "tomorrow_forecast_mwh removed",
        "tomorrow_forecast_mwh" not in out,
        f"present={'tomorrow_forecast_mwh' in out}",
    )
    all_ok = all_ok and ok

    # Case 3: focus='energy' + limit=5 — all_facilities truncated to 5 rows
    print("\n-> focus='energy' limit=5")
    out = _apply_energy_focus(_raw_energy_metrics(), "energy", limit=5)
    ok = _check(
        "all_facilities truncated to 5",
        len(out["all_facilities"]) == 5,
        f"len={len(out['all_facilities'])}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "first row is DARLSF (highest energy preserved)",
        out["all_facilities"][0].get("facility_id") == "DARLSF",
        f"first={out['all_facilities'][0]}",
    )
    all_ok = all_ok and ok

    # Case 3.5: alias normalization — LLMs invent values outside the enum.
    print("\n-> _normalize_focus aliases")
    alias_cases = [
        ("capacity_factor", "capacity"),
        ("CAPACITY_FACTOR", "capacity"),  # case insensitive
        ("capacity-factor", "capacity"),  # dash → underscore
        ("capacity factor", "capacity"),  # space → underscore
        ("efficiency", "capacity"),
        ("cf", "capacity"),
        ("energy_mwh", "energy"),
        ("mwh", "energy"),
        ("production", "energy"),
        ("summary", "overview"),
        ("", "overview"),
        (None, "overview"),
        ("bogus_value", "overview"),  # unknown → fallback with warning
    ]
    for raw, expected in alias_cases:
        got = _normalize_focus(raw)
        ok = _check(
            f"{raw!r} -> {expected}",
            got == expected,
            f"got={got}",
        )
        all_ok = all_ok and ok

    # Case 3.545: correlation scatter table must be projected to just the
    # relevant columns (facility + date + X + Y) — mart_energy_daily has 30+
    # columns, we must NOT dump them all next to a focused PR-vs-temp chart.
    print("\n-> correlation scatter projects table to relevant columns")
    wide_corr = {
        "rows": [
            {
                "facility": "Emerald", "energy_date": "2026-04-20",
                "performance_ratio_pct": 78.2, "avg_temperature_c": 24.7,
                "energy_mwh_daily": 450.71, "weighted_capacity_factor_pct": 42.68,
                "avg_cloud_cover_pct": 23.94, "daily_insolation_kwh_m2": 5.1,
                "avg_humidity": 30, "avg_wind_speed_ms": 4.2,
                "avg_shortwave_radiation": 420, "avg_direct_normal_irradiance": 513,
                "avg_diffuse_radiation": 58, "total_precipitation_mm": 0,
                "radiation_to_energy_ratio": 124.7,
            },
            {
                "facility": "Avonlie", "energy_date": "2026-04-20",
                "performance_ratio_pct": 72.5, "avg_temperature_c": 12.1,
                "energy_mwh_daily": 196.68, "weighted_capacity_factor_pct": 19.35,
                "avg_cloud_cover_pct": 6.33, "daily_insolation_kwh_m2": 4.2,
                "avg_humidity": 50, "avg_wind_speed_ms": 5.2,
                "avg_shortwave_radiation": 135, "avg_direct_normal_irradiance": 253,
                "avg_diffuse_radiation": 31, "total_precipitation_mm": 0,
                "radiation_to_energy_ratio": 363.3,
            },
        ],
    }
    table, chart, _ = ChartSpecBuilder().build(wide_corr, topic="energy_performance")
    ok = _check(
        "scatter chart built",
        chart is not None and chart.chart_type == "scatter",
        f"chart_type={chart.chart_type if chart else None}",
    )
    all_ok = all_ok and ok
    if table is not None:
        kept = {c.key for c in table.columns}
        # Must keep: facility + date + performance_ratio_pct + avg_temperature_c
        required = {"facility", "energy_date", "performance_ratio_pct", "avg_temperature_c"}
        ok = _check(
            "table keeps facility + date + X + Y",
            required.issubset(kept),
            f"kept={sorted(kept)}",
        )
        all_ok = all_ok and ok
        # Must drop the noise columns
        dropped_expected = {
            "avg_humidity", "avg_wind_speed_ms", "avg_shortwave_radiation",
            "avg_direct_normal_irradiance", "avg_diffuse_radiation",
            "total_precipitation_mm", "radiation_to_energy_ratio",
            "weighted_capacity_factor_pct", "avg_cloud_cover_pct",
            "daily_insolation_kwh_m2", "energy_mwh_daily",
        }
        leaked = dropped_expected & kept
        ok = _check(
            "table drops noise columns for scatter",
            len(leaked) == 0,
            f"leaked={sorted(leaked)}",
        )
        all_ok = all_ok and ok

    # Case 3.55: scatter (no facility column) should NOT get facility-oriented title
    print("\n-> scatter data (no facility column) keeps neutral title")
    scatter_metrics = {
        "correlation": [
            {"temperature_2m": 20 + i * 0.5, "energy_mwh": 100 + i * 4}
            for i in range(12)
        ],
    }
    table, _, _ = ChartSpecBuilder().build(scatter_metrics, topic="correlation")
    ok = _check(
        "scatter table title not 'Energy production by facility'",
        table is not None and "by facility" not in (table.title or "").lower(),
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok

    # Case 3.52: correlation scatter (temperature + capacity factor)
    print("\n-> correlation scatter for PR/capacity vs temperature")
    correlation = {
        "rows": [
            {"facility": "Emerald", "avg_temperature_c": 24.7, "avg_capacity_factor": 42.7},
            {"facility": "Avonlie", "avg_temperature_c": 12.1, "avg_capacity_factor": 19.4},
            {"facility": "Bomen",   "avg_temperature_c": 11.6, "avg_capacity_factor": 33.1},
            {"facility": "Finley",  "avg_temperature_c": 14.6, "avg_capacity_factor":  9.7},
        ],
    }
    _, chart, _ = ChartSpecBuilder().build(correlation, topic="energy_performance")
    ok = _check(
        "correlation → scatter chart_type",
        chart is not None and chart.chart_type == "scatter",
        f"chart_type={chart.chart_type if chart else None}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "correlation title 'Y vs X'",
        chart is not None and "vs" in chart.title,
        f"title={chart.title if chart else None}",
    )
    all_ok = all_ok and ok
    first_trace = (chart.plotly_spec.get("data") or [{}])[0]
    ok = _check(
        "scatter text labels = facility names",
        isinstance(first_trace.get("text"), list) and "Emerald" in first_trace.get("text", []),
        f"text={first_trace.get('text')}",
    )
    all_ok = all_ok and ok

    # Metadata columns (observation_hours) must not appear in chart
    print("\n-> metadata columns excluded from chart")
    with_metadata = {
        "rows": [
            {"facility": "A", "observation_hours": 10, "batch_id": "abc", "total_energy": 100, "avg_temperature_c": 20},
            {"facility": "B", "observation_hours": 12, "batch_id": "def", "total_energy": 150, "avg_temperature_c": 18},
        ],
    }
    _, chart, _ = ChartSpecBuilder().build(with_metadata, topic="energy_performance")
    trace_names = [t.get("name") for t in (chart.plotly_spec.get("data") or [])]
    ok = _check(
        "chart does NOT plot observation_hours",
        chart is not None and not any("Observation Hours" in (n or "") for n in trace_names),
        f"trace_names={trace_names}",
    )
    all_ok = all_ok and ok

    # Case 3.51: correlation query detection at the tool_executor layer
    print("\n-> correlation query detection")
    correlation_cases = [
        ("Tìm mối liên hệ giữa performance ratio và nhiệt độ", True),
        ("mối liên hệ giữa capacity factor và cloud cover", True),
        ("PR vs temperature", True),
        ("how does temperature affect energy", True),
        ("correlation between AQI and energy", True),
        ("so sánh năng lượng với nhiệt độ", True),
        # Not correlation — single-metric queries
        ("Summarize energy performance", False),
        ("Show top 5 by energy", False),
        ("Compare capacity factor per facility", False),  # single metric, no X-Y
        ("What is performance ratio?", False),  # definition
        ("", False),
    ]
    for query, expected in correlation_cases:
        got = _is_correlation_query(query)
        ok = _check(
            f"{query[:45]!r} -> {expected}",
            got == expected,
            f"got={got}",
        )
        all_ok = all_ok and ok

    # Case 3.53: query_gold_kpi table aliases
    print("\n-> query_gold_kpi table name aliases")
    kpi_alias_cases = [
        ("weather_impact", "weather_impact"),  # canonical
        # PR / capacity-factor vs temperature → mart_energy_daily (not weather_impact)
        ("performance_ratio_vs_temperature", "energy"),
        ("pr_vs_temperature", "energy"),
        ("temperature_correlation", "energy"),
        ("capacity_factor_vs_temperature", "energy"),
        ("performance_ratio", "energy"),
        # Weather band aggregates stay on weather_impact
        ("weather_band", "weather_impact"),
        ("cloud_band", "weather_impact"),
        # AQI
        ("aqi", "aqi_impact"),
        ("air_quality", "aqi_impact"),
        # Forecast
        ("forecast", "forecast_accuracy"),
        ("actual_vs_forecast", "forecast_accuracy"),
        # Energy
        ("daily_energy", "energy"),
        ("mart_energy_daily", "energy"),
        ("system", "system_kpi"),
        ("bogus_table", None),
        ("", None),
    ]
    for raw, expected in kpi_alias_cases:
        got = KpiRepository._resolve_table_name(raw)
        ok = _check(
            f"{raw!r} -> {expected}",
            got == expected,
            f"got={got}",
        )
        all_ok = all_ok and ok

    # Case 3.54: prefer facility NAME over facility_id in chart X-axis
    print("\n-> prefer facility name over facility_id")
    both = {
        "facilities": [
            {"facility_id": "EMERASF", "facility": "Emerald",    "capacity_factor_pct": 46.2},
            {"facility_id": "AVLSF",   "facility": "Avonlie",    "capacity_factor_pct": 41.0},
            {"facility_id": "DARLSF",  "facility": "Darlington", "capacity_factor_pct": 33.05},
        ],
    }
    _, chart, _ = ChartSpecBuilder().build(both, topic="energy_performance")
    first_trace = (chart.plotly_spec.get("data") or [{}])[0]
    x_values = first_trace.get("x") or first_trace.get("y") or []
    ok = _check(
        "chart X uses facility NAME (not facility_id)",
        "Emerald" in x_values and "EMERASF" not in x_values,
        f"x_values={x_values}",
    )
    all_ok = all_ok and ok
    xaxis_title = (chart.plotly_spec.get("layout") or {}).get("xaxis", {}).get("title") or \
                  (chart.plotly_spec.get("layout") or {}).get("yaxis", {}).get("title")
    ok = _check(
        "axis label reflects facility name (no 'Facility ID')",
        "Facility ID" != xaxis_title,
        f"axis_title={xaxis_title}",
    )
    all_ok = all_ok and ok

    # Case 3.56: time-series (hourly) title
    print("\n-> hourly data gets time-series title")
    hourly = {
        "hourly": [
            {"hour": h, "facility": "Avonlie", "energy_mwh": h * 2.0}
            for h in range(24)
        ],
    }
    table, _, _ = ChartSpecBuilder().build(hourly, topic="station_report")
    ok = _check(
        "hourly table title includes 'Hourly'",
        table is not None and "Hourly" in (table.title or ""),
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok

    # Case 3.57: mixed-unit payload (energy + capacity in same rows) → specific title
    print("\n-> mixed-unit table title")
    mixed = {
        "mix": [
            {"facility": "A", "energy_mwh": 1000, "capacity_factor_pct": 30},
            {"facility": "B", "energy_mwh":  500, "capacity_factor_pct": 20},
        ],
    }
    table, _, _ = ChartSpecBuilder().build(mixed, topic="energy_performance")
    ok = _check(
        "mixed-unit table title is 'Energy & capacity by facility'",
        table is not None and table.title == "Energy & capacity by facility",
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok

    # Case 3.6: content-aware table title
    print("\n-> content-aware table title")
    # When topic='energy_performance' but rows have only capacity_factor_pct,
    # the table title must reflect CAPACITY, not energy.
    capacity_only = _apply_energy_focus(_raw_energy_metrics(), "capacity")
    table, _, _ = ChartSpecBuilder().build(capacity_only, topic="energy_performance")
    ok = _check(
        "capacity-focused table title mentions 'Capacity'",
        table is not None and "Capacity" in table.title,
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok
    ok = _check(
        "capacity-focused table title does NOT say 'Energy performance'",
        table is not None and table.title != "Energy performance",
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok

    energy_only = _apply_energy_focus(_raw_energy_metrics(), "energy")
    table, _, _ = ChartSpecBuilder().build(energy_only, topic="energy_performance")
    ok = _check(
        "energy-focused table title mentions 'Energy'",
        table is not None and "Energy" in table.title,
        f"title={table.title if table else None}",
    )
    all_ok = all_ok and ok

    # Case 3.8: _should_visualize keyword triggers
    print("\n-> _should_visualize keyword triggers")
    viz_yes = [
        "Show chart of energy",                          # explicit
        "Plot the hourly generation",                    # explicit
        "Compare capacity factor per facility",          # implicit: compare + per facility
        "Show top 5 by energy",                          # implicit: top
        "Which facility has the highest energy output",  # implicit: highest
        "Summarize energy performance",                  # implicit: summarize
        "Energy trend over time",                        # implicit: trend
        "Hourly generation this month",                  # implicit: hourly
        "so sánh công suất",                             # implicit VN: so sánh
        "biểu đồ năng lượng",                            # explicit VN
    ]
    viz_no = [
        "What is PR ratio",                               # definition
        "Explain the ML model",                           # explanation
        "Is the pipeline running",                        # yes/no
        "What is the status",                             # factual lookup
    ]
    for q in viz_yes:
        ok = _check(
            f"viz=True for: {q!r}",
            _should_visualize(q, None),
            "expected True",
        )
        all_ok = all_ok and ok
    for q in viz_no:
        ok = _check(
            f"viz=False for: {q!r}",
            not _should_visualize(q, None),
            "expected False",
        )
        all_ok = all_ok and ok

    # Case 4: end-to-end — run _apply_energy_focus then ChartSpecBuilder; chart
    # MUST be single-unit (energy only) regardless of input mix.
    print("\n-> end-to-end: raw -> focus=overview -> ChartSpecBuilder")
    filtered = _apply_energy_focus(_raw_energy_metrics(), "overview")
    _, chart, _ = ChartSpecBuilder().build(filtered, topic="energy_performance")
    traces = chart.plotly_spec.get("data") if chart else []
    ok = _check(
        "chart has exactly 1 trace (no mixed-unit grouped bar)",
        chart is not None and len(traces) == 1,
        f"trace_count={len(traces)}",
    )
    all_ok = all_ok and ok
    layout = chart.plotly_spec.get("layout") if chart else {}
    ok = _check(
        "no yaxis2 (no dual-axis)",
        "yaxis2" not in (layout or {}),
        f"layout_keys={list((layout or {}).keys())}",
    )
    all_ok = all_ok and ok
    first_trace_name = (traces[0].get("name") if traces else "") or ""
    ok = _check(
        "trace name is Energy (not Capacity)",
        "Energy" in first_trace_name and "Capacity" not in first_trace_name,
        f"trace_name={first_trace_name}",
    )
    all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Live mode (optional; hits the running backend)
# ---------------------------------------------------------------------------


def validate_live(
    base_url: str,
    username: str,
    password: str,
    role: str,
    message: str,
    *,
    session_title: str,
    use_stream: bool,
) -> bool:
    try:
        import httpx
    except ImportError:
        print("httpx not installed; cannot run live mode.")
        return False

    print(f"\n-> Live mode: POST {base_url}/solar-ai-chat/{'stream' if use_stream else 'query'}")
    with httpx.Client(base_url=base_url, timeout=180.0, follow_redirects=False) as client:
        resp = client.post("/auth/login", data={"username": username, "password": password, "next": "/dashboard"})
        if resp.status_code not in (302, 303):
            print(f"Login failed: {resp.status_code} {resp.text[:200]}")
            return False

        sess_resp = client.post("/solar-ai-chat/sessions", json={"role": role, "title": session_title})
        if sess_resp.status_code >= 400:
            print(f"Create session failed: {sess_resp.status_code} {sess_resp.text[:200]}")
            return False
        session_id = sess_resp.json().get("session_id", "")

        payload = {"role": role, "session_id": session_id, "message": message}

        if not use_stream:
            r = client.post("/solar-ai-chat/query", json=payload)
            if r.status_code >= 400:
                print(f"Query failed: {r.status_code} {r.text[:200]}")
                return False
            body = r.json()
            return _assert_live_payload(body)

        # Streaming
        with client.stream("POST", "/solar-ai-chat/stream", json=payload) as r:
            if r.status_code >= 400:
                print(f"Stream failed: {r.status_code}")
                return False
            buffer = ""
            done_evt: dict[str, Any] | None = None
            for chunk in r.iter_text():
                buffer += chunk
                blocks = buffer.split("\n\n")
                buffer = blocks.pop()
                for block in blocks:
                    line = block.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if evt.get("event") == "done":
                        done_evt = evt
            if done_evt is None:
                print("No done event received.")
                return False
            return _assert_live_payload(done_evt)


def _assert_live_payload(body: dict[str, Any]) -> bool:
    has_any = any(body.get(k) for k in ("data_table", "chart", "kpi_cards"))
    ok = _check("response has at least one viz field", has_any, f"keys={[k for k in ('data_table','chart','kpi_cards') if body.get(k)]}")
    if body.get("data_table"):
        dt = body["data_table"]
        ok &= _check("data_table has rows", bool(dt.get("rows")), f"row_count={dt.get('row_count')}")
    if body.get("chart"):
        ch = body["chart"]
        ok &= _check("chart has plotly_spec.data", isinstance(ch.get("plotly_spec", {}).get("data"), list), f"chart_type={ch.get('chart_type')}")
    return bool(ok)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Solar AI Chat visualization pipeline.")
    p.add_argument("--live", action="store_true", help="Run live validation against a running backend.")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--username", default="admin")
    p.add_argument("--password", default="admin123")
    p.add_argument("--role", default="data_engineer")
    p.add_argument("--message", default="Station daily report")
    p.add_argument("--session-title", default="CLI viz benchmark")
    p.add_argument("--stream", action="store_true", help="In live mode, use the SSE endpoint.")
    p.add_argument("--output-json", default="", help="Dump full offline fixture outputs to this path.")
    p.add_argument("--summary-json", default="", help="Dump compact JSON summary (card/chart/table shape) to this path.")
    p.add_argument("--verbose", "-v", action="store_true", help="Print chart/table/KPI detail for each fixture.")
    p.add_argument("--skip-focus-tests", action="store_true", help="Skip the _apply_energy_focus unit tests.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dump_path = Path(args.output_json) if args.output_json else None
    summary_path = Path(args.summary_json) if args.summary_json else None

    print("== Offline fixtures ==")
    ok = validate_fixtures(dump_path, verbose=args.verbose, summary_path=summary_path)

    if not args.skip_focus_tests:
        focus_ok = validate_energy_focus()
        ok = ok and focus_ok

    if args.live:
        print("\n== Live mode ==")
        live_ok = validate_live(
            base_url=args.base_url,
            username=args.username,
            password=args.password,
            role=args.role,
            message=args.message,
            session_title=args.session_title,
            use_stream=args.stream,
        )
        ok = ok and live_ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
