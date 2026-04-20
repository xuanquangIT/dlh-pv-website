"""ChartSpecBuilder — auto-detect chart type + DataTable + KPI cards.

Inspired by Vanna AI's `visualize_data` tool and `DataFrameComponent`, but
re-implemented here with no Vanna dependency. Takes the ``key_metrics`` dict
returned by Solar AI Chat tools and derives three independent payloads:

* ``DataTablePayload`` — the most informative list-of-dicts becomes a table
* ``ChartPayload``     — Plotly spec auto-chosen by heuristics
* ``KpiCardsPayload``  — scalar numbers extracted as headline KPI cards

Heuristics deliberately stay simple (no numpy / no plotly Python dependency):
we emit raw Plotly JSON spec dicts that Plotly.js renders on the frontend.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Iterable

from app.schemas.solar_ai_chat.visualization import (
    ChartPayload,
    ColumnType,
    DataTableColumn,
    DataTablePayload,
    KpiCard,
    KpiCardsPayload,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column label & unit dictionary (project-specific)
# ---------------------------------------------------------------------------

_LABELS: dict[str, tuple[str, str | None, ColumnType]] = {
    "facility": ("Facility", None, "string"),
    "facility_name": ("Facility", None, "string"),
    "facility_id": ("Facility ID", None, "string"),
    "station": ("Station", None, "string"),
    "hour": ("Hour", "h", "number"),
    "hr": ("Hour", "h", "number"),
    "date": ("Date", None, "date"),
    "reading_date": ("Date", None, "date"),
    "report_date": ("Date", None, "date"),
    "date_hour": ("Timestamp", None, "datetime"),
    "timestamp": ("Timestamp", None, "datetime"),
    "energy_mwh": ("Energy", "MWh", "number"),
    "energy_total_mwh": ("Total Energy", "MWh", "number"),
    "capacity_factor_pct": ("Capacity Factor", "%", "number"),
    "pr_ratio_pct": ("PR Ratio", "%", "number"),
    "performance_ratio_pct": ("PR Ratio", "%", "number"),
    "shortwave_radiation": ("Shortwave Radiation", "W/m2", "number"),
    "temperature_2m": ("Temperature", "C", "number"),
    "wind_speed_10m": ("Wind Speed", "m/s", "number"),
    "wind_gusts_10m": ("Wind Gust", "m/s", "number"),
    "cloud_cover": ("Cloud Cover", "%", "number"),
    "aqi_value": ("AQI", None, "number"),
    "score": ("Score", None, "number"),
    "similarity_score": ("Similarity", None, "number"),
    "source_file": ("Source", None, "string"),
    "doc_type": ("Doc Type", None, "string"),
    "content": ("Content", None, "string"),
    "mae": ("MAE", None, "number"),
    "rmse": ("RMSE", None, "number"),
    "r2": ("R2", None, "number"),
    "nrmse_pct": ("nRMSE", "%", "number"),
    "nmae_pct": ("nMAE", "%", "number"),
    "skill_score": ("Skill Score", None, "number"),
    "total_energy_mwh": ("Total Energy", "MWh", "number"),
    "station_count": ("Stations", None, "integer"),
    "row_count": ("Rows", None, "integer"),
    "forecast_mwh": ("Forecast", "MWh", "number"),
    "actual_mwh": ("Actual", "MWh", "number"),
}

_SCALAR_PRIORITY = [
    "total_energy_mwh",
    "energy_total_mwh",
    "station_count",
    "row_count",
    "forecast_mae",
    "r2",
    "rmse",
    "mae",
    "pr_ratio_pct",
    "capacity_factor_pct",
    "skill_score",
    "fallback_rate",
]

_TIME_COLUMNS = {"hour", "hr", "date", "date_hour", "timestamp", "report_date", "reading_date", "ts"}
_CATEGORY_COLUMNS = {"facility", "facility_name", "station", "facility_id", "category", "label", "name"}


def _humanize(key: str) -> str:
    return re.sub(r"[_\s]+", " ", key).strip().title()


def _label_for(key: str) -> tuple[str, str | None, ColumnType]:
    if key in _LABELS:
        return _LABELS[key]
    humanized = _humanize(key)
    if key.endswith("_pct"):
        return humanized.replace(" Pct", ""), "%", "number"
    if key.endswith("_mwh"):
        return humanized.replace(" Mwh", ""), "MWh", "number"
    return humanized, None, "string"


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pick_best_list(metrics: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """Return the (key, list-of-dicts) candidate most suitable for a table."""
    best: tuple[str, list[dict[str, Any]], int] | None = None
    for key, value in metrics.items():
        if not isinstance(value, list) or not value:
            continue
        if not all(isinstance(item, dict) for item in value):
            continue
        size = len(value)
        if best is None or size > best[2]:
            best = (key, value, size)
    if best is None:
        return None
    return best[0], best[1]


def _infer_column_type(sample_values: Iterable[Any]) -> ColumnType:
    seen_numeric = False
    seen_non_numeric = False
    for v in sample_values:
        if v is None:
            continue
        if _is_numeric(v):
            seen_numeric = True
        else:
            seen_non_numeric = True
    if seen_numeric and not seen_non_numeric:
        return "number"
    return "string"


def _build_columns(rows: list[dict[str, Any]]) -> list[DataTableColumn]:
    if not rows:
        return []
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    columns: list[DataTableColumn] = []
    for key in keys:
        label, unit, col_type = _label_for(key)
        if col_type == "string":
            col_type = _infer_column_type(row.get(key) for row in rows[:10])
        columns.append(DataTableColumn(key=key, label=label, type=col_type, unit=unit))
    return columns


def _build_data_table(
    metric_key: str,
    rows: list[dict[str, Any]],
    title_hint: str | None = None,
) -> DataTablePayload:
    columns = _build_columns(rows)
    title = title_hint or _humanize(metric_key)
    sanitized_rows = [
        {col.key: _coerce_scalar(row.get(col.key)) for col in columns}
        for row in rows
    ]
    return DataTablePayload(
        title=title,
        columns=columns,
        rows=sanitized_rows,
        row_count=len(sanitized_rows),
    )


def _detect_time_column(columns: list[DataTableColumn]) -> DataTableColumn | None:
    for col in columns:
        if col.key in _TIME_COLUMNS or col.type in ("date", "datetime"):
            return col
    return None


def _detect_category_column(columns: list[DataTableColumn]) -> DataTableColumn | None:
    for col in columns:
        if col.key in _CATEGORY_COLUMNS:
            return col
    # fallback: first non-numeric column
    for col in columns:
        if col.type == "string":
            return col
    return None


def _detect_numeric_columns(columns: list[DataTableColumn]) -> list[DataTableColumn]:
    return [c for c in columns if c.type in ("number", "integer")]


def _sorted_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    def _sort_key(row: dict[str, Any]) -> Any:
        v = row.get(key)
        if v is None:
            return ""
        return v

    try:
        return sorted(rows, key=_sort_key)
    except TypeError:
        return rows


_TEXT_HEAVY_KEYS = {"content", "text", "description", "body", "snippet", "message"}


def _has_text_heavy_column(rows: list[dict[str, Any]], columns: list[DataTableColumn]) -> bool:
    """Return True if any string column looks like long-form text (e.g. RAG `content`)."""
    for col in columns:
        if col.key in _TEXT_HEAVY_KEYS:
            return True
        if col.type != "string":
            continue
        for row in rows[:10]:
            v = row.get(col.key)
            if isinstance(v, str) and len(v) > 80:
                return True
    return False


def _build_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    columns: list[DataTableColumn],
) -> ChartPayload | None:
    if not rows or not columns:
        return None

    numeric_cols = _detect_numeric_columns(columns)
    if not numeric_cols:
        return None

    if _has_text_heavy_column(rows, columns):
        return None

    time_col = _detect_time_column(columns)
    cat_col = _detect_category_column(columns)

    # Line chart: time series (possibly multi-series by category)
    if time_col is not None:
        y_col = next((c for c in numeric_cols if c.key != time_col.key), None)
        if y_col is None:
            return None
        title = f"{y_col.label} over {time_col.label}"
        if cat_col is not None and cat_col.key != time_col.key:
            categories: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                key = str(row.get(cat_col.key, "all"))
                categories.setdefault(key, []).append(row)
            traces = []
            for cat_name, cat_rows in categories.items():
                cat_rows = _sorted_by(cat_rows, time_col.key)
                traces.append({
                    "x": [_coerce_scalar(r.get(time_col.key)) for r in cat_rows],
                    "y": [r.get(y_col.key) for r in cat_rows],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": cat_name,
                })
        else:
            sorted_rows = _sorted_by(rows, time_col.key)
            traces = [{
                "x": [_coerce_scalar(r.get(time_col.key)) for r in sorted_rows],
                "y": [r.get(y_col.key) for r in sorted_rows],
                "type": "scatter",
                "mode": "lines+markers",
                "name": y_col.label,
            }]
        layout = {
            "title": title,
            "xaxis": {"title": time_col.label},
            "yaxis": {"title": f"{y_col.label}{f' ({y_col.unit})' if y_col.unit else ''}"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
        }
        return ChartPayload(
            chart_type="line",
            title=title,
            plotly_spec={"data": traces, "layout": layout},
            source_metric_key=metric_key,
        )

    # Bar chart: categorical vs numeric
    if cat_col is not None:
        y_col = next((c for c in numeric_cols if c.key != cat_col.key), None)
        if y_col is None:
            return None
        sorted_rows = sorted(
            rows,
            key=lambda r: (r.get(y_col.key) if _is_numeric(r.get(y_col.key)) else float("-inf")),
            reverse=True,
        )
        title = f"{y_col.label} by {cat_col.label}"
        traces = [{
            "x": [str(r.get(cat_col.key, "")) for r in sorted_rows],
            "y": [r.get(y_col.key) for r in sorted_rows],
            "type": "bar",
            "name": y_col.label,
        }]
        layout = {
            "title": title,
            "xaxis": {"title": cat_col.label},
            "yaxis": {"title": f"{y_col.label}{f' ({y_col.unit})' if y_col.unit else ''}"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 80},
        }
        return ChartPayload(
            chart_type="bar",
            title=title,
            plotly_spec={"data": traces, "layout": layout},
            source_metric_key=metric_key,
        )

    return None


def _format_for_kpi(key: str, value: Any) -> KpiCard | None:
    if not _is_numeric(value):
        return None
    label, unit, _ = _label_for(key)
    fmt: str
    if key.endswith("_pct") or unit == "%":
        fmt = "percent"
    elif isinstance(value, int) or key.endswith("_count") or key in ("row_count", "station_count"):
        fmt = "integer"
    else:
        fmt = "number"
    return KpiCard(label=label, value=value, unit=unit, format=fmt)


def _build_kpi_cards(metrics: dict[str, Any]) -> KpiCardsPayload | None:
    # Keep scalar cards consistent with list payloads for common facility/station outputs.
    facilities = metrics.get("facilities")
    if isinstance(facilities, list):
        inferred_count = len(facilities)
        current_count = metrics.get("facility_count")
        if inferred_count > 0 and (not isinstance(current_count, (int, float)) or int(current_count) <= 0):
            metrics = {**metrics, "facility_count": inferred_count}

    cards: list[KpiCard] = []
    seen: set[str] = set()

    for key in _SCALAR_PRIORITY:
        if key in metrics and key not in seen:
            card = _format_for_kpi(key, metrics[key])
            if card is not None:
                cards.append(card)
                seen.add(key)

    for key, value in metrics.items():
        if key in seen:
            continue
        if not _is_numeric(value):
            continue
        card = _format_for_kpi(key, value)
        if card is not None:
            cards.append(card)
            seen.add(key)
        if len(cards) >= 6:
            break

    if not cards:
        return None
    return KpiCardsPayload(cards=cards[:6])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChartSpecBuilder:
    """Build DataTable / Chart / KPI payloads from a Solar AI Chat metrics dict."""

    def build(
        self,
        metrics: dict[str, Any],
        *,
        topic: str | None = None,
    ) -> tuple[DataTablePayload | None, ChartPayload | None, KpiCardsPayload | None]:
        if not isinstance(metrics, dict) or not metrics:
            return None, None, None

        table: DataTablePayload | None = None
        chart: ChartPayload | None = None

        candidate = _pick_best_list(metrics)
        if candidate is not None:
            metric_key, rows = candidate
            try:
                title_hint = _title_for_topic(topic, metric_key)
                table = _build_data_table(metric_key, rows, title_hint=title_hint)
                chart = _build_chart(metric_key, rows, table.columns)
            except Exception as exc:
                logger.warning("chart_spec_build_failed key=%s error=%s", metric_key, exc)
                table = None
                chart = None

        kpi = _build_kpi_cards(metrics)
        return table, chart, kpi


def _title_for_topic(topic: str | None, metric_key: str) -> str:
    topic_titles = {
        "station_report": "Station daily report",
        "forecast_72h": "72h forecast",
        "energy_performance": "Energy performance",
        "data_quality": "Data quality issues",
        "ml_model_info": "ML model metrics",
        "pipeline_status": "Pipeline status",
    }
    if topic and topic in topic_titles:
        return topic_titles[topic]
    return _humanize(metric_key)


__all__ = ["ChartSpecBuilder"]
