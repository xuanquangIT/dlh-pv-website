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
    "latitude": ("Latitude", "°", "number"),
    "longitude": ("Longitude", "°", "number"),
    "location_lat": ("Latitude", "°", "number"),
    "location_lng": ("Longitude", "°", "number"),
    "timezone_name": ("Timezone", None, "string"),
    "timezone_utc_offset": ("UTC Offset", None, "string"),
    "capacity_mw": ("Capacity", "MW", "number"),
    "quality_score": ("Quality", None, "number"),
    "percent_complete": ("Complete", "%", "number"),
    "expected_mwh": ("Expected", "MWh", "number"),
    "confidence_low": ("Lower bound", "MWh", "number"),
    "confidence_high": ("Upper bound", "MWh", "number"),
    # weather_impact / mart_weather_impact_daily columns (long + short names)
    "avg_temperature_c": ("Avg Temperature", "°C", "number"),
    "avg_temperature": ("Avg Temperature", "°C", "number"),
    "avg_humidity": ("Avg Humidity", "%", "number"),
    "avg_humidity_pct": ("Avg Humidity", "%", "number"),
    "avg_wind_speed_ms": ("Avg Wind Speed", "m/s", "number"),
    "avg_wind_speed": ("Avg Wind Speed", "m/s", "number"),
    "clear_sky_energy_loss_mwh": ("Clear Sky Energy Loss", "MWh", "number"),
    "radiation_to_energy_ratio": ("Rad/Energy Ratio", None, "number"),
    "avg_wind_speed": ("Avg Wind Speed", "m/s", "number"),
    "avg_cloud_cover": ("Avg Cloud Cover", "%", "number"),
    "avg_shortwave_radiation": ("Avg Shortwave Rad", "W/m²", "number"),
    "avg_shortwave_rad": ("Avg Shortwave Rad", "W/m²", "number"),
    "avg_direct_normal_irradiance": ("Avg DNI", "W/m²", "number"),
    "avg_dni": ("Avg DNI", "W/m²", "number"),
    "avg_diffuse_radiation": ("Avg Diffuse Rad", "W/m²", "number"),
    "avg_diffuse_rad": ("Avg Diffuse Rad", "W/m²", "number"),
    "avg_energy_mwh_per_hour": ("Avg Energy", "MWh/h", "number"),
    "avg_energy": ("Avg Energy", "MWh/h", "number"),
    "total_energy": ("Total Energy", "MWh", "number"),
    "avg_capacity_factor": ("Avg Capacity Factor", "%", "number"),
    "avg_capacity_factor_pct": ("Avg Capacity Factor", "%", "number"),
    "total_precipitation_mm": ("Total Precip", "mm", "number"),
    "total_precip": ("Total Precip", "mm", "number"),
    "radiation_to_energy_ratio": ("Rad/Energy Ratio (PR proxy)", None, "number"),
    "rad_energy_ratio": ("Rad/Energy Ratio (PR proxy)", None, "number"),
    "clear_sky_energy_loss": ("Clear Sky Loss", "MWh", "number"),
    # mart_energy_daily columns (has real PR + temperature — authoritative
    # source for "PR vs temperature" correlations)
    "performance_ratio_pct": ("Performance Ratio", "%", "number"),
    "weighted_capacity_factor_pct": ("Weighted Cap Factor", "%", "number"),
    "energy_mwh_daily": ("Daily Energy", "MWh", "number"),
    "avg_cloud_cover_pct": ("Avg Cloud Cover", "%", "number"),
    "max_temperature_c": ("Max Temperature", "°C", "number"),
    "min_temperature_c": ("Min Temperature", "°C", "number"),
    "daily_insolation_kwh_m2": ("Daily Insolation", "kWh/m²", "number"),
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
    "pr_ratio_pct": ("Performance Ratio", "%", "number"),
    # Note: canonical `performance_ratio_pct` label is defined near the top of
    # this dict; intentionally not duplicated here.
    "shortwave_radiation": ("Shortwave Radiation", "W/m2", "number"),
    "temperature_2m": ("Temperature", "C", "number"),
    "wind_speed_10m": ("Wind Speed", "m/s", "number"),
    "wind_gusts_10m": ("Wind Gust", "m/s", "number"),
    "cloud_cover": ("Cloud Cover", "%", "number"),
    "aqi_value": ("AQI", None, "number"),
    "avg_aqi_value": ("Avg AQI", None, "number"),
    "max_aqi_value": ("Max AQI", None, "number"),
    "avg_pm25": ("Avg PM2.5", "µg/m³", "number"),
    "avg_pm10": ("Avg PM10", "µg/m³", "number"),
    "avg_no2": ("Avg NO2", "µg/m³", "number"),
    "avg_o3": ("Avg O3", "µg/m³", "number"),
    "avg_dust": ("Avg Dust", "µg/m³", "number"),
    "aqi_date": ("Date", None, "date"),
    "aqi_category": ("AQI Category", None, "string"),
    "estimated_aqi_energy_loss_pct": ("Est. AQI Energy Loss", "%", "number"),
    "radiation_efficiency_ratio": ("Radiation Efficiency", None, "number"),
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
    "day_count": ("Days", None, "integer"),
    "row_count": ("Rows", None, "integer"),
    "forecast_mwh": ("Forecast", "MWh", "number"),
    "actual_mwh": ("Actual", "MWh", "number"),
}

_SCALAR_PRIORITY = [
    "total_energy_mwh",
    "energy_total_mwh",
    "station_count",
    "day_count",
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

_TIME_COLUMNS = {
    "hour", "hr",
    "date", "date_hour", "timestamp", "ts",
    "report_date", "reading_date",
    "energy_date", "forecast_date", "weather_date", "event_date", "aqi_date",
}

# Columns that are operational / metadata noise — they're numeric but the user
# almost never wants them plotted. Keep them in the table, drop from chart.
_METADATA_NUMERIC_KEYS = {
    "batch_id",
    "aqi_category_key",  # integer code, not meaningful to user
    "observation_hours",
    "daytime_observation_hours",
    "row_count",  # row_count is a scalar KPI, not a chart series
    "station_count",
    "facility_count",
    "window_days",
    "eta_minutes",
    "uncertainty_factor",
    "confidence_low",
    "confidence_high",
    "id",
}
_METADATA_STRING_KEYS = {
    "batch_id",
    "created_at",
    "updated_at",
    "weather_condition_description",
    "region",
    "cloud_band",
    "temperature_band",
    "rain_band",
    "weather_date",  # already covered by _TIME_COLUMNS but include defensively
    "dominant_aqi_category",
    "dominant_weather_condition",
}

# Operational / audit / duplicate columns that clutter the table but rarely
# answer the user's actual question.
_NOISY_TABLE_KEYS = {
    # audit / quality metadata
    "completeness_pct_daily",
    "good_quality_row",
    "warning_quality_row",
    "bad_quality_row",
    "quality_flag",
    # rank metadata
    "facility_rank_by_energy",
    "facility_rank_by_cf",
    "facility_rank",
    # duplicated/derived energy columns
    "energy_kwh_daily",
    "specific_yield_kwh_per_kwp",
    # infrastructure columns
    "rated_capacity_mw",
    # peak/secondary stats (user usually wants avg)
    "peak_capacity_factor",
    "peak_production_hour_utc",
    "peak_production_hour_local",
    "max_temperature",
    "max_temperature_c",
    "min_temperature_c",
    "daytime_hours_above_threshold",
    "zero_daytime_hours",
    "night_energy_events",
    # duplicate precipitation
    "avg_precipitation_mm",
}

_HIDDEN_TABLE_KEYS = _METADATA_NUMERIC_KEYS | _METADATA_STRING_KEYS | _NOISY_TABLE_KEYS
# Ordered list: earlier entries are preferred as the chart's X-axis category.
# Human-readable names (facility, facility_name, station) must come BEFORE
# opaque codes (facility_id) so charts display "Emerald" instead of "EMERASF".
_CATEGORY_COLUMNS_PRIORITY = [
    "facility",
    "facility_name",
    "station",
    "station_name",
    "category",
    "label",
    "name",
    "stage",
    "facility_id",  # fallback code
]
_CATEGORY_COLUMNS = set(_CATEGORY_COLUMNS_PRIORITY)

# Chart rendering limits (inspired by Vanna's Plotly chart_generator).
_MAX_BAR_CATEGORIES = 15        # top-N truncation for bar charts
_MAX_LINE_SERIES = 5            # cap multi-series line charts
_HORIZONTAL_LABEL_LEN = 14      # rotate to horizontal bar when labels are long
_HORIZONTAL_CAT_COUNT = 8       # or when too many categories
_HISTOGRAM_MIN_ROWS = 12        # need enough data points to be meaningful

# Brand palette — PV Lakehouse green/blue/solar.
_COLOR_PALETTE = [
    "#1a8a5a",  # green
    "#1b6ca8",  # blue
    "#f4b942",  # solar
    "#e07b39",  # orange
    "#7f57c5",  # purple
    "#2bb3c0",  # teal
    "#c0392b",  # red
    "#5a7684",  # slate
]


_LAT_KEYS: frozenset[str] = frozenset({"location_lat", "latitude", "lat"})
_LNG_KEYS: frozenset[str] = frozenset({"location_lng", "longitude", "lng"})
_GEO_KEYS: frozenset[str] = _LAT_KEYS | _LNG_KEYS


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
    # Drop operational/metadata columns from the user-facing table (batch_id,
    # created_at, weather_condition_description, region, etc.). The full
    # record is still available in the raw tool output if needed.
    columns = [c for c in columns if c.key not in _HIDDEN_TABLE_KEYS]
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
    # Walk the priority list so "facility" (human-readable name) wins over
    # "facility_id" (opaque code) when both are present in the row.
    by_key = {col.key: col for col in columns}
    for key in _CATEGORY_COLUMNS_PRIORITY:
        if key in by_key:
            return by_key[key]
    # Fallback: first non-numeric column in the original order.
    for col in columns:
        if col.type == "string":
            return col
    return None


def _detect_geo_columns(
    columns: list[DataTableColumn],
) -> tuple[DataTableColumn, DataTableColumn] | None:
    """Return (lat_col, lng_col) if both lat and lng columns are present."""
    by_key = {c.key: c for c in columns}
    lat_col = next((by_key[k] for k in _LAT_KEYS if k in by_key), None)
    lng_col = next((by_key[k] for k in _LNG_KEYS if k in by_key), None)
    if lat_col and lng_col:
        return lat_col, lng_col
    return None


def _build_geo_map_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    columns: list[DataTableColumn],
    lat_col: DataTableColumn,
    lng_col: DataTableColumn,
    cat_col: DataTableColumn | None,
) -> ChartPayload:
    """Build a scattergeo Plotly map with markers sized by capacity."""
    by_key = {c.key: c for c in columns}
    cap_col = next(
        (by_key[k] for k in ("total_capacity_mw", "capacity_mw", "total_capacity_maximum_mw") if k in by_key),
        None,
    )

    lats = [r.get(lat_col.key) for r in rows]
    lons = [r.get(lng_col.key) for r in rows]
    labels = [
        str(r.get(cat_col.key, "")) if cat_col else str(i)
        for i, r in enumerate(rows)
    ]

    # Size markers proportionally to capacity; default 14px if no capacity col.
    sizes: list[int] = [14] * len(rows)
    cap_values: list[Any] | None = None
    if cap_col:
        raw_caps = [r.get(cap_col.key) for r in rows]
        valid_caps = [c for c in raw_caps if _is_numeric(c) and c > 0]
        if valid_caps:
            max_cap = max(valid_caps)
            sizes = [int(10 + ((r.get(cap_col.key) or 0) / max_cap) * 25) for r in rows]
            cap_values = raw_caps

    state_col = by_key.get("state")
    customdata = [
        [r.get("state", ""), r.get(cap_col.key, "") if cap_col else ""]
        for r in rows
    ]
    hover_parts = ["<b>%{text}</b>"]
    if cap_col:
        hover_parts.append(f"{cap_col.label}: %{{customdata[1]:.0f}} MW")
    if state_col:
        hover_parts.append("State: %{customdata[0]}")
    hover_parts.append("<extra></extra>")

    marker: dict[str, Any] = {"size": sizes, "sizemode": "diameter"}
    if cap_values is not None:
        marker.update({
            "color": cap_values,
            "colorscale": [[0, "#1a8a5a"], [0.5, "#f4b942"], [1, "#1b6ca8"]],
            "showscale": True,
            "colorbar": {"title": "Capacity<br>(MW)", "thickness": 12, "len": 0.6},
        })
    else:
        marker["color"] = "#1a8a5a"

    trace: dict[str, Any] = {
        "type": "scattergeo",
        "lat": lats,
        "lon": lons,
        "text": labels,
        "mode": "markers+text",
        "textposition": "top center",
        "textfont": {"size": 10, "color": "#333333"},
        "marker": marker,
        "customdata": customdata,
        "hovertemplate": "<br>".join(hover_parts),
    }

    layout: dict[str, Any] = {
        "geo": {
            "projection": {"type": "mercator"},
            "showland": True,
            "landcolor": "#f5f5f0",
            "showocean": True,
            "oceancolor": "#ddeef9",
            "showcoastlines": True,
            "coastlinecolor": "#aaaaaa",
            "showcountries": True,
            "countrycolor": "#cccccc",
            "fitbounds": "locations",
            "resolution": 50,
        },
        "hovermode": "closest",
        "height": 420,
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
        "paper_bgcolor": "#ffffff",
    }

    title = "Facility Locations & Capacity (MW)" if cap_col else "Facility Locations"
    return ChartPayload(
        chart_type="scatter_geo",
        title=title,
        plotly_spec={"data": [trace], "layout": layout},
        source_metric_key=metric_key,
    )


def _detect_numeric_columns(columns: list[DataTableColumn]) -> list[DataTableColumn]:
    """Return numeric columns suitable for charting (metadata and geo coords filtered out)."""
    return [
        c for c in columns
        if c.type in ("number", "integer")
        and c.key not in _METADATA_NUMERIC_KEYS
        and c.key not in _GEO_KEYS
    ]


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


def _axis_title(col: DataTableColumn) -> str:
    return f"{col.label} ({col.unit})" if col.unit else col.label


def _base_layout(title: str, *, bottom_margin: int = 50) -> dict[str, Any]:
    return {
        "title": title,
        "margin": {"l": 50, "r": 20, "t": 40, "b": bottom_margin},
        "colorway": list(_COLOR_PALETTE),
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
    }


def _needs_horizontal_bar(categories: list[str]) -> bool:
    if len(categories) > _HORIZONTAL_CAT_COUNT:
        return True
    return any(len(c) > _HORIZONTAL_LABEL_LEN for c in categories)


def _build_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    columns: list[DataTableColumn],
    user_query: str | None = None,
) -> ChartPayload | None:
    if not rows or not columns:
        return None

    if _has_text_heavy_column(rows, columns):
        return None

    numeric_cols = _detect_numeric_columns(columns)
    time_col = _detect_time_column(columns)
    cat_col = _detect_category_column(columns)

    # --- Geo map: lat + lng present (highest priority) ----------------------
    geo_pair = _detect_geo_columns(columns)
    if geo_pair is not None:
        lat_col, lng_col = geo_pair
        return _build_geo_map_chart(metric_key, rows, columns, lat_col, lng_col, cat_col)

    # Multi-day per-facility data (report_date present AND >1 unique date)
    # should render as a time-series (one line per facility), not a scatter.
    # Single-date multi-facility data correlates naturally (weather vs energy)
    # and still goes through the scatter branch below.
    multi_date = False
    if time_col is not None:
        seen: set[Any] = set()
        for r in rows:
            seen.add(r.get(time_col.key))
            if len(seen) > 1:
                multi_date = True
                break

    # --- Correlation scatter -------------------------------------------------
    # Check BEFORE time-series so that weather-impact data that happens to
    # include an `energy_date` column still renders as a per-facility X-Y
    # scatter rather than a time-series line per facility.
    if not multi_date and cat_col is not None and numeric_cols:
        candidate_y = [c for c in numeric_cols if c.key != cat_col.key]
        correlation = _detect_correlation_axes(candidate_y, user_query)
        if correlation is not None:
            x_col, y_col = correlation
            return _build_correlation_scatter(metric_key, rows, cat_col, x_col, y_col)

    # --- Time series: datetime + ≥1 numeric ---------------------------------
    if time_col is not None and numeric_cols:
        return _build_timeseries_chart(metric_key, rows, time_col, cat_col, numeric_cols)

    if not numeric_cols:
        return None

    # --- Bar: 1 categorical + 1+ numeric -----------------------------------
    if cat_col is not None:
        y_cols = [c for c in numeric_cols if c.key != cat_col.key]
        if not y_cols:
            return None
        if len(y_cols) >= 2:
            return _build_grouped_bar_chart(metric_key, rows, cat_col, y_cols[:3])
        return _build_bar_chart(metric_key, rows, cat_col, y_cols[0])

    # --- Scatter: exactly 2 numeric (no cat / no time) ---------------------
    if len(numeric_cols) >= 2:
        return _build_scatter_chart(metric_key, rows, numeric_cols[0], numeric_cols[1])

    # --- Histogram: 1 numeric only -----------------------------------------
    if len(numeric_cols) == 1 and len(rows) >= _HISTOGRAM_MIN_ROWS:
        return _build_histogram_chart(metric_key, rows, numeric_cols[0])

    return None


def _build_timeseries_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    time_col: DataTableColumn,
    cat_col: DataTableColumn | None,
    numeric_cols: list[DataTableColumn],
) -> ChartPayload | None:
    y_col = next((c for c in numeric_cols if c.key != time_col.key), None)
    if y_col is None:
        return None
    title = f"{y_col.label} over {time_col.label}"

    if cat_col is not None and cat_col.key != time_col.key:
        categories: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(cat_col.key, "all"))
            categories.setdefault(key, []).append(row)
        # Cap to top N series by total y-value (Vanna caps at 5).
        if len(categories) > _MAX_LINE_SERIES:
            ranked = sorted(
                categories.items(),
                key=lambda kv: sum(r.get(y_col.key) or 0 for r in kv[1] if _is_numeric(r.get(y_col.key))),
                reverse=True,
            )[: _MAX_LINE_SERIES]
            categories = dict(ranked)
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

    layout = _base_layout(title)
    layout["xaxis"] = {"title": time_col.label}
    layout["yaxis"] = {"title": _axis_title(y_col)}
    return ChartPayload(
        chart_type="line",
        title=title,
        plotly_spec={"data": traces, "layout": layout},
        source_metric_key=metric_key,
    )


def _build_bar_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    cat_col: DataTableColumn,
    y_col: DataTableColumn,
) -> ChartPayload:
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get(y_col.key) if _is_numeric(r.get(y_col.key)) else float("-inf")),
        reverse=True,
    )
    truncated = sorted_rows[:_MAX_BAR_CATEGORIES]
    categories = [str(r.get(cat_col.key, "")) for r in truncated]
    values = [r.get(y_col.key) for r in truncated]
    horizontal = _needs_horizontal_bar(categories)

    title = f"{y_col.label} by {cat_col.label}"
    if len(sorted_rows) > _MAX_BAR_CATEGORIES:
        title = f"Top {_MAX_BAR_CATEGORIES} — {title}"

    if horizontal:
        # Reverse so the largest sits at the top of a horizontal bar.
        categories = list(reversed(categories))
        values = list(reversed(values))
        trace = {
            "y": categories,
            "x": values,
            "type": "bar",
            "orientation": "h",
            "name": y_col.label,
        }
        layout = _base_layout(title, bottom_margin=50)
        layout["xaxis"] = {"title": _axis_title(y_col)}
        layout["yaxis"] = {"title": cat_col.label, "automargin": True}
    else:
        trace = {
            "x": categories,
            "y": values,
            "type": "bar",
            "name": y_col.label,
        }
        layout = _base_layout(title, bottom_margin=80)
        layout["xaxis"] = {"title": cat_col.label}
        layout["yaxis"] = {"title": _axis_title(y_col)}

    return ChartPayload(
        chart_type="bar",
        title=title,
        plotly_spec={"data": [trace], "layout": layout},
        source_metric_key=metric_key,
    )


def _build_grouped_bar_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    cat_col: DataTableColumn,
    y_cols: list[DataTableColumn],
) -> ChartPayload:
    # Only plot numeric columns that share the same unit as the primary.
    # Mixed-unit grouped bars (MWh + %) are visually misleading and Plotly's
    # dual-axis + barmode:group has well-known rendering quirks. The table
    # payload still carries every column for the user.
    primary = y_cols[0]
    primary_unit = primary.unit
    rendered_cols = [c for c in y_cols if c.unit == primary_unit]

    # If only one column ends up comparable, fall back to single bar chart.
    if len(rendered_cols) == 1:
        return _build_bar_chart(metric_key, rows, cat_col, rendered_cols[0])

    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get(primary.key) if _is_numeric(r.get(primary.key)) else float("-inf")),
        reverse=True,
    )[:_MAX_BAR_CATEGORIES]
    categories = [str(r.get(cat_col.key, "")) for r in sorted_rows]

    traces = [
        {
            "x": categories,
            "y": [r.get(col.key) for r in sorted_rows],
            "type": "bar",
            "name": _axis_title(col),
        }
        for col in rendered_cols
    ]

    labels = " & ".join(c.label for c in rendered_cols)
    title = f"{labels} by {cat_col.label}"
    layout = _base_layout(title, bottom_margin=80)
    layout["xaxis"] = {"title": cat_col.label}
    layout["yaxis"] = {"title": primary_unit or primary.label}
    layout["barmode"] = "group"
    layout["legend"] = {"orientation": "h", "y": -0.25}

    return ChartPayload(
        chart_type="bar",
        title=title,
        plotly_spec={"data": traces, "layout": layout},
        source_metric_key=metric_key,
    )


# Keyword patterns for correlation axis detection. Order within each tuple
# is priority: first-match wins. Kept as substrings so we match both
# `avg_temperature_c` and `avg_temperature`, both `avg_capacity_factor` and
# `capacity_factor_pct`, etc. — DB schema naming drifts over time.
_CORRELATION_X_PATTERNS = (
    "temperature", "humidity", "cloud_cover", "cloud",
    "shortwave", "radiation", "irradiance",
    "wind_speed", "wind",
    "aqi", "pm2_5", "pm10",
    "precipitation",
)

# Maps query keywords → X-axis column patterns to try FIRST.
# When a user asks "how does cloud cover affect PR?", cloud_cover must be
# prioritised over temperature (which appears first in the default list above).
_QUERY_TO_X_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloud",        ("cloud_cover", "cloud", "avg_cloud")),
    ("mây",          ("cloud_cover", "cloud", "avg_cloud")),
    ("humidity",     ("humidity", "avg_humidity")),
    ("độ ẩm",        ("humidity", "avg_humidity")),
    ("wind",         ("wind_speed", "wind", "avg_wind")),
    ("gió",          ("wind_speed", "wind", "avg_wind")),
    ("tốc độ gió",   ("wind_speed", "wind", "avg_wind")),
    ("hướng gió",    ("wind_speed", "wind", "avg_wind")),
    ("radiation",    ("shortwave", "radiation", "irradiance", "avg_shortwave")),
    ("bức xạ",       ("shortwave", "radiation", "irradiance")),
    ("precipitation", ("precipitation", "rain")),
    ("mưa",          ("precipitation", "rain")),
    ("aqi",          ("aqi", "pm2_5", "pm10", "aqi_value")),
    ("temperature",  ("temperature", "avg_temperature")),
    ("nhiệt độ",     ("temperature", "avg_temperature")),
)


def _reorder_x_patterns_for_query(user_query: str | None) -> tuple[str, ...]:
    """Return _CORRELATION_X_PATTERNS with query-relevant entries promoted to front."""
    if not user_query:
        return _CORRELATION_X_PATTERNS
    q = user_query.lower()
    promoted: list[str] = []
    for keyword, priority_pats in _QUERY_TO_X_PRIORITY:
        if keyword in q:
            for p in priority_pats:
                if p not in promoted:
                    promoted.append(p)
    if not promoted:
        return _CORRELATION_X_PATTERNS
    remaining = [p for p in _CORRELATION_X_PATTERNS if p not in promoted]
    return tuple(promoted) + tuple(remaining)


# Same idea for the Y-axis: when user explicitly names the Y metric (e.g.
# "năng lượng sản xuất"), energy columns must be preferred over capacity_factor
# which appears earlier in the default _CORRELATION_Y_PATTERNS order.
_QUERY_TO_Y_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Explicit energy-output intent → prefer energy cols over CF/PR
    ("energy production", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    ("năng lượng sản xuất", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    ("sản lượng điện", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    ("sản lượng", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    ("energy output", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    ("energy generation", ("total_energy", "energy_mwh", "avg_energy", "energy")),
    # Efficiency / performance intent (Vietnamese "hiệu suất") → PR first, then CF
    ("hiệu suất", ("performance_ratio", "pr_ratio", "pr_pct", "capacity_factor", "weighted_capacity_factor")),
    ("phát điện", ("performance_ratio", "pr_ratio", "pr_pct", "capacity_factor", "energy_mwh", "avg_energy", "energy")),
    ("performance ratio", ("performance_ratio", "pr_ratio", "pr_pct")),
    ("performance_ratio", ("performance_ratio", "pr_ratio", "pr_pct")),
    ("pr ", ("performance_ratio", "pr_ratio", "pr_pct")),
    (" pr", ("performance_ratio", "pr_ratio", "pr_pct")),
    ("capacity factor", ("capacity_factor",)),
    ("capacity_factor", ("capacity_factor",)),
)


def _reorder_y_patterns_for_query(user_query: str | None) -> tuple[str, ...]:
    """Return _CORRELATION_Y_PATTERNS with query-relevant entries promoted to front."""
    if not user_query:
        return _CORRELATION_Y_PATTERNS
    q = user_query.lower()
    promoted: list[str] = []
    for keyword, priority_pats in _QUERY_TO_Y_PRIORITY:
        if keyword in q:
            for p in priority_pats:
                if p not in promoted:
                    promoted.append(p)
    if not promoted:
        return _CORRELATION_Y_PATTERNS
    remaining = [p for p in _CORRELATION_Y_PATTERNS if p not in promoted]
    return tuple(promoted) + tuple(remaining)
# Performance-like Y columns take precedence over raw-energy ones because a
# correlation question is usually about efficiency, not absolute output.
_CORRELATION_Y_PATTERNS = (
    "performance_ratio", "pr_ratio", "pr_pct",
    # rad/energy ratio is the closest proxy for PR available in our weather
    # mart; prefer it over capacity factor because the user usually asks PR.
    "radiation_to_energy", "rad_energy_ratio", "rad_to_energy",
    "capacity_factor",
    "energy_per_hour", "mwh_per_hour",
    "total_energy", "energy_mwh", "avg_energy", "energy",
)


def _find_by_pattern(
    cols: list[DataTableColumn], patterns: tuple[str, ...]
) -> DataTableColumn | None:
    """Return first column whose key contains any pattern, in pattern order."""
    for pat in patterns:
        for c in cols:
            if pat in c.key.lower():
                return c
    return None


def _detect_correlation_axes(
    y_cols: list[DataTableColumn],
    user_query: str | None = None,
) -> tuple[DataTableColumn, DataTableColumn] | None:
    """If the numeric columns include a weather-like X and a performance Y,
    return (x_col, y_col) for a scatter plot. Otherwise None.

    user_query is used to reorder X-axis patterns so that the column the user
    explicitly mentioned (cloud cover, AQI, humidity, …) is tried before the
    default first-match pattern (temperature).
    """
    x_patterns = _reorder_x_patterns_for_query(user_query)
    x_col = _find_by_pattern(y_cols, x_patterns)
    if x_col is None:
        return None
    y_cols_minus_x = [c for c in y_cols if c.key != x_col.key]
    y_patterns = _reorder_y_patterns_for_query(user_query)
    y_col = _find_by_pattern(y_cols_minus_x, y_patterns)
    if y_col is None:
        return None
    return x_col, y_col


def _aggregate_by_category(
    rows: list[dict[str, Any]],
    cat_key: str,
    value_keys: list[str],
) -> list[dict[str, Any]]:
    """Group rows by category key and return one row per group with MEAN of
    numeric values. Used to collapse multi-row-per-facility correlation data
    into one point per facility for the scatter plot.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        key = str(r.get(cat_key) or "")
        if not key:
            continue
        slot = buckets.setdefault(key, {k: [] for k in value_keys})
        for vk in value_keys:
            v = r.get(vk)
            if _is_numeric(v):
                slot[vk].append(float(v))
    out: list[dict[str, Any]] = []
    for cat, values in buckets.items():
        agg: dict[str, Any] = {cat_key: cat}
        for vk, vs in values.items():
            if vs:
                agg[vk] = sum(vs) / len(vs)
        out.append(agg)
    return out


def _build_correlation_scatter(
    metric_key: str,
    rows: list[dict[str, Any]],
    cat_col: DataTableColumn,
    x_col: DataTableColumn,
    y_col: DataTableColumn,
) -> ChartPayload:
    """Scatter plot where each category (facility) is one marker labelled by
    its name. Multi-row-per-facility inputs are averaged.
    """
    aggregated = _aggregate_by_category(
        rows, cat_col.key, [x_col.key, y_col.key]
    )
    # Filter points with valid x AND y.
    points = [r for r in aggregated if _is_numeric(r.get(x_col.key)) and _is_numeric(r.get(y_col.key))]

    title = f"{y_col.label} vs {x_col.label}"
    trace = {
        "x": [r[x_col.key] for r in points],
        "y": [r[y_col.key] for r in points],
        "text": [r[cat_col.key] for r in points],
        "mode": "markers+text",
        "type": "scatter",
        "name": title,
        "textposition": "top center",
        "marker": {"size": 12, "opacity": 0.85},
    }
    layout = _base_layout(title)
    layout["xaxis"] = {"title": _axis_title(x_col)}
    layout["yaxis"] = {"title": _axis_title(y_col)}
    return ChartPayload(
        chart_type="scatter",
        title=title,
        plotly_spec={"data": [trace], "layout": layout},
        source_metric_key=metric_key,
    )


def _build_scatter_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    x_col: DataTableColumn,
    y_col: DataTableColumn,
) -> ChartPayload:
    title = f"{y_col.label} vs {x_col.label}"
    trace = {
        "x": [r.get(x_col.key) for r in rows],
        "y": [r.get(y_col.key) for r in rows],
        "type": "scatter",
        "mode": "markers",
        "name": title,
        "marker": {"size": 8, "opacity": 0.75},
    }
    layout = _base_layout(title)
    layout["xaxis"] = {"title": _axis_title(x_col)}
    layout["yaxis"] = {"title": _axis_title(y_col)}
    return ChartPayload(
        chart_type="scatter",
        title=title,
        plotly_spec={"data": [trace], "layout": layout},
        source_metric_key=metric_key,
    )


def _build_histogram_chart(
    metric_key: str,
    rows: list[dict[str, Any]],
    y_col: DataTableColumn,
) -> ChartPayload:
    title = f"Distribution of {y_col.label}"
    values = [r.get(y_col.key) for r in rows if _is_numeric(r.get(y_col.key))]
    trace = {
        "x": values,
        "type": "histogram",
        "name": y_col.label,
        "marker": {"color": _COLOR_PALETTE[0]},
    }
    layout = _base_layout(title)
    layout["xaxis"] = {"title": _axis_title(y_col)}
    layout["yaxis"] = {"title": "Count"}
    layout["bargap"] = 0.05
    return ChartPayload(
        chart_type="histogram",
        title=title,
        plotly_spec={"data": [trace], "layout": layout},
        source_metric_key=metric_key,
    )


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
        user_query: str | None = None,
    ) -> tuple[DataTablePayload | None, ChartPayload | None, KpiCardsPayload | None]:
        if not isinstance(metrics, dict) or not metrics:
            return None, None, None

        table: DataTablePayload | None = None
        chart: ChartPayload | None = None

        candidate = _pick_best_list(metrics)
        if candidate is not None:
            metric_key, rows = candidate
            try:
                title_hint = _title_for_topic(topic, metric_key, rows)
                table = _build_data_table(metric_key, rows, title_hint=title_hint)
                chart = _build_chart(metric_key, rows, table.columns, user_query)
                # When the chart is a correlation scatter (PR vs cloud cover,
                # AQI vs energy, etc.), project the table to just the columns
                # that answer the question: facility + date + X + Y. Everything
                # else is noise.
                if chart is not None and chart.chart_type == "scatter" and table is not None:
                    table = _project_table_for_scatter(table, rows, chart)
            except Exception as exc:
                logger.warning("chart_spec_build_failed key=%s error=%s", metric_key, exc)
                table = None
                chart = None

        kpi = _build_kpi_cards(metrics)
        return table, chart, kpi


def _project_table_for_scatter(
    table: DataTablePayload,
    rows: list[dict[str, Any]],
    chart: ChartPayload,
) -> DataTablePayload:
    """Keep only facility/date + the scatter's X/Y axis columns in the table."""
    layout = chart.plotly_spec.get("layout") or {}
    x_title = (layout.get("xaxis") or {}).get("title") or ""
    y_title = (layout.get("yaxis") or {}).get("title") or ""

    def _key_from_title(title: str) -> str | None:
        # Strip unit suffix like " (°C)" to get the bare label.
        bare = re.sub(r"\s*\(.*?\)\s*$", "", str(title)).strip()
        for col in table.columns:
            if col.label == bare:
                return col.key
        return None

    x_key = _key_from_title(x_title)
    y_key = _key_from_title(y_title)
    if not x_key or not y_key:
        return table

    keep_keys: set[str] = {x_key, y_key}
    # Keep the category and any date column for context.
    for col in table.columns:
        if col.key in _FACILITY_LIKE_KEYS or col.key in _TIME_COLUMNS:
            keep_keys.add(col.key)

    projected_cols = [c for c in table.columns if c.key in keep_keys]
    projected_rows = [
        {k: row.get(k) for k in keep_keys if k in row}
        for row in table.rows
    ]
    # Use chart title so the table label matches the scatter relationship
    # (e.g. "Performance Ratio vs Avg Cloud Cover" instead of a generic title).
    scatter_title = chart.title or table.title
    return DataTablePayload(
        title=scatter_title,
        columns=projected_cols,
        rows=projected_rows,
        row_count=len(projected_rows),
    )


_FACILITY_LIKE_KEYS = {"facility", "facility_id", "facility_name", "station", "station_name"}
_TIME_LIKE_KEYS = {"hour", "hr", "date", "date_hour", "timestamp", "report_date", "reading_date", "ts", "aqi_date", "energy_date", "forecast_date", "weather_date"}


def _title_for_topic(
    topic: str | None,
    metric_key: str,
    rows: list[dict[str, Any]] | None = None,
) -> str:
    """Derive a table title by inspecting the actual data shape.

    Precedence:
      1. Time-series with facility+metric → "Hourly/Daily X over time".
      2. Per-facility breakdown with recognisable metrics → content-aware.
      3. Known topic → topic title.
      4. Fallback → humanised metric_key.
    """
    sample_keys: set[str] = set()
    if rows:
        for row in rows[:5]:
            if isinstance(row, dict):
                sample_keys.update(row.keys())

    has_facility_col = bool(sample_keys & _FACILITY_LIKE_KEYS)
    has_time_col = bool(sample_keys & _TIME_LIKE_KEYS)
    has_energy = "energy_mwh" in sample_keys
    has_capacity = "capacity_factor_pct" in sample_keys
    has_actual_forecast = "actual_mwh" in sample_keys and "forecast_mwh" in sample_keys
    has_perf_ratio = "performance_ratio_pct" in sample_keys

    # 1. Time-series takes precedence — the breakdown is over time, not facility.
    if has_time_col:
        is_hourly = "hour" in sample_keys or "hr" in sample_keys
        period = "Hourly" if is_hourly else "Daily"
        if has_energy and has_capacity:
            return f"{period} energy & capacity"
        if has_energy:
            return f"{period} energy generation"
        if has_capacity:
            return f"{period} capacity factor"

    # 2. Per-facility breakdown.
    if has_facility_col:
        if has_actual_forecast:
            return "Actual vs forecast energy"
        if has_capacity and not has_energy:
            return "Capacity factor by facility"
        if has_energy and not has_capacity:
            return "Energy production by facility"
        if has_energy and has_capacity:
            return "Energy & capacity by facility"
        if has_perf_ratio:
            return "Performance ratio by facility"

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
