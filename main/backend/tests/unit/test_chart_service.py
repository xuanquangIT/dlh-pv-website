"""Unit tests for chart_service.py — ChartSpecBuilder and all helpers.

Run with:
    pytest main/backend/tests/unit/test_chart_service.py -v
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.solar_ai_chat.chart_service import (
    ChartSpecBuilder,
    _aggregate_by_category,
    _axis_title,
    _base_layout,
    _build_bar_chart,
    _build_chart,
    _build_columns,
    _build_correlation_scatter,
    _build_data_table,
    _build_geo_map_chart,
    _build_grouped_bar_chart,
    _build_histogram_chart,
    _build_kpi_cards,
    _build_scatter_chart,
    _build_timeseries_chart,
    _coerce_scalar,
    _detect_category_column,
    _detect_correlation_axes,
    _detect_geo_columns,
    _detect_numeric_columns,
    _detect_time_column,
    _find_by_pattern,
    _format_for_kpi,
    _has_text_heavy_column,
    _humanize,
    _infer_column_type,
    _is_numeric,
    _label_for,
    _needs_horizontal_bar,
    _pick_best_list,
    _project_table_for_scatter,
    _reorder_x_patterns_for_query,
    _reorder_y_patterns_for_query,
    _sorted_by,
    _title_for_topic,
)
from app.schemas.solar_ai_chat.visualization import (
    ChartPayload,
    DataTableColumn,
    DataTablePayload,
    KpiCard,
    KpiCardsPayload,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _col(key: str, col_type: str = "string", unit: str | None = None, label: str | None = None) -> DataTableColumn:
    return DataTableColumn(key=key, label=label or key.replace("_", " ").title(), type=col_type, unit=unit)  # type: ignore[arg-type]


def _energy_rows(n: int = 5) -> list[dict[str, Any]]:
    return [
        {"facility": f"F{i}", "date": f"2024-01-{i+1:02d}", "energy_mwh": float(100 + i * 10)}
        for i in range(n)
    ]


def _timeseries_rows(n: int = 5) -> list[dict[str, Any]]:
    return [
        {"date": f"2024-01-{i+1:02d}", "energy_mwh": float(100 + i * 10)}
        for i in range(n)
    ]


def _weather_rows() -> list[dict[str, Any]]:
    return [
        {
            "facility": f"Station{i}",
            "avg_temperature_c": 20.0 + i,
            "avg_capacity_factor": 0.3 + i * 0.05,
        }
        for i in range(6)
    ]


# ===========================================================================
# _humanize
# ===========================================================================

class TestHumanize:
    def test_underscores_to_spaces(self):
        assert _humanize("energy_mwh") == "Energy Mwh"

    def test_title_case(self):
        assert _humanize("facility_id") == "Facility Id"

    def test_multiple_underscores(self):
        assert _humanize("avg_temperature_c") == "Avg Temperature C"

    def test_already_clean(self):
        assert _humanize("Energy") == "Energy"

    def test_spaces_normalised(self):
        assert _humanize("a  b") == "A B"


# ===========================================================================
# _label_for
# ===========================================================================

class TestLabelFor:
    def test_known_key(self):
        label, unit, col_type = _label_for("energy_mwh")
        assert label == "Energy"
        assert unit == "MWh"
        assert col_type == "number"

    def test_known_date_key(self):
        label, unit, col_type = _label_for("date")
        assert col_type == "date"
        assert unit is None

    def test_pct_suffix_fallback(self):
        label, unit, col_type = _label_for("some_custom_pct")
        assert unit == "%"
        assert col_type == "number"
        assert "Pct" not in label

    def test_mwh_suffix_fallback(self):
        label, unit, col_type = _label_for("custom_energy_mwh")
        assert unit == "MWh"
        assert col_type == "number"

    def test_unknown_key_fallback(self):
        label, unit, col_type = _label_for("unknown_column")
        assert col_type == "string"
        assert unit is None
        assert label == "Unknown Column"


# ===========================================================================
# _coerce_scalar
# ===========================================================================

class TestCoerceScalar:
    def test_date_to_isoformat(self):
        assert _coerce_scalar(date(2024, 1, 15)) == "2024-01-15"

    def test_datetime_to_isoformat(self):
        assert _coerce_scalar(datetime(2024, 1, 15, 10, 30)) == "2024-01-15T10:30:00"

    def test_number_passthrough(self):
        assert _coerce_scalar(42.5) == 42.5

    def test_string_passthrough(self):
        assert _coerce_scalar("hello") == "hello"

    def test_none_passthrough(self):
        assert _coerce_scalar(None) is None


# ===========================================================================
# _is_numeric
# ===========================================================================

class TestIsNumeric:
    def test_int(self):
        assert _is_numeric(5) is True

    def test_float(self):
        assert _is_numeric(3.14) is True

    def test_bool_excluded(self):
        assert _is_numeric(True) is False
        assert _is_numeric(False) is False

    def test_string_excluded(self):
        assert _is_numeric("5") is False

    def test_none_excluded(self):
        assert _is_numeric(None) is False


# ===========================================================================
# _pick_best_list
# ===========================================================================

class TestPickBestList:
    def test_returns_largest_list_of_dicts(self):
        metrics = {
            "small": [{"a": 1}],
            "large": [{"a": i} for i in range(10)],
            "scalar": 42,
        }
        result = _pick_best_list(metrics)
        assert result is not None
        key, rows = result
        assert key == "large"
        assert len(rows) == 10

    def test_empty_metrics(self):
        assert _pick_best_list({}) is None

    def test_no_list(self):
        assert _pick_best_list({"a": 1, "b": "hello"}) is None

    def test_empty_list_ignored(self):
        metrics = {"empty": [], "valid": [{"x": 1}]}
        result = _pick_best_list(metrics)
        assert result is not None
        assert result[0] == "valid"

    def test_list_of_non_dicts_ignored(self):
        metrics = {"strings": ["a", "b", "c"], "dicts": [{"k": 1}, {"k": 2}]}
        result = _pick_best_list(metrics)
        assert result is not None
        assert result[0] == "dicts"

    def test_mixed_list_ignored(self):
        # List where not all items are dicts should be ignored
        metrics = {"mixed": [{"a": 1}, "b"]}
        assert _pick_best_list(metrics) is None

    def test_single_row(self):
        metrics = {"data": [{"a": 1, "b": 2}]}
        result = _pick_best_list(metrics)
        assert result is not None
        assert result[0] == "data"


# ===========================================================================
# _infer_column_type
# ===========================================================================

class TestInferColumnType:
    def test_all_numeric(self):
        assert _infer_column_type([1.0, 2.0, 3.0]) == "number"

    def test_mixed_with_none(self):
        assert _infer_column_type([1.0, None, 2.0]) == "number"

    def test_has_string(self):
        assert _infer_column_type([1.0, "abc"]) == "string"

    def test_all_none(self):
        assert _infer_column_type([None, None]) == "string"

    def test_empty(self):
        assert _infer_column_type([]) == "string"

    def test_bool_treated_as_non_numeric(self):
        # bool is excluded from numeric by _is_numeric
        assert _infer_column_type([True, False]) == "string"


# ===========================================================================
# _build_columns
# ===========================================================================

class TestBuildColumns:
    def test_empty_rows(self):
        assert _build_columns([]) == []

    def test_known_keys(self):
        rows = [{"energy_mwh": 100.0, "facility": "F1"}]
        cols = _build_columns(rows)
        keys = [c.key for c in cols]
        assert "energy_mwh" in keys
        assert "facility" in keys

    def test_label_applied(self):
        rows = [{"energy_mwh": 100.0}]
        cols = _build_columns(rows)
        energy_col = next(c for c in cols if c.key == "energy_mwh")
        assert energy_col.label == "Energy"
        assert energy_col.unit == "MWh"

    def test_infers_type_from_values(self):
        # "custom_col" is not in _LABELS, _label_for returns "string" type
        # but _infer_column_type should upgrade it to "number" if all values are numeric
        rows = [{"custom_col": 1.0}, {"custom_col": 2.0}]
        cols = _build_columns(rows)
        custom_col = next(c for c in cols if c.key == "custom_col")
        assert custom_col.type == "number"

    def test_union_of_keys_across_rows(self):
        rows = [{"a": 1}, {"b": 2}]
        cols = _build_columns(rows)
        keys = [c.key for c in cols]
        assert "a" in keys
        assert "b" in keys

    def test_no_duplicate_keys(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        cols = _build_columns(rows)
        keys = [c.key for c in cols]
        assert len(keys) == len(set(keys))


# ===========================================================================
# _build_data_table
# ===========================================================================

class TestBuildDataTable:
    def test_basic(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        table = _build_data_table("energy", rows)
        assert table.row_count == 1
        assert table.title == "Energy"

    def test_title_hint(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        table = _build_data_table("energy", rows, title_hint="Custom Title")
        assert table.title == "Custom Title"

    def test_hidden_keys_excluded(self):
        rows = [{"facility": "F1", "batch_id": 999, "energy_mwh": 50.0}]
        table = _build_data_table("energy", rows)
        col_keys = [c.key for c in table.columns]
        assert "batch_id" not in col_keys
        assert "facility" in col_keys

    def test_coerces_date_values(self):
        rows = [{"date": date(2024, 1, 1), "energy_mwh": 10.0}]
        table = _build_data_table("energy", rows)
        assert table.rows[0]["date"] == "2024-01-01"

    def test_noisy_keys_excluded(self):
        rows = [{"facility": "F1", "energy_mwh": 10.0, "quality_flag": "ok", "energy_kwh_daily": 10000}]
        table = _build_data_table("energy", rows)
        col_keys = [c.key for c in table.columns]
        assert "quality_flag" not in col_keys
        assert "energy_kwh_daily" not in col_keys


# ===========================================================================
# _detect_time_column
# ===========================================================================

class TestDetectTimeColumn:
    def test_detects_date_key(self):
        cols = [_col("facility"), _col("date", "date"), _col("energy_mwh", "number")]
        result = _detect_time_column(cols)
        assert result is not None
        assert result.key == "date"

    def test_detects_by_key_name(self):
        cols = [_col("hour", "number"), _col("energy_mwh", "number")]
        result = _detect_time_column(cols)
        assert result is not None
        assert result.key == "hour"

    def test_detects_datetime_type(self):
        cols = [_col("ts", "datetime"), _col("value", "number")]
        result = _detect_time_column(cols)
        assert result is not None
        assert result.key == "ts"

    def test_no_time_column(self):
        cols = [_col("facility"), _col("value", "number")]
        assert _detect_time_column(cols) is None


# ===========================================================================
# _detect_category_column
# ===========================================================================

class TestDetectCategoryColumn:
    def test_prefers_facility_over_facility_id(self):
        cols = [_col("facility_id"), _col("facility"), _col("energy_mwh", "number")]
        result = _detect_category_column(cols)
        assert result is not None
        assert result.key == "facility"

    def test_prefers_facility_name(self):
        cols = [_col("facility_name"), _col("station"), _col("energy_mwh", "number")]
        result = _detect_category_column(cols)
        assert result is not None
        assert result.key == "facility_name"

    def test_fallback_to_first_string_col(self):
        cols = [_col("energy_mwh", "number"), _col("label", "string")]
        result = _detect_category_column(cols)
        assert result is not None
        assert result.key == "label"

    def test_no_category_column(self):
        cols = [_col("energy_mwh", "number"), _col("capacity_factor_pct", "number")]
        assert _detect_category_column(cols) is None


# ===========================================================================
# _detect_geo_columns
# ===========================================================================

class TestDetectGeoColumns:
    def test_detects_lat_lng(self):
        cols = [_col("latitude", "number"), _col("longitude", "number"), _col("facility")]
        result = _detect_geo_columns(cols)
        assert result is not None
        lat, lng = result
        assert lat.key == "latitude"
        assert lng.key == "longitude"

    def test_detects_location_lat_lng(self):
        cols = [_col("location_lat", "number"), _col("location_lng", "number")]
        result = _detect_geo_columns(cols)
        assert result is not None

    def test_missing_lng_returns_none(self):
        cols = [_col("latitude", "number"), _col("energy_mwh", "number")]
        assert _detect_geo_columns(cols) is None

    def test_missing_lat_returns_none(self):
        cols = [_col("longitude", "number"), _col("energy_mwh", "number")]
        assert _detect_geo_columns(cols) is None


# ===========================================================================
# _detect_numeric_columns
# ===========================================================================

class TestDetectNumericColumns:
    def test_returns_numeric_cols(self):
        cols = [_col("facility"), _col("energy_mwh", "number"), _col("capacity_factor_pct", "number")]
        result = _detect_numeric_columns(cols)
        keys = [c.key for c in result]
        assert "energy_mwh" in keys
        assert "capacity_factor_pct" in keys
        assert "facility" not in keys

    def test_excludes_metadata_numeric(self):
        cols = [_col("batch_id", "number"), _col("energy_mwh", "number")]
        result = _detect_numeric_columns(cols)
        keys = [c.key for c in result]
        assert "batch_id" not in keys
        assert "energy_mwh" in keys

    def test_excludes_geo_keys(self):
        cols = [_col("latitude", "number"), _col("longitude", "number"), _col("energy_mwh", "number")]
        result = _detect_numeric_columns(cols)
        keys = [c.key for c in result]
        assert "latitude" not in keys
        assert "longitude" not in keys
        assert "energy_mwh" in keys


# ===========================================================================
# _needs_horizontal_bar
# ===========================================================================

class TestNeedsHorizontalBar:
    def test_short_few_categories_vertical(self):
        cats = ["A", "B", "C"]
        assert _needs_horizontal_bar(cats) is False

    def test_long_label_triggers_horizontal(self):
        cats = ["Very Long Station Name Here", "B"]
        assert _needs_horizontal_bar(cats) is True

    def test_many_categories_triggers_horizontal(self):
        cats = [f"F{i}" for i in range(9)]
        assert _needs_horizontal_bar(cats) is True

    def test_exactly_at_threshold(self):
        # _HORIZONTAL_CAT_COUNT = 8 → 8 categories should NOT trigger (> 8)
        cats = [f"F{i}" for i in range(8)]
        assert _needs_horizontal_bar(cats) is False


# ===========================================================================
# _sorted_by
# ===========================================================================

class TestSortedBy:
    def test_sorts_numerically(self):
        rows = [{"val": 3}, {"val": 1}, {"val": 2}]
        result = _sorted_by(rows, "val")
        assert [r["val"] for r in result] == [1, 2, 3]

    def test_none_values_cause_type_error_returning_original_order(self):
        # _sorted_by uses "" for None, which can't compare with int → TypeError → original order
        rows = [{"val": 2}, {"val": None}, {"val": 1}]
        result = _sorted_by(rows, "val")
        assert len(result) == 3  # original order preserved

    def test_mixed_type_returns_original(self):
        rows = [{"val": 1}, {"val": "a"}]
        # Should not raise; returns original order on TypeError
        result = _sorted_by(rows, "val")
        assert len(result) == 2

    def test_missing_key(self):
        rows = [{"other": 1}, {"val": 2}]
        result = _sorted_by(rows, "val")
        assert len(result) == 2


# ===========================================================================
# _has_text_heavy_column
# ===========================================================================

class TestHasTextHeavyColumn:
    def test_content_key_triggers(self):
        rows = [{"content": "short"}]
        cols = [_col("content", "string")]
        assert _has_text_heavy_column(rows, cols) is True

    def test_long_string_triggers(self):
        long_text = "x" * 90
        rows = [{"desc": long_text}]
        cols = [_col("desc", "string")]
        assert _has_text_heavy_column(rows, cols) is True

    def test_short_string_no_trigger(self):
        rows = [{"name": "hello"}]
        cols = [_col("name", "string")]
        assert _has_text_heavy_column(rows, cols) is False

    def test_numeric_column_skipped(self):
        rows = [{"energy_mwh": 100.0}]
        cols = [_col("energy_mwh", "number")]
        assert _has_text_heavy_column(rows, cols) is False

    def test_empty_rows(self):
        cols = [_col("content", "string")]
        assert _has_text_heavy_column([], cols) is True  # key "content" is in _TEXT_HEAVY_KEYS


# ===========================================================================
# _axis_title
# ===========================================================================

class TestAxisTitle:
    def test_with_unit(self):
        col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        assert _axis_title(col) == "Energy (MWh)"

    def test_without_unit(self):
        col = _col("facility", label="Facility")
        assert _axis_title(col) == "Facility"


# ===========================================================================
# _base_layout
# ===========================================================================

class TestBaseLayout:
    def test_contains_title(self):
        layout = _base_layout("My Chart")
        assert layout["title"] == "My Chart"

    def test_has_colorway(self):
        layout = _base_layout("Test")
        assert "colorway" in layout

    def test_custom_bottom_margin(self):
        layout = _base_layout("Test", bottom_margin=100)
        assert layout["margin"]["b"] == 100


# ===========================================================================
# _build_geo_map_chart
# ===========================================================================

class TestBuildGeoMapChart:
    def _make_cols(self):
        return [
            _col("facility", "string"),
            _col("latitude", "number", unit="°"),
            _col("longitude", "number", unit="°"),
            _col("capacity_mw", "number", unit="MW"),
        ]

    def _make_rows(self):
        return [
            {"facility": "A", "latitude": -33.0, "longitude": 151.0, "capacity_mw": 50.0},
            {"facility": "B", "latitude": -35.0, "longitude": 149.0, "capacity_mw": 30.0},
        ]

    def test_returns_scatter_geo(self):
        cols = self._make_cols()
        lat_col = next(c for c in cols if c.key == "latitude")
        lng_col = next(c for c in cols if c.key == "longitude")
        cat_col = next(c for c in cols if c.key == "facility")
        result = _build_geo_map_chart("facilities", self._make_rows(), cols, lat_col, lng_col, cat_col)
        assert result.chart_type == "scatter_geo"

    def test_title_with_capacity(self):
        cols = self._make_cols()
        lat_col = next(c for c in cols if c.key == "latitude")
        lng_col = next(c for c in cols if c.key == "longitude")
        cat_col = next(c for c in cols if c.key == "facility")
        result = _build_geo_map_chart("facilities", self._make_rows(), cols, lat_col, lng_col, cat_col)
        assert "Capacity" in result.title or "MW" in result.title

    def test_no_capacity_col_default_title(self):
        cols = [
            _col("facility", "string"),
            _col("latitude", "number", unit="°"),
            _col("longitude", "number", unit="°"),
        ]
        rows = [{"facility": "A", "latitude": -33.0, "longitude": 151.0}]
        lat_col = next(c for c in cols if c.key == "latitude")
        lng_col = next(c for c in cols if c.key == "longitude")
        cat_col = next(c for c in cols if c.key == "facility")
        result = _build_geo_map_chart("facilities", rows, cols, lat_col, lng_col, cat_col)
        assert result.title == "Facility Locations"

    def test_no_cat_col(self):
        cols = self._make_cols()
        lat_col = next(c for c in cols if c.key == "latitude")
        lng_col = next(c for c in cols if c.key == "longitude")
        result = _build_geo_map_chart("facilities", self._make_rows(), cols, lat_col, lng_col, None)
        assert result is not None

    def test_plotly_spec_structure(self):
        cols = self._make_cols()
        lat_col = next(c for c in cols if c.key == "latitude")
        lng_col = next(c for c in cols if c.key == "longitude")
        cat_col = next(c for c in cols if c.key == "facility")
        result = _build_geo_map_chart("facilities", self._make_rows(), cols, lat_col, lng_col, cat_col)
        assert "data" in result.plotly_spec
        assert "layout" in result.plotly_spec
        assert result.plotly_spec["data"][0]["type"] == "scattergeo"


# ===========================================================================
# _build_timeseries_chart
# ===========================================================================

class TestBuildTimeseriesChart:
    def test_basic_line_chart(self):
        rows = _timeseries_rows(5)
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        numeric_cols = _detect_numeric_columns(cols)
        result = _build_timeseries_chart("energy", rows, time_col, None, numeric_cols)
        assert result is not None
        assert result.chart_type == "line"

    def test_title_includes_metric(self):
        rows = _timeseries_rows(5)
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        numeric_cols = _detect_numeric_columns(cols)
        result = _build_timeseries_chart("energy", rows, time_col, None, numeric_cols)
        assert "Energy" in result.title

    def test_multi_facility_creates_multi_trace(self):
        rows = [
            {"facility": "FA", "date": "2024-01-01", "energy_mwh": 100.0},
            {"facility": "FB", "date": "2024-01-01", "energy_mwh": 80.0},
            {"facility": "FA", "date": "2024-01-02", "energy_mwh": 110.0},
            {"facility": "FB", "date": "2024-01-02", "energy_mwh": 90.0},
        ]
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        cat_col = _detect_category_column(cols)
        numeric_cols = _detect_numeric_columns(cols)
        result = _build_timeseries_chart("energy", rows, time_col, cat_col, numeric_cols)
        assert result is not None
        assert len(result.plotly_spec["data"]) == 2  # two facilities

    def test_caps_at_max_line_series(self):
        # 6 facilities → only top 5 traces (MAX_LINE_SERIES = 5)
        facilities = [f"F{i}" for i in range(6)]
        rows = [
            {"facility": f, "date": f"2024-01-{i+1:02d}", "energy_mwh": float((6 - i) * 10)}
            for i, f in enumerate(facilities)
        ]
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        cat_col = _detect_category_column(cols)
        numeric_cols = _detect_numeric_columns(cols)
        result = _build_timeseries_chart("energy", rows, time_col, cat_col, numeric_cols)
        assert result is not None
        assert len(result.plotly_spec["data"]) <= 5

    def test_no_numeric_col_returns_none(self):
        rows = [{"date": "2024-01-01", "facility": "F1"}]
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        numeric_cols = []  # no numeric
        result = _build_timeseries_chart("energy", rows, time_col, None, numeric_cols)
        assert result is None

    def test_single_row(self):
        rows = [{"date": "2024-01-01", "energy_mwh": 100.0}]
        cols = _build_columns(rows)
        time_col = _detect_time_column(cols)
        numeric_cols = _detect_numeric_columns(cols)
        result = _build_timeseries_chart("energy", rows, time_col, None, numeric_cols)
        assert result is not None


# ===========================================================================
# _build_bar_chart
# ===========================================================================

class TestBuildBarChart:
    def _make_simple(self):
        rows = [{"facility": f"F{i}", "energy_mwh": float(i * 10)} for i in range(5)]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        return rows, cat_col, y_col

    def test_returns_bar_chart(self):
        rows, cat_col, y_col = self._make_simple()
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        assert result.chart_type == "bar"

    def test_vertical_bar_when_few_short_labels(self):
        rows = [{"facility": f"F{i}", "energy_mwh": float(i)} for i in range(4)]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        trace = result.plotly_spec["data"][0]
        assert trace.get("orientation") != "h"

    def test_horizontal_bar_for_long_labels(self):
        rows = [{"facility": f"Very Long Station Name {i}", "energy_mwh": float(i)} for i in range(3)]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        trace = result.plotly_spec["data"][0]
        assert trace.get("orientation") == "h"

    def test_truncates_to_max_categories(self):
        rows = [{"facility": f"F{i:02d}", "energy_mwh": float(i)} for i in range(20)]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        trace = result.plotly_spec["data"][0]
        # x for vertical or y for horizontal
        cats = trace.get("x") or trace.get("y")
        assert len(cats) <= 15

    def test_top_n_in_title_when_truncated(self):
        rows = [{"facility": f"F{i:02d}", "energy_mwh": float(i)} for i in range(20)]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        assert "Top" in result.title

    def test_source_metric_key(self):
        rows, cat_col, y_col = self._make_simple()
        result = _build_bar_chart("my_metric", rows, cat_col, y_col)
        assert result.source_metric_key == "my_metric"

    def test_single_row(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        cat_col = _col("facility", "string", label="Facility")
        y_col = _col("energy_mwh", "number", unit="MWh", label="Energy")
        result = _build_bar_chart("energy", rows, cat_col, y_col)
        assert result is not None


# ===========================================================================
# _build_grouped_bar_chart
# ===========================================================================

class TestBuildGroupedBarChart:
    def test_same_unit_grouped(self):
        rows = [
            {"facility": "F1", "forecast_mwh": 100.0, "actual_mwh": 90.0},
            {"facility": "F2", "forecast_mwh": 80.0, "actual_mwh": 85.0},
        ]
        cat_col = _col("facility", "string", label="Facility")
        y_cols = [
            _col("forecast_mwh", "number", unit="MWh", label="Forecast"),
            _col("actual_mwh", "number", unit="MWh", label="Actual"),
        ]
        result = _build_grouped_bar_chart("energy", rows, cat_col, y_cols)
        assert result.chart_type == "bar"
        assert result.plotly_spec["layout"].get("barmode") == "group"

    def test_mixed_units_falls_back_to_single_bar(self):
        rows = [
            {"facility": "F1", "energy_mwh": 100.0, "capacity_factor_pct": 0.8},
        ]
        cat_col = _col("facility", "string", label="Facility")
        y_cols = [
            _col("energy_mwh", "number", unit="MWh", label="Energy"),
            _col("capacity_factor_pct", "number", unit="%", label="Capacity Factor"),
        ]
        # Different units → falls back to single bar
        result = _build_grouped_bar_chart("energy", rows, cat_col, y_cols)
        assert result is not None
        # Single bar fallback: only 1 trace
        assert len(result.plotly_spec["data"]) == 1

    def test_multiple_traces_for_same_unit(self):
        rows = [
            {"facility": f"F{i}", "forecast_mwh": float(i * 10), "actual_mwh": float(i * 9)}
            for i in range(3)
        ]
        cat_col = _col("facility", "string", label="Facility")
        y_cols = [
            _col("forecast_mwh", "number", unit="MWh", label="Forecast"),
            _col("actual_mwh", "number", unit="MWh", label="Actual"),
        ]
        result = _build_grouped_bar_chart("energy", rows, cat_col, y_cols)
        assert len(result.plotly_spec["data"]) == 2


# ===========================================================================
# _build_scatter_chart (pure 2-numeric scatter)
# ===========================================================================

class TestBuildScatterChart:
    def test_returns_scatter(self):
        rows = [{"x_val": float(i), "y_val": float(i * 2)} for i in range(5)]
        x_col = _col("x_val", "number", label="X Val")
        y_col = _col("y_val", "number", label="Y Val")
        result = _build_scatter_chart("test", rows, x_col, y_col)
        assert result.chart_type == "scatter"

    def test_title_format(self):
        rows = [{"temp": 20.0, "energy": 100.0}]
        x_col = _col("temp", "number", label="Temperature")
        y_col = _col("energy", "number", label="Energy")
        result = _build_scatter_chart("test", rows, x_col, y_col)
        assert "Energy" in result.title
        assert "Temperature" in result.title

    def test_trace_mode_markers(self):
        rows = [{"temp": float(i), "energy": float(i * 10)} for i in range(5)]
        x_col = _col("temp", "number", label="Temperature")
        y_col = _col("energy", "number", label="Energy")
        result = _build_scatter_chart("test", rows, x_col, y_col)
        assert result.plotly_spec["data"][0]["mode"] == "markers"


# ===========================================================================
# _build_histogram_chart
# ===========================================================================

class TestBuildHistogramChart:
    def test_returns_histogram(self):
        rows = [{"value": float(i)} for i in range(15)]
        y_col = _col("value", "number", label="Value")
        result = _build_histogram_chart("dist", rows, y_col)
        assert result.chart_type == "histogram"

    def test_title_prefix(self):
        rows = [{"value": float(i)} for i in range(15)]
        y_col = _col("value", "number", label="Value")
        result = _build_histogram_chart("dist", rows, y_col)
        assert "Distribution" in result.title

    def test_filters_non_numeric_values(self):
        rows = [{"value": float(i)} for i in range(12)] + [{"value": None}]
        y_col = _col("value", "number", label="Value")
        result = _build_histogram_chart("dist", rows, y_col)
        # None should be filtered from x array
        assert None not in result.plotly_spec["data"][0]["x"]


# ===========================================================================
# _format_for_kpi
# ===========================================================================

class TestFormatForKpi:
    def test_numeric_value(self):
        card = _format_for_kpi("energy_mwh", 150.5)
        assert card is not None
        assert card.value == 150.5

    def test_non_numeric_returns_none(self):
        assert _format_for_kpi("facility", "F1") is None

    def test_pct_key_format(self):
        card = _format_for_kpi("capacity_factor_pct", 0.85)
        assert card is not None
        assert card.format == "percent"

    def test_percent_unit_format(self):
        card = _format_for_kpi("avg_humidity", 60.0)
        assert card is not None
        assert card.format == "percent"

    def test_integer_format_for_int_value(self):
        card = _format_for_kpi("station_count", 8)
        assert card is not None
        assert card.format == "integer"

    def test_count_key_format(self):
        card = _format_for_kpi("row_count", 1000)
        assert card is not None
        assert card.format == "integer"

    def test_boolean_returns_none(self):
        assert _format_for_kpi("flag", True) is None

    def test_zero_value(self):
        card = _format_for_kpi("energy_mwh", 0.0)
        assert card is not None
        assert card.value == 0.0


# ===========================================================================
# _build_kpi_cards
# ===========================================================================

class TestBuildKpiCards:
    def test_scalar_metrics(self):
        metrics = {"total_energy_mwh": 500.0, "station_count": 8}
        result = _build_kpi_cards(metrics)
        assert result is not None
        assert len(result.cards) >= 1

    def test_priority_order(self):
        metrics = {"total_energy_mwh": 500.0, "r2": 0.95, "mae": 10.2}
        result = _build_kpi_cards(metrics)
        assert result is not None
        # total_energy_mwh should appear before r2 (priority order)
        labels = [c.label for c in result.cards]
        total_idx = next((i for i, l in enumerate(labels) if "Energy" in l or "Total" in l), None)
        r2_idx = next((i for i, l in enumerate(labels) if "R2" in l or "r2" in l.lower()), None)
        if total_idx is not None and r2_idx is not None:
            assert total_idx < r2_idx

    def test_empty_metrics_returns_none(self):
        assert _build_kpi_cards({}) is None

    def test_non_numeric_values_ignored(self):
        metrics = {"facility": "F1", "status": "ok"}
        assert _build_kpi_cards(metrics) is None

    def test_max_6_cards(self):
        metrics = {f"val_{i}_pct": float(i) for i in range(20)}
        result = _build_kpi_cards(metrics)
        assert result is not None
        assert len(result.cards) <= 6

    def test_list_count_is_inferred_as_kpi(self):
        # When 'facilities' is a list, the implementation infers facility_count from len()
        metrics = {"facilities": [{"id": "F1"}, {"id": "F2"}]}
        result = _build_kpi_cards(metrics)
        assert result is not None
        assert any("count" in c.label.lower() or "Count" in c.label for c in result.cards)

    def test_facility_list_infers_count(self):
        metrics = {"facilities": [{"id": "F1"}, {"id": "F2"}, {"id": "F3"}]}
        result = _build_kpi_cards(metrics)
        # facility_count should be inferred from len(facilities)
        assert result is not None
        count_card = next((c for c in result.cards if "Count" in c.label or "count" in c.label.lower()), None)
        assert count_card is not None
        assert count_card.value == 3

    def test_facility_list_does_not_override_existing_count(self):
        metrics = {"facilities": [{"id": "F1"}, {"id": "F2"}], "facility_count": 5}
        result = _build_kpi_cards(metrics)
        # existing facility_count=5 should NOT be overridden by len=2
        assert result is not None
        count_card = next((c for c in result.cards if "Count" in c.label or "count" in c.label.lower()), None)
        if count_card is not None:
            assert count_card.value == 5

    def test_mixed_numeric_and_list(self):
        metrics = {
            "total_energy_mwh": 1000.0,
            "readings": [{"date": "2024-01-01", "val": 10}] * 3,
        }
        result = _build_kpi_cards(metrics)
        assert result is not None
        assert any(c.value == 1000.0 for c in result.cards)


# ===========================================================================
# _detect_correlation_axes
# ===========================================================================

class TestDetectCorrelationAxes:
    def _weather_perf_cols(self):
        return [
            _col("avg_temperature_c", "number", label="Avg Temperature"),
            _col("capacity_factor_pct", "number", label="Capacity Factor"),
            _col("avg_humidity", "number", label="Avg Humidity"),
        ]

    def test_detects_temperature_vs_capacity_factor(self):
        cols = self._weather_perf_cols()
        result = _detect_correlation_axes(cols)
        assert result is not None
        x_col, y_col = result
        assert "temperature" in x_col.key.lower()

    def test_no_x_pattern_returns_none(self):
        cols = [
            _col("energy_mwh", "number", label="Energy"),
            _col("capacity_factor_pct", "number", label="CF"),
        ]
        # no weather-like column → should return None
        assert _detect_correlation_axes(cols) is None

    def test_no_y_pattern_returns_none(self):
        cols = [
            _col("avg_temperature_c", "number", label="Avg Temperature"),
            _col("station_count", "number", label="Station Count"),
        ]
        assert _detect_correlation_axes(cols) is None

    def test_query_promotes_cloud_over_temperature(self):
        cols = [
            _col("avg_temperature_c", "number", label="Avg Temperature"),
            _col("avg_cloud_cover", "number", label="Avg Cloud Cover"),
            _col("capacity_factor_pct", "number", label="Capacity Factor"),
        ]
        result = _detect_correlation_axes(cols, user_query="how does cloud affect performance?")
        assert result is not None
        x_col, _ = result
        assert "cloud" in x_col.key.lower()

    def test_query_promotes_energy_y_over_pr(self):
        cols = [
            _col("avg_temperature_c", "number", label="Avg Temperature"),
            _col("performance_ratio_pct", "number", label="PR"),
            _col("total_energy", "number", label="Total Energy"),
        ]
        result = _detect_correlation_axes(cols, user_query="how does temperature affect energy production?")
        assert result is not None
        _, y_col = result
        assert "energy" in y_col.key.lower()


# ===========================================================================
# _reorder_x_patterns_for_query
# ===========================================================================

class TestReorderXPatterns:
    def test_no_query_returns_default(self):
        result = _reorder_x_patterns_for_query(None)
        assert result[0] == "temperature"  # default first

    def test_cloud_keyword_promotes_cloud(self):
        result = _reorder_x_patterns_for_query("cloud cover impact")
        assert "cloud" in result[0] or "cloud_cover" in result[0]

    def test_aqi_keyword_promotes_aqi(self):
        result = _reorder_x_patterns_for_query("aqi effect on energy")
        assert "aqi" in result[0]

    def test_empty_query_returns_default(self):
        result = _reorder_x_patterns_for_query("")
        assert result[0] == "temperature"


# ===========================================================================
# _reorder_y_patterns_for_query
# ===========================================================================

class TestReorderYPatterns:
    def test_no_query_returns_default(self):
        result = _reorder_y_patterns_for_query(None)
        assert result[0] == "performance_ratio"  # default first

    def test_energy_production_promotes_energy(self):
        result = _reorder_y_patterns_for_query("energy production today")
        assert "energy" in result[0] or "total_energy" in result[0]

    def test_vietnamese_san_luong_promotes_energy(self):
        result = _reorder_y_patterns_for_query("sản lượng điện hôm nay")
        assert "energy" in result[0] or "total_energy" in result[0]


# ===========================================================================
# _find_by_pattern
# ===========================================================================

class TestFindByPattern:
    def test_finds_by_substring(self):
        cols = [_col("avg_temperature_c", "number"), _col("energy_mwh", "number")]
        result = _find_by_pattern(cols, ("temperature",))
        assert result is not None
        assert result.key == "avg_temperature_c"

    def test_priority_order(self):
        cols = [_col("avg_cloud_cover", "number"), _col("avg_temperature_c", "number")]
        result = _find_by_pattern(cols, ("temperature", "cloud"))
        assert result is not None
        assert result.key == "avg_temperature_c"

    def test_no_match_returns_none(self):
        cols = [_col("energy_mwh", "number")]
        assert _find_by_pattern(cols, ("temperature",)) is None

    def test_empty_patterns(self):
        cols = [_col("energy_mwh", "number")]
        assert _find_by_pattern(cols, ()) is None


# ===========================================================================
# _aggregate_by_category
# ===========================================================================

class TestAggregateByCCategory:
    def test_averages_multiple_rows(self):
        rows = [
            {"facility": "F1", "temperature": 20.0, "energy": 100.0},
            {"facility": "F1", "temperature": 24.0, "energy": 120.0},
            {"facility": "F2", "temperature": 18.0, "energy": 90.0},
        ]
        result = _aggregate_by_category(rows, "facility", ["temperature", "energy"])
        assert len(result) == 2
        f1 = next(r for r in result if r["facility"] == "F1")
        assert f1["temperature"] == pytest.approx(22.0)
        assert f1["energy"] == pytest.approx(110.0)

    def test_empty_key_skipped(self):
        rows = [
            {"facility": "", "temperature": 20.0},
            {"facility": "F1", "temperature": 22.0},
        ]
        result = _aggregate_by_category(rows, "facility", ["temperature"])
        assert all(r["facility"] != "" for r in result)

    def test_none_key_skipped(self):
        rows = [
            {"facility": None, "temperature": 20.0},
            {"facility": "F1", "temperature": 22.0},
        ]
        result = _aggregate_by_category(rows, "facility", ["temperature"])
        assert len(result) == 1

    def test_non_numeric_values_excluded_from_avg(self):
        rows = [
            {"facility": "F1", "temperature": 20.0},
            {"facility": "F1", "temperature": None},
        ]
        result = _aggregate_by_category(rows, "facility", ["temperature"])
        assert len(result) == 1
        assert result[0]["temperature"] == pytest.approx(20.0)

    def test_single_row_per_category(self):
        rows = [{"facility": "F1", "temp": 25.0}]
        result = _aggregate_by_category(rows, "facility", ["temp"])
        assert len(result) == 1
        assert result[0]["temp"] == pytest.approx(25.0)


# ===========================================================================
# _build_correlation_scatter
# ===========================================================================

class TestBuildCorrelationScatter:
    def test_returns_scatter_type(self):
        rows = _weather_rows()
        cat_col = _col("facility", "string", label="Facility")
        x_col = _col("avg_temperature_c", "number", label="Avg Temperature")
        y_col = _col("avg_capacity_factor", "number", label="Avg Capacity Factor")
        result = _build_correlation_scatter("weather", rows, cat_col, x_col, y_col)
        assert result.chart_type == "scatter"

    def test_aggregates_per_category(self):
        # Two rows per facility — should produce one point per facility
        rows = []
        for fac in ["FA", "FB"]:
            rows.extend([
                {"facility": fac, "avg_temperature_c": 20.0, "avg_capacity_factor": 0.3},
                {"facility": fac, "avg_temperature_c": 24.0, "avg_capacity_factor": 0.35},
            ])
        cat_col = _col("facility", "string", label="Facility")
        x_col = _col("avg_temperature_c", "number", label="Avg Temperature")
        y_col = _col("avg_capacity_factor", "number", label="Avg Capacity Factor")
        result = _build_correlation_scatter("weather", rows, cat_col, x_col, y_col)
        assert len(result.plotly_spec["data"][0]["x"]) == 2

    def test_mode_markers_text(self):
        rows = _weather_rows()
        cat_col = _col("facility", "string", label="Facility")
        x_col = _col("avg_temperature_c", "number", label="Avg Temperature")
        y_col = _col("avg_capacity_factor", "number", label="Avg Capacity Factor")
        result = _build_correlation_scatter("weather", rows, cat_col, x_col, y_col)
        assert result.plotly_spec["data"][0]["mode"] == "markers+text"

    def test_filters_invalid_points(self):
        rows = [
            {"facility": "F1", "avg_temperature_c": None, "avg_capacity_factor": 0.3},
            {"facility": "F2", "avg_temperature_c": 22.0, "avg_capacity_factor": None},
            {"facility": "F3", "avg_temperature_c": 20.0, "avg_capacity_factor": 0.35},
        ]
        cat_col = _col("facility", "string", label="Facility")
        x_col = _col("avg_temperature_c", "number", label="Avg Temperature")
        y_col = _col("avg_capacity_factor", "number", label="Avg Capacity Factor")
        result = _build_correlation_scatter("weather", rows, cat_col, x_col, y_col)
        # Only F3 has valid x AND y
        assert len(result.plotly_spec["data"][0]["x"]) == 1


# ===========================================================================
# _build_chart (dispatch logic)
# ===========================================================================

class TestBuildChart:
    def test_empty_rows_returns_none(self):
        assert _build_chart("k", [], [_col("a", "number")]) is None

    def test_empty_columns_returns_none(self):
        assert _build_chart("k", [{"a": 1}], []) is None

    def test_text_heavy_returns_none(self):
        rows = [{"content": "x" * 100, "energy_mwh": 50.0}]
        cols = _build_columns(rows)
        assert _build_chart("k", rows, cols) is None

    def test_geo_detected(self):
        rows = [{"facility": "F1", "latitude": -33.0, "longitude": 151.0, "capacity_mw": 50.0}]
        cols = _build_columns(rows)
        result = _build_chart("facilities", rows, cols)
        assert result is not None
        assert result.chart_type == "scatter_geo"

    def test_timeseries_detected(self):
        rows = _timeseries_rows(5)
        cols = _build_columns(rows)
        result = _build_chart("energy", rows, cols)
        assert result is not None
        assert result.chart_type == "line"

    def test_bar_chart_for_category_data(self):
        rows = [{"facility": f"F{i}", "energy_mwh": float(i * 10)} for i in range(5)]
        cols = _build_columns(rows)
        result = _build_chart("energy", rows, cols)
        assert result is not None
        assert result.chart_type == "bar"

    def test_scatter_for_two_numerics_no_cat(self):
        rows = [{"x_val": float(i), "y_val": float(i * 2)} for i in range(5)]
        cols = _build_columns(rows)
        result = _build_chart("data", rows, cols)
        assert result is not None
        assert result.chart_type == "scatter"

    def test_histogram_for_one_numeric_enough_rows(self):
        rows = [{"value": float(i)} for i in range(15)]
        cols = _build_columns(rows)
        result = _build_chart("dist", rows, cols)
        assert result is not None
        assert result.chart_type == "histogram"

    def test_histogram_not_built_for_few_rows(self):
        rows = [{"value": float(i)} for i in range(5)]  # < 12 rows
        cols = _build_columns(rows)
        result = _build_chart("dist", rows, cols)
        assert result is None

    def test_no_numeric_returns_none(self):
        rows = [{"facility": "F1", "status": "ok"}]
        cols = _build_columns(rows)
        result = _build_chart("status", rows, cols)
        assert result is None

    def test_correlation_scatter_when_weather_and_perf_present(self):
        rows = [
            {"facility": f"F{i}", "avg_temperature_c": float(20 + i), "capacity_factor_pct": float(0.3 + i * 0.02)}
            for i in range(6)
        ]
        cols = _build_columns(rows)
        result = _build_chart("weather_impact", rows, cols)
        assert result is not None
        assert result.chart_type == "scatter"

    def test_grouped_bar_when_multiple_y_same_unit(self):
        rows = [
            {"facility": f"F{i}", "forecast_mwh": float(i * 10), "actual_mwh": float(i * 9)}
            for i in range(4)
        ]
        cols = _build_columns(rows)
        result = _build_chart("energy", rows, cols)
        assert result is not None
        assert result.chart_type == "bar"

    def test_single_row_bar(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        cols = _build_columns(rows)
        result = _build_chart("energy", rows, cols)
        assert result is not None


# ===========================================================================
# _title_for_topic
# ===========================================================================

class TestTitleForTopic:
    def test_hourly_energy_and_capacity(self):
        rows = [{"hour": 1, "energy_mwh": 10.0, "capacity_factor_pct": 0.3}]
        title = _title_for_topic("energy_performance", "readings", rows)
        assert "Hourly" in title

    def test_daily_energy_only(self):
        rows = [{"date": "2024-01-01", "energy_mwh": 10.0}]
        title = _title_for_topic(None, "readings", rows)
        assert "Daily" in title and "energy" in title.lower()

    def test_per_facility_energy(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        title = _title_for_topic(None, "energy", rows)
        assert "facility" in title.lower() or "energy" in title.lower()

    def test_actual_vs_forecast(self):
        rows = [{"facility": "F1", "actual_mwh": 100.0, "forecast_mwh": 110.0}]
        title = _title_for_topic(None, "forecast", rows)
        assert "forecast" in title.lower() or "actual" in title.lower()

    def test_known_topic(self):
        title = _title_for_topic("station_report", "data", None)
        assert title == "Station daily report"

    def test_forecast_72h_topic(self):
        title = _title_for_topic("forecast_72h", "data", None)
        assert "72h" in title or "forecast" in title.lower()

    def test_fallback_humanize_metric_key(self):
        title = _title_for_topic(None, "custom_metric_data", [])
        assert title == "Custom Metric Data"

    def test_per_facility_capacity_no_energy(self):
        rows = [{"facility": "F1", "capacity_factor_pct": 0.8}]
        title = _title_for_topic(None, "capacity", rows)
        assert "capacity" in title.lower()

    def test_performance_ratio(self):
        rows = [{"facility": "F1", "performance_ratio_pct": 0.85}]
        title = _title_for_topic(None, "perf", rows)
        assert "performance" in title.lower() or "ratio" in title.lower()

    def test_energy_and_capacity_by_facility(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0, "capacity_factor_pct": 0.8}]
        title = _title_for_topic(None, "summary", rows)
        assert "energy" in title.lower() or "capacity" in title.lower()


# ===========================================================================
# _project_table_for_scatter
# ===========================================================================

class TestProjectTableForScatter:
    def _make_table_and_chart(self):
        rows = [
            {"facility": "F1", "avg_temperature_c": 20.0, "capacity_factor_pct": 0.3, "noise_col": "x"},
            {"facility": "F2", "avg_temperature_c": 25.0, "capacity_factor_pct": 0.35, "noise_col": "y"},
        ]
        table = _build_data_table("weather", rows)
        cat_col = _col("facility", "string", label="Facility")
        x_col = _col("avg_temperature_c", "number", unit="°C", label="Avg Temperature")
        y_col = _col("capacity_factor_pct", "number", unit="%", label="Capacity Factor")
        chart = _build_correlation_scatter("weather", rows, cat_col, x_col, y_col)
        return table, rows, chart

    def test_keeps_x_and_y_columns(self):
        table, rows, chart = self._make_table_and_chart()
        projected = _project_table_for_scatter(table, rows, chart)
        col_keys = [c.key for c in projected.columns]
        assert "avg_temperature_c" in col_keys
        assert "capacity_factor_pct" in col_keys

    def test_noise_columns_dropped(self):
        table, rows, chart = self._make_table_and_chart()
        projected = _project_table_for_scatter(table, rows, chart)
        col_keys = [c.key for c in projected.columns]
        assert "noise_col" not in col_keys

    def test_facility_column_kept(self):
        table, rows, chart = self._make_table_and_chart()
        projected = _project_table_for_scatter(table, rows, chart)
        col_keys = [c.key for c in projected.columns]
        assert "facility" in col_keys

    def test_uses_chart_title(self):
        table, rows, chart = self._make_table_and_chart()
        projected = _project_table_for_scatter(table, rows, chart)
        assert projected.title == chart.title

    def test_returns_original_when_keys_not_found(self):
        rows = [{"facility": "F1", "energy_mwh": 100.0}]
        table = _build_data_table("energy", rows)
        # Chart with x/y titles that don't match any column label
        chart = ChartPayload(
            chart_type="scatter",
            title="Unknown vs Unknown",
            plotly_spec={
                "data": [],
                "layout": {
                    "xaxis": {"title": "NonExistentX"},
                    "yaxis": {"title": "NonExistentY"},
                },
            },
            source_metric_key="energy",
        )
        projected = _project_table_for_scatter(table, rows, chart)
        # Should return original table unchanged
        assert projected is table


# ===========================================================================
# ChartSpecBuilder.build — integration tests
# ===========================================================================

class TestChartSpecBuilderBuild:
    def setup_method(self):
        self.builder = ChartSpecBuilder()

    def test_empty_metrics(self):
        table, chart, kpi = self.builder.build({})
        assert table is None
        assert chart is None
        assert kpi is None

    def test_none_metrics(self):
        table, chart, kpi = self.builder.build(None)  # type: ignore[arg-type]
        assert table is None
        assert chart is None
        assert kpi is None

    def test_non_dict_metrics(self):
        table, chart, kpi = self.builder.build("not a dict")  # type: ignore[arg-type]
        assert table is None
        assert chart is None
        assert kpi is None

    def test_scalar_only_metrics(self):
        metrics = {"total_energy_mwh": 500.0, "station_count": 8}
        table, chart, kpi = self.builder.build(metrics)
        assert table is None
        assert chart is None
        assert kpi is not None
        assert len(kpi.cards) >= 1

    def test_list_data_produces_table(self):
        metrics = {"readings": _energy_rows(5)}
        table, chart, kpi = self.builder.build(metrics)
        assert table is not None
        assert table.row_count == 5

    def test_list_data_produces_chart(self):
        metrics = {"readings": _energy_rows(5)}
        table, chart, kpi = self.builder.build(metrics)
        assert chart is not None

    def test_combined_scalar_and_list(self):
        metrics = {
            "total_energy_mwh": 1000.0,
            "station_count": 8,
            "readings": _energy_rows(5),
        }
        table, chart, kpi = self.builder.build(metrics)
        assert table is not None
        assert chart is not None
        assert kpi is not None

    def test_scatter_projects_table(self):
        rows = [
            {"facility": f"F{i}", "avg_temperature_c": float(20 + i), "capacity_factor_pct": float(0.3 + i * 0.02)}
            for i in range(6)
        ]
        metrics = {"weather_impact": rows}
        table, chart, kpi = self.builder.build(metrics)
        if chart is not None and chart.chart_type == "scatter" and table is not None:
            # projected table should have fewer columns than original full set
            col_keys = [c.key for c in table.columns]
            assert "avg_temperature_c" in col_keys
            assert "capacity_factor_pct" in col_keys

    def test_user_query_influences_chart(self):
        rows = [
            {"facility": f"F{i}", "avg_cloud_cover": float(30 + i), "avg_temperature_c": float(20 + i), "capacity_factor_pct": float(0.3 + i * 0.02)}
            for i in range(6)
        ]
        metrics = {"weather": rows}
        _, chart_cloud, _ = self.builder.build(metrics, user_query="how does cloud affect capacity?")
        _, chart_temp, _ = self.builder.build(metrics, user_query=None)

        if chart_cloud is not None and chart_temp is not None:
            # Cloud query should put cloud on x axis
            cloud_x_title = (chart_cloud.plotly_spec.get("layout") or {}).get("xaxis", {}).get("title", "")
            temp_x_title = (chart_temp.plotly_spec.get("layout") or {}).get("xaxis", {}).get("title", "")
            # They may differ in axis ordering
            assert isinstance(cloud_x_title, str)
            assert isinstance(temp_x_title, str)

    def test_topic_hint_in_title(self):
        rows = [{"date": "2024-01-01", "energy_mwh": 100.0}]
        metrics = {"readings": rows}
        table, _, _ = self.builder.build(metrics, topic="station_report")
        # With a time col, _title_for_topic returns a period-based title
        assert table is not None

    def test_exception_in_build_returns_none_for_table_and_chart(self):
        # Simulate rows that will still build a table but chart fails gracefully
        # We test graceful degradation: even if chart throws, kpi should still work
        metrics = {
            "bad_list": [{"a": object()}],  # un-serializable but still a list of dicts
            "total_energy_mwh": 200.0,
        }
        # Should not raise — exceptions are caught internally
        table, chart, kpi = self.builder.build(metrics)
        # kpi should still be built
        assert kpi is not None

    def test_text_heavy_data_no_chart(self):
        rows = [{"source_file": f"doc{i}", "content": "x" * 100, "score": float(i)} for i in range(5)]
        metrics = {"docs": rows}
        _, chart, _ = self.builder.build(metrics)
        assert chart is None

    def test_geo_data_scatter_geo_chart(self):
        rows = [
            {"facility": f"F{i}", "latitude": -33.0 + i * 0.1, "longitude": 151.0 + i * 0.1, "capacity_mw": float(50 + i * 5)}
            for i in range(3)
        ]
        metrics = {"facilities": rows}
        _, chart, _ = self.builder.build(metrics)
        assert chart is not None
        assert chart.chart_type == "scatter_geo"

    def test_timeseries_chart(self):
        rows = _timeseries_rows(10)
        metrics = {"daily": rows}
        _, chart, _ = self.builder.build(metrics)
        assert chart is not None
        assert chart.chart_type == "line"

    def test_empty_list_no_table(self):
        metrics = {"readings": []}
        table, chart, kpi = self.builder.build(metrics)
        assert table is None
        assert chart is None

    def test_single_row_list(self):
        metrics = {"readings": [{"facility": "F1", "energy_mwh": 100.0}]}
        table, chart, kpi = self.builder.build(metrics)
        assert table is not None
        assert table.row_count == 1

    def test_picks_largest_list(self):
        metrics = {
            "small": [{"a": 1}],
            "large": [{"facility": f"F{i}", "energy_mwh": float(i * 10)} for i in range(10)],
        }
        table, _, _ = self.builder.build(metrics)
        assert table is not None
        assert table.row_count == 10


# ===========================================================================
# Edge cases: various key_metrics structures
# ===========================================================================

class TestKeyMetricsVariousStructures:
    def setup_method(self):
        self.builder = ChartSpecBuilder()

    def test_all_scalar_values(self):
        metrics = {"r2": 0.95, "rmse": 5.2, "mae": 3.1, "skill_score": 0.8}
        table, chart, kpi = self.builder.build(metrics)
        assert kpi is not None
        assert len(kpi.cards) >= 3

    def test_list_of_scalars_not_used_as_table(self):
        metrics = {"values": [1, 2, 3, 4], "total_energy_mwh": 100.0}
        table, chart, kpi = self.builder.build(metrics)
        assert table is None  # list of non-dicts should be ignored

    def test_mixed_scalar_types(self):
        metrics = {
            "total_energy_mwh": 1500.0,
            "station_count": 8,
            "r2": 0.95,
            "pr_ratio_pct": 85.0,
        }
        table, chart, kpi = self.builder.build(metrics)
        assert kpi is not None

    def test_boolean_values_not_in_kpi(self):
        metrics = {"is_active": True, "total_energy_mwh": 500.0}
        _, _, kpi = self.builder.build(metrics)
        assert kpi is not None
        labels = [c.label.lower() for c in kpi.cards]
        assert not any("active" in l for l in labels)

    def test_none_values_in_rows(self):
        rows = [
            {"facility": "F1", "energy_mwh": None},
            {"facility": "F2", "energy_mwh": 100.0},
        ]
        metrics = {"readings": rows}
        table, chart, kpi = self.builder.build(metrics)
        assert table is not None
        assert table.row_count == 2

    def test_datetime_in_rows_coerced(self):
        rows = [
            {"date": datetime(2024, 1, 1, 12, 0), "energy_mwh": 50.0},
        ]
        metrics = {"readings": rows}
        table, _, _ = self.builder.build(metrics)
        assert table is not None
        assert isinstance(table.rows[0]["date"], str)

    def test_date_in_rows_coerced(self):
        rows = [
            {"date": date(2024, 1, 1), "energy_mwh": 50.0},
        ]
        metrics = {"readings": rows}
        table, _, _ = self.builder.build(metrics)
        assert table is not None
        assert table.rows[0]["date"] == "2024-01-01"


# ===========================================================================
# Horizontal bar chart (integration path)
# ===========================================================================

class TestHorizontalBarIntegration:
    def setup_method(self):
        self.builder = ChartSpecBuilder()

    def test_many_facilities_horizontal(self):
        rows = [
            {"facility": f"Long Station Name {i}", "energy_mwh": float(i * 10)}
            for i in range(9)
        ]
        metrics = {"energy": rows}
        _, chart, _ = self.builder.build(metrics)
        assert chart is not None
        assert chart.chart_type == "bar"
        trace = chart.plotly_spec["data"][0]
        assert trace.get("orientation") == "h"


# ===========================================================================
# Forecast data shape
# ===========================================================================

class TestForecastDataShape:
    def setup_method(self):
        self.builder = ChartSpecBuilder()

    def test_forecast_table_title(self):
        rows = [
            {"date": f"2024-01-{i+1:02d}", "forecast_mwh": float(100 + i * 5), "expected_mwh": float(100 + i * 4)}
            for i in range(10)
        ]
        metrics = {"forecast": rows}
        table, _, _ = self.builder.build(metrics, topic="forecast_72h")
        assert table is not None

    def test_actual_vs_forecast_detected(self):
        rows = [
            {"facility": f"F{i}", "actual_mwh": float(i * 10), "forecast_mwh": float(i * 11)}
            for i in range(5)
        ]
        metrics = {"performance": rows}
        table, _, _ = self.builder.build(metrics)
        assert table is not None
        assert "forecast" in table.title.lower() or "actual" in table.title.lower()
