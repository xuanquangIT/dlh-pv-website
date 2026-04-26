"""Schemas for data visualization payloads returned by Solar AI Chat.

Inspired by Vanna AI's `DataFrameComponent` and `ChartComponent`, re-implemented
independently. These payloads attach to ``SolarChatResponse`` so the frontend can
render rich UI (sortable tables, Plotly charts, KPI cards) alongside the text
answer.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ColumnType = Literal["string", "number", "date", "datetime", "boolean"]


class DataTableColumn(BaseModel):
    key: str = Field(description="Row dictionary key.")
    label: str = Field(description="Human-readable header label.")
    type: ColumnType = Field(default="string", description="Value type for formatting / sorting.")
    unit: str | None = Field(default=None, description="Optional unit shown in the header, e.g. 'MWh', '%', 'C'.")


class DataTablePayload(BaseModel):
    """Tabular data ready for an interactive frontend table."""

    title: str
    description: str | None = None
    columns: list[DataTableColumn]
    rows: list[dict[str, Any]]
    row_count: int
    exportable: bool = True
    sortable: bool = True
    filterable: bool = True
    paginated: bool = True
    page_size: int = 25


class ChartPayload(BaseModel):
    """Chart specification built on the backend.

    v1 engine: emits Plotly (`format='plotly'`, `plotly_spec` populated).
    v2 engine: emits Vega-Lite (`format='vega-lite'`, `spec` populated).
    Frontend chart_renderer.js dispatches on `format`.
    """

    model_config = ConfigDict(protected_namespaces=())

    chart_type: str = Field(
        description=(
            "Chart kind. v1 Plotly: line|bar|pie|scatter|histogram|area|scatter_geo. "
            "v2 Vega-Lite: bar|line|geoshape|point|circle|... (any Vega-Lite mark)."
        ),
    )
    title: str
    description: str | None = None
    format: Literal["plotly", "vega-lite", "leaflet-map"] = Field(
        default="plotly",
        description="Renderer: 'plotly' (v1), 'vega-lite' (v2 charts), 'leaflet-map' (v2 geographic maps with native pan/zoom).",
    )
    plotly_spec: dict[str, Any] | None = Field(
        default=None,
        description="v1 only: Plotly.newPlot dict with `data` + `layout`.",
    )
    spec: dict[str, Any] | None = Field(
        default=None,
        description="v2 only: Vega-Lite spec with `mark` + `encoding` + `data.values`.",
    )
    points: list[dict[str, Any]] | None = Field(
        default=None,
        description="leaflet-map only: list of {lat, lng, label, size_value, attrs} points.",
    )
    size_field: str | None = Field(
        default=None,
        description="leaflet-map only: column name used to size circle markers (legend label).",
    )
    label_field: str | None = Field(
        default=None,
        description="leaflet-map only: column name used as the marker label / popup title.",
    )
    row_count: int | None = Field(
        default=None,
        description="v2: number of rows backing the chart (Plotly carries this in plotly_spec.data).",
    )
    source_metric_key: str | None = Field(
        default=None,
        description="Key inside `key_metrics` that the chart was derived from (for debugging).",
    )


class KpiCard(BaseModel):
    """A single KPI card: headline number + label."""

    label: str
    value: float | int | str
    unit: str | None = None
    format: Literal["number", "percent", "integer", "text"] = "number"
    trend: Literal["up", "down", "neutral"] | None = None
    trend_value: float | None = None
    description: str | None = None


class KpiCardsPayload(BaseModel):
    title: str | None = None
    cards: list[KpiCard]


__all__ = [
    "ColumnType",
    "DataTableColumn",
    "DataTablePayload",
    "ChartPayload",
    "KpiCard",
    "KpiCardsPayload",
]
