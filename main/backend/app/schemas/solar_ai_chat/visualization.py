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

    The engine emits Vega-Lite (`format='vega-lite'`, `spec` populated)
    or Leaflet maps (`format='leaflet-map'`, `points` populated). Frontend
    `chart_renderer.js` dispatches on `format`. Plotly fields stay on the
    model so old persisted snapshots keep deserialising.
    """

    model_config = ConfigDict(protected_namespaces=())

    chart_type: str = Field(
        description=(
            "Chart kind: bar|line|scatter|point|circle|area|geoshape|map|"
            "histogram|pie|scatter_geo. Vega-Lite renders any built-in mark."
        ),
    )
    title: str
    description: str | None = None
    format: Literal["plotly", "vega-lite", "leaflet-map"] = Field(
        default="vega-lite",
        description="Renderer: 'vega-lite' (default), 'leaflet-map' (geo maps with native pan/zoom), 'plotly' (legacy snapshots only).",
    )
    plotly_spec: dict[str, Any] | None = Field(
        default=None,
        description="Legacy Plotly.newPlot dict; populated only on stored snapshots from before the Vega-Lite cutover.",
    )
    spec: dict[str, Any] | None = Field(
        default=None,
        description="Vega-Lite spec with `mark` + `encoding` + `data.values`.",
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
        description="Number of rows backing the chart.",
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
