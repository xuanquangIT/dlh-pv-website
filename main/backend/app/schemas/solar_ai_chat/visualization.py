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
    """Plotly chart specification built on the backend; rendered by Plotly.js."""

    model_config = ConfigDict(protected_namespaces=())

    chart_type: Literal["line", "bar", "pie", "scatter", "histogram", "area"]
    title: str
    description: str | None = None
    plotly_spec: dict[str, Any] = Field(
        description="Dict with `data` (list of traces) and `layout` compatible with Plotly.newPlot."
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
