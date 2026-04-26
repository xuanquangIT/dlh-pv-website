"""Solar Chat v2 — semantic layer loader.

Reads `semantic/metrics.yaml` and exposes typed accessors for primitives
(discover_schema, recall_metric, etc.).

The YAML is loaded once at startup (lru_cache) — operators edit YAML, restart
server. For hot-reload, call invalidate_cache().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_YAML_PATH = Path(__file__).parent / "semantic" / "metrics.yaml"


@dataclass(frozen=True)
class TableColumn:
    name: str
    type: str
    description: str = ""


@dataclass(frozen=True)
class TableDefinition:
    catalog: str
    schema: str
    name: str
    description: str
    grain: tuple[str, ...]
    columns: tuple[TableColumn, ...]
    primary_key: tuple[str, ...] = ()
    sample_questions: tuple[str, ...] = ()

    @property
    def fqn(self) -> str:
        """Fully-qualified table name: catalog.schema.name."""
        return f"{self.catalog}.{self.schema}.{self.name}"

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class MetricParameter:
    name: str
    type: str
    default: Any = None
    values: tuple[str, ...] = ()         # for enum
    range: tuple[Any, Any] | None = None # for int/float bounds


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    sql_template: str
    parameters: tuple[MetricParameter, ...] = ()
    suggested_chart: dict[str, Any] | None = None
    suggested_kpi_cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class RolePolicy:
    role_id: str
    description: str
    allowed_tables: tuple[str, ...]   # FQNs or "*"
    allowed_metrics: tuple[str, ...]  # metric names or "*"

    def can_access_table(self, fqn: str) -> bool:
        if "*" in self.allowed_tables:
            return True
        return fqn in self.allowed_tables

    def can_call_metric(self, metric_name: str) -> bool:
        if "*" in self.allowed_metrics:
            return True
        return metric_name in self.allowed_metrics


@dataclass(frozen=True)
class SemanticLayer:
    version: int
    tables: tuple[TableDefinition, ...]
    metrics: tuple[MetricDefinition, ...]
    role_policies: dict[str, RolePolicy] = field(default_factory=dict)

    def get_table(self, fqn: str) -> TableDefinition | None:
        for t in self.tables:
            if t.fqn == fqn:
                return t
        return None

    def get_metric(self, name: str) -> MetricDefinition | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def tables_for_role(self, role_id: str) -> tuple[TableDefinition, ...]:
        policy = self.role_policies.get(role_id)
        if policy is None:
            return ()
        return tuple(t for t in self.tables if policy.can_access_table(t.fqn))

    def metrics_for_role(self, role_id: str) -> tuple[MetricDefinition, ...]:
        policy = self.role_policies.get(role_id)
        if policy is None:
            return ()
        return tuple(m for m in self.metrics if policy.can_call_metric(m.name))


def _parse_columns(raw: list[Any]) -> tuple[TableColumn, ...]:
    out: list[TableColumn] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        out.append(TableColumn(
            name=str(item.get("name", "")),
            type=str(item.get("type", "string")),
            description=str(item.get("description", "")),
        ))
    return tuple(out)


def _parse_parameters(raw: list[Any]) -> tuple[MetricParameter, ...]:
    out: list[MetricParameter] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        rng = item.get("range")
        rng_tuple = tuple(rng) if isinstance(rng, list) and len(rng) == 2 else None
        values = item.get("values") or []
        out.append(MetricParameter(
            name=str(item.get("name", "")),
            type=str(item.get("type", "string")),
            default=item.get("default"),
            values=tuple(str(v) for v in values),
            range=rng_tuple,
        ))
    return tuple(out)


def _parse_role_policies(raw: dict[str, Any]) -> dict[str, RolePolicy]:
    out: dict[str, RolePolicy] = {}
    for role_id, body in (raw or {}).items():
        if not isinstance(body, dict):
            continue
        out[role_id] = RolePolicy(
            role_id=role_id,
            description=str(body.get("description", "")),
            allowed_tables=tuple(body.get("allowed_tables") or []),
            allowed_metrics=tuple(body.get("allowed_metrics") or []),
        )
    return out


def _load_from_yaml(path: Path) -> SemanticLayer:
    if not path.is_file():
        raise FileNotFoundError(f"Semantic YAML not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    tables: list[TableDefinition] = []
    for catalog_name, catalog in (raw.get("catalogs") or {}).items():
        for schema_name, schema in (catalog.get("schemas") or {}).items():
            for table_name, table in (schema.get("tables") or {}).items():
                if not isinstance(table, dict):
                    continue
                tables.append(TableDefinition(
                    catalog=str(catalog_name),
                    schema=str(schema_name),
                    name=str(table_name),
                    description=str(table.get("description", "")),
                    grain=tuple(table.get("grain") or []),
                    columns=_parse_columns(table.get("columns") or []),
                    primary_key=tuple(
                        [table["primary_key"]] if isinstance(table.get("primary_key"), str)
                        else (table.get("primary_key") or [])
                    ),
                    sample_questions=tuple(table.get("sample_questions") or []),
                ))

    metrics: list[MetricDefinition] = []
    for metric_name, metric in (raw.get("metrics") or {}).items():
        if not isinstance(metric, dict):
            continue
        metrics.append(MetricDefinition(
            name=str(metric_name),
            description=str(metric.get("description", "")),
            sql_template=str(metric.get("sql_template", "")).strip(),
            parameters=_parse_parameters(metric.get("parameters") or []),
            suggested_chart=metric.get("suggested_chart"),
            suggested_kpi_cards=tuple(metric.get("suggested_kpi_cards") or []),
        ))

    return SemanticLayer(
        version=int(raw.get("version", 1)),
        tables=tuple(tables),
        metrics=tuple(metrics),
        role_policies=_parse_role_policies(raw.get("roles") or {}),
    )


@lru_cache(maxsize=1)
def load_semantic_layer(path: str | None = None) -> SemanticLayer:
    """Load and cache the semantic layer. Pass a custom path for tests."""
    yaml_path = Path(path) if path else DEFAULT_YAML_PATH
    layer = _load_from_yaml(yaml_path)
    logger.info(
        "semantic_layer_loaded version=%d tables=%d metrics=%d roles=%d",
        layer.version, len(layer.tables), len(layer.metrics), len(layer.role_policies),
    )
    return layer


def invalidate_cache() -> None:
    load_semantic_layer.cache_clear()
