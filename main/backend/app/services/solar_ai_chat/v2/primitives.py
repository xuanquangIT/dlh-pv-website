"""Solar Chat v2 — the 6 generic primitives the LLM composes.

Each primitive is a pure function: input dict → output dict (or DataFrame).
No hidden state, no per-question hardcoding. The LLM picks which primitive
to call based on schema descriptions in the semantic layer.

Design doc: implementations/solar_chat_architecture_redesign_2026-04-26.md
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.solar_ai_chat.v2.semantic_loader import (
    MetricDefinition,
    SemanticLayer,
    TableDefinition,
    load_semantic_layer,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & errors
# -----------------------------------------------------------------------------

DEFAULT_MAX_ROWS = 1000
HARD_MAX_ROWS = 10000

# SQL safety: only SELECT/WITH allowed; no DDL/DML.
_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "TRUNCATE", "ALTER",
    "CREATE", "GRANT", "REVOKE", "COPY", "ATTACH", "DETACH", "USE",
    "REFRESH", "CALL", "EXECUTE", "REPLACE", "OPTIMIZE", "VACUUM",
)
_ALLOWED_SQL_PREFIXES = ("SELECT", "WITH")
# System catalogs we never want to expose
_BLOCKED_CATALOG_PREFIXES = ("information_schema", "system", "pg_catalog")


class SqlSafetyError(ValueError):
    """Raised when execute_sql receives a query that fails safety checks."""


# -----------------------------------------------------------------------------
# Primitive 1: discover_schema(domain?)
# -----------------------------------------------------------------------------

def discover_schema(
    *,
    role_id: str = "admin",
    domain: str | None = None,
    semantic_layer: SemanticLayer | None = None,
) -> dict[str, Any]:
    """List tables visible to ``role_id``, optionally filtered by ``domain``
    keyword (matches table description / schema name).

    Returns:
        {
          "tables": [
            {"fqn": "pv.gold.dim_facility", "description": "...",
             "column_count": 11, "sample_questions": [...]},
            ...
          ],
          "total": N,
          "filtered_by_role": "admin",
          "filtered_by_domain": "weather"
        }
    """
    layer = semantic_layer or load_semantic_layer()
    role_tables = layer.tables_for_role(role_id)
    if not role_tables:
        return {"tables": [], "total": 0, "filtered_by_role": role_id,
                "filtered_by_domain": domain,
                "message": f"Role {role_id!r} has no table access."}

    if domain:
        d = domain.lower()
        filtered = tuple(
            t for t in role_tables
            if d in t.description.lower()
            or d in t.schema.lower()
            or d in t.name.lower()
        )
    else:
        filtered = role_tables

    return {
        "tables": [
            {
                "fqn": t.fqn,
                "description": t.description,
                "grain": list(t.grain),
                "column_count": len(t.columns),
                "sample_questions": list(t.sample_questions),
            }
            for t in filtered
        ],
        "total": len(filtered),
        "filtered_by_role": role_id,
        "filtered_by_domain": domain,
    }


# -----------------------------------------------------------------------------
# Primitive 2: inspect_table(table_fqn)
# -----------------------------------------------------------------------------

def inspect_table(
    *,
    table_fqn: str,
    role_id: str = "admin",
    sample_executor: Callable[[str], list[dict[str, Any]]] | None = None,
    semantic_layer: SemanticLayer | None = None,
) -> dict[str, Any]:
    """Show columns + types + (optionally) sample rows for one table.

    `sample_executor` is a callable that runs `SELECT * FROM {fqn} LIMIT 3` —
    pass a Databricks-backed executor in production, or None to skip sampling.

    Returns:
        {"fqn": "...", "description": "...", "columns": [...],
         "primary_key": [...], "sample_rows": [...]}
    """
    layer = semantic_layer or load_semantic_layer()
    table = layer.get_table(table_fqn)
    if table is None:
        return {"error": f"Table {table_fqn!r} not in semantic layer.",
                "available_tables": [t.fqn for t in layer.tables]}

    policy = layer.role_policies.get(role_id)
    if policy and not policy.can_access_table(table_fqn):
        return {"error": f"Role {role_id!r} not allowed to inspect {table_fqn!r}."}

    sample_rows: list[dict[str, Any]] = []
    sample_error: str | None = None
    if sample_executor is not None:
        try:
            sample_sql = f"SELECT * FROM {table_fqn} LIMIT 3"
            sample_rows = list(sample_executor(sample_sql) or [])
        except Exception as exc:  # noqa: BLE001 - report any sample failure
            sample_error = f"{type(exc).__name__}: {exc}"

    return {
        "fqn": table.fqn,
        "description": table.description,
        "grain": list(table.grain),
        "primary_key": list(table.primary_key),
        "columns": [
            {"name": c.name, "type": c.type, "description": c.description}
            for c in table.columns
        ],
        "sample_rows": sample_rows,
        "sample_error": sample_error,
        "sample_questions": list(table.sample_questions),
    }


# -----------------------------------------------------------------------------
# Primitive 3: recall_metric(query)
# -----------------------------------------------------------------------------

def _score_metric_relevance(query: str, metric: MetricDefinition) -> float:
    """Lightweight keyword overlap score (0-1). Replace with embedding RAG
    once the v2 cutover is happening at scale."""
    q = query.lower()
    text = (metric.name + " " + metric.description).lower()
    hits = 0
    total = 0
    for token in re.findall(r"\w+", q):
        if len(token) < 3:
            continue
        total += 1
        if token in text:
            hits += 1
    return hits / total if total else 0.0


def recall_metric(
    *,
    query: str,
    role_id: str = "admin",
    top_k: int = 5,
    semantic_layer: SemanticLayer | None = None,
) -> dict[str, Any]:
    """Find canonical metrics matching the user's intent. Returns the top-K
    metric definitions with their SQL templates, parameters, and suggested
    chart specs. The LLM picks one and fills in parameters.

    Returns:
        {"matches": [{"name": "...", "score": 0.6, "description": "...",
                      "sql_template": "...", "parameters": [...],
                      "suggested_chart": {...}, "suggested_kpi_cards": [...]}, ...]}
    """
    layer = semantic_layer or load_semantic_layer()
    role_metrics = layer.metrics_for_role(role_id)
    scored = [
        (m, _score_metric_relevance(query, m))
        for m in role_metrics
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    return {
        "matches": [
            {
                "name": m.name,
                "score": round(score, 3),
                "description": m.description,
                "sql_template": m.sql_template,
                "parameters": [
                    {
                        "name": p.name, "type": p.type, "default": p.default,
                        "values": list(p.values), "range": list(p.range) if p.range else None,
                    }
                    for p in m.parameters
                ],
                "suggested_chart": m.suggested_chart,
                "suggested_kpi_cards": list(m.suggested_kpi_cards),
            }
            for m, score in top
            if score > 0  # don't return 0-score noise
        ],
        "query": query,
        "filtered_by_role": role_id,
    }


# -----------------------------------------------------------------------------
# Primitive 4: execute_sql(sql, max_rows)
# -----------------------------------------------------------------------------

@dataclass
class SqlValidationResult:
    safe: bool
    normalized_sql: str
    violations: list[str] = field(default_factory=list)
    auto_limit_applied: bool = False


def validate_sql(sql: str, *, max_rows: int = DEFAULT_MAX_ROWS) -> SqlValidationResult:
    """Static safety check. Lightweight regex-based — for a real refactor,
    swap in sqlglot AST parsing for stronger guarantees.

    Rules enforced:
      1. Must start with SELECT or WITH (case-insensitive)
      2. No DDL/DML keywords as standalone tokens
      3. No system catalog references (information_schema, system.*, pg_catalog)
      4. Append LIMIT N if missing
      5. No semicolons (prevents stacked queries)
    """
    violations: list[str] = []
    s = sql.strip().rstrip(";").strip()

    if ";" in s:
        violations.append("Stacked queries (multiple statements via ';') not allowed.")

    head = s.split(None, 1)[0].upper() if s else ""
    if head not in _ALLOWED_SQL_PREFIXES:
        violations.append(
            f"Query must start with SELECT or WITH; got {head!r}."
        )

    upper = re.sub(r"\s+", " ", s.upper())
    for kw in _FORBIDDEN_SQL_KEYWORDS:
        # match as standalone word (avoid false positives in identifiers)
        if re.search(rf"\b{kw}\b", upper):
            violations.append(f"Forbidden SQL keyword: {kw}")

    for blocked in _BLOCKED_CATALOG_PREFIXES:
        if re.search(rf"\b{re.escape(blocked)}\b", s, re.IGNORECASE):
            violations.append(f"Access to {blocked} catalog/schema is blocked.")

    # Auto-LIMIT
    auto_limit = False
    if not re.search(r"\bLIMIT\b\s+\d+", upper):
        capped = min(max_rows, HARD_MAX_ROWS)
        s = f"{s}\nLIMIT {capped}"
        auto_limit = True

    return SqlValidationResult(
        safe=not violations,
        normalized_sql=s,
        violations=violations,
        auto_limit_applied=auto_limit,
    )


def execute_sql(
    *,
    sql: str,
    role_id: str = "admin",
    max_rows: int = DEFAULT_MAX_ROWS,
    sql_executor: Callable[[str], list[dict[str, Any]]] | None = None,
    semantic_layer: SemanticLayer | None = None,
) -> dict[str, Any]:
    """Validate and run a read-only SELECT. Caller passes ``sql_executor``
    (a Databricks-backed callable). Returns rows + metadata, or an
    error dict the LLM can react to.

    Returns:
        {"rows": [...], "row_count": N, "columns": [...],
         "executed_sql": "...", "auto_limit_applied": bool,
         "duration_ms": int}
        OR
        {"error": "...", "violations": [...]}
    """
    if not sql or not sql.strip():
        return {"error": "Empty SQL.", "violations": ["sql is empty"]}

    if max_rows > HARD_MAX_ROWS:
        max_rows = HARD_MAX_ROWS

    validation = validate_sql(sql, max_rows=max_rows)
    if not validation.safe:
        logger.warning(
            "execute_sql_blocked role=%s violations=%s sql=%r",
            role_id, validation.violations, sql[:200],
        )
        return {
            "error": "SQL failed safety validation.",
            "violations": validation.violations,
            "guidance": (
                "Rewrite the query to be a single read-only SELECT. "
                "No DDL/DML. No semicolons. No system catalogs."
            ),
        }

    if sql_executor is None:
        return {
            "error": "No sql_executor configured.",
            "executed_sql": validation.normalized_sql,
            "auto_limit_applied": validation.auto_limit_applied,
            "guidance": (
                "Pass sql_executor=<callable returning list[dict]> when "
                "calling execute_sql in production."
            ),
        }

    import time as _time
    started = _time.time()
    try:
        rows = list(sql_executor(validation.normalized_sql) or [])
    except Exception as exc:  # noqa: BLE001 - return error to LLM
        logger.warning(
            "execute_sql_runtime_error role=%s err=%s sql=%r",
            role_id, exc, validation.normalized_sql[:300],
        )
        return {
            "error": f"SQL execution failed: {type(exc).__name__}: {exc}",
            "executed_sql": validation.normalized_sql,
            "guidance": (
                "Inspect the error. Common causes: wrong column name "
                "(use inspect_table), missing JOIN, or warehouse unavailable."
            ),
        }
    duration_ms = int((_time.time() - started) * 1000)

    columns = list(rows[0].keys()) if rows else []
    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": columns,
        "executed_sql": validation.normalized_sql,
        "auto_limit_applied": validation.auto_limit_applied,
        "duration_ms": duration_ms,
    }


# -----------------------------------------------------------------------------
# Primitive 5: render_visualization(spec, data)
# -----------------------------------------------------------------------------

# v2 emits Vega-Lite spec; v1's chart_service is replaced.
# For Phase 1 prototype we pass the spec through and tag it for the frontend.

_SUPPORTED_VEGA_MARKS = (
    "bar", "line", "area", "point", "circle", "square",
    "tick", "rect", "rule", "text", "geoshape", "arc",
)


def render_visualization(
    *,
    spec: dict[str, Any],
    data: list[dict[str, Any]],
    title: str | None = None,
) -> dict[str, Any]:
    """Combine a Vega-Lite spec with row data into a chart artifact the
    frontend renders via vega-embed.

    The LLM is responsible for emitting a spec that fits the data —
    inspect_table / sql columns help it reason. Heuristic auto-detection
    is REMOVED in v2 (was the source of "wrong chart type" bugs).

    Returns:
        {"format": "vega-lite", "spec": {...with data injected...},
         "row_count": N, "title": "..."}
        OR
        {"error": "...", "guidance": "..."}
    """
    if not isinstance(spec, dict):
        return {"error": "spec must be a dict (Vega-Lite spec)",
                "guidance": "See https://vega.github.io/vega-lite/docs/"}

    mark = spec.get("mark")
    mark_type = mark if isinstance(mark, str) else (mark or {}).get("type") if isinstance(mark, dict) else None
    if mark_type and mark_type not in _SUPPORTED_VEGA_MARKS:
        return {"error": f"Unsupported mark type: {mark_type!r}",
                "guidance": f"Supported: {', '.join(_SUPPORTED_VEGA_MARKS)}"}

    full_spec = dict(spec)
    full_spec["data"] = {"values": data}
    if title and "title" not in full_spec:
        full_spec["title"] = title

    return {
        "format": "vega-lite",
        "spec": full_spec,
        "row_count": len(data),
        "title": title or full_spec.get("title", ""),
    }
