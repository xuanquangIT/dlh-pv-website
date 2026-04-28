"""Solar AI Chat — the 5 generic primitives the LLM composes.

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

from app.services.solar_ai_chat.semantic_loader import (
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

_STOPWORDS: frozenset[str] = frozenset({
    # English filler / interrogative words that show up in sample questions
    # without carrying domain meaning. Keep this list short — words that
    # legitimately discriminate metrics (e.g. "energy", "facility", "weather")
    # MUST NOT be here.
    "the", "and", "for", "with", "this", "that", "from", "have", "are", "you",
    "all", "any", "now", "today", "show", "give", "tell", "what", "whats",
    "let", "can", "could", "would", "may", "much", "many", "how", "why",
    "when", "where", "which", "who", "whom", "into", "over", "out", "off",
    "one", "two", "ten", "yes", "see", "got", "get", "use",
    # Vietnamese function words / interrogatives (with + without diacritics)
    "cho", "tôi", "toi", "bạn", "ban", "các", "cac", "của", "cua",
    "là", "la", "có", "co", "không", "khong", "này", "nay", "đó", "do",
    "với", "voi", "đang", "dang", "đã", "da", "sẽ", "se",
    "và", "va", "hay", "thì", "thi", "rồi", "roi", "mà", "ma",
    "hiện", "hien", "tại", "tai", "bao", "nhiêu", "nhieu",
    "thông", "thong", "tin", "từng", "tung", "mỗi", "moi",
    "trong", "trên", "tren", "dưới", "duoi", "ngoài", "ngoai",
    "đây", "day", "đấy", "day2", "kia", "ai", "gì", "gi",
    "ra", "vào", "vao", "lên", "len", "xuống", "xuong",
    "nó", "no", "nào", "nao", "đến", "den",
})


def _score_metric_relevance(query: str, metric: MetricDefinition) -> float:
    """Lightweight relevance score (0-1+). Combines:
      - token overlap against name + description
      - synonym phrase boost (VN/EN keywords from YAML)
      - sample-question phrase boost
    Score can exceed 1.0 when multiple signals fire — sort still works.

    Replace with embedding RAG once Phase 3 cutover lands at scale.
    """
    q = query.lower()
    # Include synonyms in the searchable text so a query like "summarize"
    # matches a metric whose synonym list includes "summary" / "tóm tắt".
    text_parts = [metric.name, metric.description, *metric.sample_questions, *metric.synonyms]
    text = " ".join(text_parts).lower()

    # Token overlap (skip stopwords < 3 chars). Use word-boundary matching so
    # "all" doesn't match "overall" and "map" doesn't match "summarize".
    # Filter Vietnamese + English function words on BOTH sides so a query
    # full of "hiện tại / bao nhiêu / đang / của" doesn't incidentally
    # match a metric whose sample question contains the same fillers.
    raw_tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
    tokens = [t for t in raw_tokens if t not in _STOPWORDS]
    if not tokens:
        # Fall back to raw tokens if the query is *entirely* stopwords
        # (rare — happens for greetings only).
        tokens = raw_tokens
        if not tokens:
            return 0.0
    text_tokens = set(re.findall(r"\w+", text)) - _STOPWORDS
    hits = sum(1 for t in tokens if t in text_tokens)
    base = hits / len(tokens)

    # Synonym phrase boost — each matched synonym phrase adds 0.5. Use
    # word-boundary regex to avoid "map" matching "summary".
    synonym_boost = 0.0
    for syn in metric.synonyms:
        if not syn:
            continue
        s = syn.lower()
        if " " in s:                   # multi-word phrase: substring is fine
            if s in q:
                synonym_boost += 0.5
        else:
            if re.search(rf"\b{re.escape(s)}\b", q):
                synonym_boost += 0.5

    # Sample-question phrase overlap — only count distinctive words (skip the
    # generic "all"/"the"/"show" set that fires on every English sample).
    sample_boost = 0.0
    GENERIC = {"all", "the", "show", "give", "tell", "what", "what's", "for",
               "and", "with", "this", "that", "from", "now", "today", "have",
               "you", "can", "cho", "toi", "tôi", "xem", "các", "cac"}
    q_set = set(tokens)
    for sample in metric.sample_questions:
        sample_tokens = [
            t for t in re.findall(r"\w+", sample.lower())
            if len(t) >= 3 and t not in GENERIC
        ]
        if len(sample_tokens) < 2:
            continue
        overlap = sum(1 for t in sample_tokens if t in q_set)
        if overlap >= 2:
            sample_boost = max(sample_boost, overlap / len(sample_tokens))

    return base + synonym_boost + sample_boost


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

    matches = [
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
            "synonyms": list(m.synonyms),
            "sample_questions": list(m.sample_questions),
        }
        for m, score in top
        if score > 0  # don't return 0-score noise
    ]
    if matches:
        next_action = (
            f"Use the top match's sql_template ({matches[0]['name']}). "
            "Substitute parameter placeholders like {window_days} with actual "
            "values, then call execute_sql with the resulting SQL string. "
            "Do NOT call recall_metric again — pick a match and run it."
        )
    else:
        next_action = (
            "No metric matched. Call discover_schema(domain=...) to list "
            "tables, then inspect_table on a likely table, then write your "
            "own SELECT for execute_sql."
        )
    return {
        "matches": matches,
        "query": query,
        "filtered_by_role": role_id,
        "next_action": next_action,
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
      6. Must reference at least one real lakehouse table (pv.<schema>.<table>)
         — blocks `SELECT * FROM (VALUES ...)` literals where weak models
         hallucinate facility data and pass it off as real.
    """
    violations: list[str] = []
    s = sql.strip().rstrip(";").strip()

    # Strip leading SQL comments (-- line comments and /* ... */ block comments)
    # before the prefix check. The LLM frequently copies metric templates that
    # begin with explanatory comments — those are valid SQL and shouldn't be
    # rejected as "doesn't start with SELECT/WITH".
    def _strip_leading_sql_comments(text: str) -> str:
        while True:
            text = text.lstrip()
            if text.startswith("--"):
                nl = text.find("\n")
                text = text[nl + 1:] if nl != -1 else ""
                continue
            if text.startswith("/*"):
                end = text.find("*/")
                text = text[end + 2:] if end != -1 else ""
                continue
            return text

    s_for_check = _strip_leading_sql_comments(s)

    if ";" in s:
        violations.append("Stacked queries (multiple statements via ';') not allowed.")

    head = s_for_check.split(None, 1)[0].upper() if s_for_check else ""
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

    # Real-table requirement. Look for at least one FROM/JOIN pv.<schema>.<table>
    # pattern. Reject if missing — the model is hallucinating data via a
    # VALUES literal or a CTE that never reads from the lakehouse.
    if not re.search(r"\b(?:FROM|JOIN)\s+pv\.\w+\.\w+", s, re.IGNORECASE):
        violations.append(
            "SQL must reference at least one lakehouse table (pv.<schema>.<table>). "
            "VALUES literals and synthetic CTEs are not allowed — quote real data only."
        )

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
        "mark": mark_type,
    }
