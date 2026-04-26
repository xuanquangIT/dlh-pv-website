"""Solar Chat v2 — primitives + semantic layer architecture.

Design doc: implementations/solar_chat_architecture_redesign_2026-04-26.md

This package replaces the v1 14-hardcoded-tools approach with 6 generic
primitives the LLM composes itself:

    discover_schema       — list tables filtered by domain
    inspect_table         — show columns, types, sample rows
    recall_metric         — semantic search over canonical SQL templates
    execute_sql           — read-only SELECT with safety guards
    render_visualization  — Vega-Lite spec → chart artifact
    search_documents      — RAG over manuals/incidents (re-used from v1)

Status: Phase 1 prototype. Behind SOLAR_CHAT_ENGINE=v2 feature flag.
NOT production-ready. Run side-by-side with v1, judge with eval CLI,
cut over only after accuracy parity is confirmed.
"""
from app.services.solar_ai_chat.v2.semantic_loader import (
    SemanticLayer,
    MetricDefinition,
    TableDefinition,
    load_semantic_layer,
)
from app.services.solar_ai_chat.v2.primitives import (
    SqlSafetyError,
    discover_schema,
    inspect_table,
    recall_metric,
    execute_sql,
    render_visualization,
)

__all__ = [
    "SemanticLayer",
    "MetricDefinition",
    "TableDefinition",
    "SqlSafetyError",
    "discover_schema",
    "execute_sql",
    "inspect_table",
    "load_semantic_layer",
    "recall_metric",
    "render_visualization",
]
