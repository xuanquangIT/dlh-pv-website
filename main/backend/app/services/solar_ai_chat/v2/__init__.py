"""Solar Chat v2 — primitives + semantic layer architecture.

Design doc: implementations/solar_chat_architecture_redesign_2026-04-26.md

This package replaces the v1 14-hardcoded-tools approach with 5 generic
primitives the LLM composes itself:

    discover_schema       — list tables filtered by domain
    inspect_table         — show columns, types, sample rows
    recall_metric         — semantic search over canonical SQL templates
    execute_sql           — read-only SELECT with safety guards
    render_visualization  — Vega-Lite spec → chart artifact

Status: Phase 4.5 — sole runtime path. The v1 surface and the unused
search_documents/RAG primitive have been removed.
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
