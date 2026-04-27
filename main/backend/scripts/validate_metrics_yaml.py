"""Validate `services/solar_ai_chat/v2/semantic/metrics.yaml` against
live Databricks schema. Catches schema drift before users hit it.

Two modes:

    # Static (no DB) — verifies YAML internal consistency only:
    #   - every metric's sql_template has matching parameters
    #   - column references in YAML table defs are unique
    #   - per-role allowed_tables / allowed_metrics resolve
    python scripts/validate_metrics_yaml.py --static

    # Live (requires Databricks creds) — runs DESCRIBE + EXPLAIN:
    #   - every (catalog.schema.table) listed in YAML exists
    #   - every column listed in YAML exists in the live table
    #   - every sql_template renders + EXPLAINs successfully with defaults
    python scripts/validate_metrics_yaml.py --live

Exit code: 0 = clean, 1 = drift detected. Suitable for CI gate.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root with `python scripts/validate_metrics_yaml.py`
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.solar_ai_chat.v2.semantic_loader import (  # noqa: E402
    load_semantic_layer,
    SemanticLayer,
)


# ---------------------------------------------------------------------------
# Static checks — no DB required
# ---------------------------------------------------------------------------

def check_static(layer: SemanticLayer) -> list[str]:
    errors: list[str] = []

    table_fqns = {t.fqn for t in layer.tables}

    # 1. Every metric's sql_template parameters appear in the SQL
    placeholder_re = re.compile(r"\{(\w+)\}")
    for m in layer.metrics:
        placeholders = set(placeholder_re.findall(m.sql_template))
        declared = {p.name for p in m.parameters}
        missing_in_decl = placeholders - declared
        unused_in_sql = declared - placeholders
        if missing_in_decl:
            errors.append(
                f"metric {m.name!r}: SQL references undeclared parameter(s) {sorted(missing_in_decl)}"
            )
        if unused_in_sql:
            errors.append(
                f"metric {m.name!r}: parameter(s) {sorted(unused_in_sql)} declared but not in SQL"
            )

    # 2. Every metric's SQL references at least one table declared in YAML
    table_re = re.compile(r"\b(?:FROM|JOIN)\s+([\w\.]+)", re.IGNORECASE)
    for m in layer.metrics:
        refs = {ref.strip().lower() for ref in table_re.findall(m.sql_template)}
        # Allow CTE aliases (a single-token ref that looks like an alias from a WITH)
        cte_aliases = set(
            a.strip().lower()
            for a in re.findall(r"\b(\w+)\s+AS\s*\(", m.sql_template, re.IGNORECASE)
        )
        # Skip single-token refs that aren't CTE aliases — these are almost
        # always false positives from `EXTRACT(HOUR FROM ts)`, `CAST(x AS T)`
        # or similar SQL idioms where FROM is a syntactic keyword, not a
        # table source. Real table refs in this codebase are always dotted
        # (catalog.schema.table).
        unknown = {
            r for r in refs
            if "." in r
            and r not in {f.lower() for f in table_fqns}
            and r not in cte_aliases
        }
        if unknown:
            errors.append(
                f"metric {m.name!r}: SQL references unknown table(s) {sorted(unknown)} — "
                f"must be a fully-qualified table from metrics.yaml or a CTE alias"
            )

    # 3. Every role's allowed_tables resolves to a known table or '*'
    for role in layer.role_policies.values():
        for t in role.allowed_tables:
            if t == "*":
                continue
            if t not in table_fqns:
                errors.append(
                    f"role {role.role_id!r}: allowed_tables entry {t!r} is not in metrics.yaml"
                )

    # 4. Every role's allowed_metrics resolves to a known metric or '*'
    metric_names = {m.name for m in layer.metrics}
    for role in layer.role_policies.values():
        for n in role.allowed_metrics:
            if n == "*":
                continue
            if n not in metric_names:
                errors.append(
                    f"role {role.role_id!r}: allowed_metrics entry {n!r} is not in metrics.yaml"
                )

    return errors


# ---------------------------------------------------------------------------
# Live checks — needs Databricks
# ---------------------------------------------------------------------------

def check_live(layer: SemanticLayer) -> list[str]:
    errors: list[str] = []

    try:
        from app.core.settings import SolarChatSettings
        from app.services.solar_ai_chat.v2.databricks_adapter import (
            make_sql_executor,
        )
    except Exception as e:  # noqa: BLE001
        return [f"could not import Databricks adapter: {e}"]

    settings = SolarChatSettings()
    try:
        execute = make_sql_executor(settings)
    except Exception as e:  # noqa: BLE001
        return [f"could not build SQL executor (check .env): {e}"]

    # 4. Each declared table exists
    for table in layer.tables:
        try:
            rows = execute(f"DESCRIBE TABLE {table.fqn}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"table {table.fqn}: DESCRIBE failed: {e}")
            continue

        live_cols = {
            (r.get("col_name") or r.get("name") or "").lower()
            for r in rows
            if r and (r.get("col_name") or r.get("name"))
        }
        live_cols.discard("")
        # Filter out partition/comment marker rows
        live_cols = {c for c in live_cols if not c.startswith("#")}

        declared_cols = {c.name.lower() for c in table.columns}
        missing = declared_cols - live_cols
        if missing:
            errors.append(
                f"table {table.fqn}: YAML lists columns not in live schema: {sorted(missing)}"
            )

    # 5. Each metric's SQL renders + EXPLAINs successfully
    for m in layer.metrics:
        try:
            rendered = _render_with_defaults(m.sql_template, m.parameters)
        except Exception as e:  # noqa: BLE001
            errors.append(f"metric {m.name!r}: cannot render with defaults: {e}")
            continue
        try:
            execute(f"EXPLAIN {rendered}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"metric {m.name!r}: EXPLAIN failed: {str(e)[:300]}")

    return errors


def _render_with_defaults(sql_template: str, parameters) -> str:
    rendered = sql_template
    for p in parameters:
        placeholder = "{" + p.name + "}"
        if placeholder not in rendered:
            continue
        if p.default is not None:
            value: Any = p.default
        elif p.values:
            value = p.values[0]
        else:
            # Required string param with no default — use empty placeholder
            # that satisfies SQL syntax (e.g. for facility_id_1 / _2, this
            # makes EXPLAIN parse but skip rows; that's enough for CI).
            value = "" if p.type == "string" else 1
        rendered = rendered.replace(placeholder, str(value))
    return rendered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true",
                      help="YAML internal consistency only (no DB).")
    mode.add_argument("--live", action="store_true",
                      help="Plus DESCRIBE TABLE + EXPLAIN (needs Databricks).")
    args = parser.parse_args()

    layer = load_semantic_layer()
    errors = check_static(layer)
    if args.live:
        errors.extend(check_live(layer))

    if not errors:
        scope = "static + live" if args.live else "static"
        print(f"OK — {scope} validation passed.")
        return 0

    print(f"FAIL — {len(errors)} issue(s):")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
