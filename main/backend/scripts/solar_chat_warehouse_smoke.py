"""Smoke test: verify Solar Chat is wired to the configured Databricks SQL warehouse.

Usage (from dlh-pv-website/):
    python -m main.backend.scripts.solar_chat_warehouse_smoke
or  python main/backend/scripts/solar_chat_warehouse_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.settings import get_solar_chat_settings  # noqa: E402
from app.repositories.solar_ai_chat.base_repository import (  # noqa: E402
    BaseRepository,
    DatabricksDataUnavailableError,
)


def main() -> int:
    settings = get_solar_chat_settings()

    host = settings.solar_chat_databricks_host_resolved
    http_path = settings.solar_chat_databricks_http_path_resolved
    using_split = settings.use_separate_warehouse_for_solar_chat

    print(f"[settings] USE_SEPARATE_WAREHOUSE_FOR_SOLAR_CHAT={using_split}")
    print(f"[settings] resolved host       = {host}")
    print(f"[settings] resolved http_path  = {http_path}")
    print(f"[settings] catalog/silver/gold = {settings.uc_catalog}/{settings.uc_silver_schema}/{settings.uc_gold_schema}")

    if using_split:
        primary_path = settings.resolved_databricks_http_path
        if http_path == primary_path and primary_path is not None:
            print("[warn] solar-chat http_path equals primary; check SOLAR_CHAT_DATABRICKS_* values")

    repo = BaseRepository(settings)

    queries = [
        ("ping", "SELECT 1 AS ok"),
        ("silver.energy_readings count", f"SELECT COUNT(*) AS n FROM {settings.uc_catalog}.silver.energy_readings"),
        ("gold.dim_facility count", f"SELECT COUNT(*) AS n FROM {settings.uc_catalog}.gold.dim_facility"),
        ("gold.fact_energy latest", f"SELECT MAX(date_hour) AS latest FROM {settings.uc_catalog}.gold.fact_energy"),
    ]

    failures = 0
    for label, sql in queries:
        try:
            rows = repo._execute_query(sql)
            print(f"[ok]    {label}: {rows[:1]}")
        except DatabricksDataUnavailableError as exc:
            print(f"[fail]  {label}: {exc}")
            failures += 1
        except Exception as exc:
            print(f"[error] {label}: {type(exc).__name__}: {exc}")
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
