"""Smoke test the new range mode for fetch_station_hourly_report.

Asks for the hourly profile across the current month and prints a tiny summary.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.settings import get_solar_chat_settings  # noqa: E402
from app.repositories.solar_ai_chat.report_repository import ReportRepository  # noqa: E402


def main() -> int:
    settings = get_solar_chat_settings()
    repo = ReportRepository(settings)

    today = date.today()
    month_start = today.replace(day=1)

    # 1. Single-day backwards-compat
    single, _ = repo.fetch_station_hourly_report(station_name=None)
    print(f"[single] report_date={single.get('report_date')} rows={single.get('row_count')} total_mwh={single.get('total_energy_mwh')}")

    # 2. Range mode: this month
    rng, _ = repo.fetch_station_hourly_report(
        station_name=None,
        start_date=month_start,
        end_date=today,
    )
    print(
        f"[range ] period={rng.get('report_period')} days={rng.get('days_covered')} "
        f"rows={rng.get('row_count')} period_total_mwh={rng.get('period_total_mwh')} "
        f"agg={rng.get('aggregation')}"
    )
    facilities = sorted({r["facility"] for r in rng.get("hourly_rows", [])})
    print(f"[range ] facilities={facilities}")

    if facilities:
        sample = [r for r in rng["hourly_rows"] if r["facility"] == facilities[0]]
        print(f"[range ] {facilities[0]} hourly avg (first 6h):")
        for r in sample[:6]:
            print(f"           hr={r['hour']:>2}  energy_mwh_avg={r['energy_mwh']:>7.3f}  cf={r['capacity_factor_pct']}")

    return 0 if rng.get("has_data") else 1


if __name__ == "__main__":
    raise SystemExit(main())
