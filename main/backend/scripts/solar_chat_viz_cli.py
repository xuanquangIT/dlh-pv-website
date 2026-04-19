"""CLI tool to validate the Solar AI Chat Data Visualization pipeline (Feature Group D).

It has two modes:

1. Offline mode (default): exercises ``ChartSpecBuilder`` directly against a
   set of fixture metric payloads mimicking real tool outputs (station daily
   report, hourly report, extreme energy, ML model metrics, RAG chunks). It
   asserts that:

     * A DataTable is built for list-of-dicts metrics with correct columns.
     * A Chart is built with the appropriate chart_type (line for time series,
       bar for categorical comparison).
     * KPI cards are extracted for scalar numeric metrics.

2. Live mode (``--live``): posts a message to the real ``/solar-ai-chat/query``
   and ``/solar-ai-chat/stream`` endpoints, then asserts the response / done
   event contains a matching visualization payload.

Usage examples::

    # Offline fixtures (no backend required)
    python scripts/solar_chat_viz_cli.py

    # Live check against a running backend
    python scripts/solar_chat_viz_cli.py --live \\
        --message "Station daily report for Avonlie"

    # Dump JSON output to inspect structure
    python scripts/solar_chat_viz_cli.py --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure we can import the app when invoked directly from /main/backend/scripts
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.solar_ai_chat.chart_service import ChartSpecBuilder  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture payloads (shapes chosen to match real tool outputs)
# ---------------------------------------------------------------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "station_daily_report": {
        "topic": "station_report",
        "expect_table": True,
        "expect_chart": "bar",
        "expect_kpi_keys": ["total_energy_mwh", "station_count"],
        "metrics": {
            "report_date": "2026-04-10",
            "station_count": 4,
            "total_energy_mwh": 812.45,
            "has_data": True,
            "stations": [
                {"facility": "Avonlie", "energy_mwh": 221.5, "capacity_factor_pct": 18.4},
                {"facility": "Bomen", "energy_mwh": 198.3, "capacity_factor_pct": 15.7},
                {"facility": "Darlington Point", "energy_mwh": 205.8, "capacity_factor_pct": 16.9},
                {"facility": "Finley", "energy_mwh": 186.9, "capacity_factor_pct": 14.1},
            ],
        },
    },
    "station_hourly_report": {
        "topic": "station_report",
        "expect_table": True,
        "expect_chart": "line",
        "expect_kpi_keys": ["total_energy_mwh", "row_count"],
        "metrics": {
            "report_date": "2026-04-10",
            "row_count": 24,
            "total_energy_mwh": 112.3,
            "hourly_rows": [
                {"hour": h, "facility": "Avonlie",
                 "energy_mwh": round(max(0.0, 8 * ((1.0 - abs(12 - h) / 12)) + 0.5), 3),
                 "capacity_factor_pct": round(max(0.0, 20 - abs(12 - h)), 2)}
                for h in range(24)
            ],
        },
    },
    "ml_model_info": {
        "topic": "ml_model_info",
        "expect_table": False,
        "expect_chart": None,
        "expect_kpi_keys": ["r2", "rmse", "mae"],
        "metrics": {
            "champion_model": "GBT",
            "r2": 0.89,
            "rmse": 0.34,
            "mae": 0.21,
            "nrmse_pct": 7.1,
            "skill_score": 0.58,
        },
    },
    "rag_chunks": {
        "topic": "search_documents",
        "expect_table": True,
        "expect_chart": None,
        "expect_kpi_keys": ["total_results"],
        "metrics": {
            "total_results": 3,
            "chunks": [
                {"content": "Inverter maintenance guide...", "source_file": "manual_abb_100.pdf",
                 "doc_type": "equipment_manual", "score": 0.91},
                {"content": "Incident report 2025-11-03 ...", "source_file": "incident_20251103.pdf",
                 "doc_type": "incident_report", "score": 0.85},
                {"content": "Champion model changelog ...", "source_file": "changelog_v4.md",
                 "doc_type": "model_changelog", "score": 0.78},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Offline validation
# ---------------------------------------------------------------------------


def _check(label: str, cond: bool, message: str) -> bool:
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {label}: {message}")
    return cond


def validate_fixtures(dump_path: Path | None) -> bool:
    builder = ChartSpecBuilder()
    all_ok = True
    dump: dict[str, Any] = {}

    for name, fixture in FIXTURES.items():
        print(f"\n-> Fixture: {name}")
        table, chart, kpi = builder.build(fixture["metrics"], topic=fixture["topic"])

        dump[name] = {
            "data_table": table.model_dump() if table else None,
            "chart": chart.model_dump() if chart else None,
            "kpi_cards": kpi.model_dump() if kpi else None,
        }

        expect_table = fixture["expect_table"]
        ok = _check(
            "data_table present",
            (table is not None) == expect_table,
            f"expected={expect_table} got={table is not None}",
        )
        all_ok = all_ok and ok

        if table and expect_table:
            ok = _check(
                "data_table columns match keys",
                all(col.key in table.rows[0] for col in table.columns) if table.rows else True,
                f"columns={[c.key for c in table.columns]}",
            )
            all_ok = all_ok and ok
            ok = _check(
                "row_count matches rows length",
                table.row_count == len(table.rows),
                f"row_count={table.row_count} rows={len(table.rows)}",
            )
            all_ok = all_ok and ok

        expect_chart = fixture["expect_chart"]
        if expect_chart is None:
            ok = _check("chart absent", chart is None, f"got={chart.chart_type if chart else None}")
            all_ok = all_ok and ok
        else:
            ok = _check(
                f"chart_type == {expect_chart}",
                chart is not None and chart.chart_type == expect_chart,
                f"got={chart.chart_type if chart else None}",
            )
            all_ok = all_ok and ok
            if chart:
                spec = chart.plotly_spec
                has_data = isinstance(spec, dict) and isinstance(spec.get("data"), list) and len(spec["data"]) >= 1
                ok = _check("plotly_spec has data traces", has_data, f"spec keys={list(spec.keys())}")
                all_ok = all_ok and ok
                first_trace = spec["data"][0] if has_data else {}
                ok = _check(
                    "first trace has x and y arrays",
                    isinstance(first_trace.get("x"), list) and isinstance(first_trace.get("y"), list),
                    f"x_len={len(first_trace.get('x', []))} y_len={len(first_trace.get('y', []))}",
                )
                all_ok = all_ok and ok

        expected_kpis = set(fixture["expect_kpi_keys"])
        if expected_kpis:
            kpi_labels = {c.label for c in kpi.cards} if kpi else set()
            got_kpi_keys = set()
            for key in expected_kpis:
                # Match by label derived from key
                from app.services.solar_ai_chat.chart_service import _label_for as label_for  # type: ignore
                label, _, _ = label_for(key)
                if label in kpi_labels:
                    got_kpi_keys.add(key)
            ok = _check(
                "expected KPI keys surface",
                expected_kpis.issubset(got_kpi_keys),
                f"expected={sorted(expected_kpis)} got={sorted(got_kpi_keys)}",
            )
            all_ok = all_ok and ok

    if dump_path:
        dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        print(f"\nDumped fixture outputs to: {dump_path}")

    return all_ok


# ---------------------------------------------------------------------------
# Live mode (optional; hits the running backend)
# ---------------------------------------------------------------------------


def validate_live(
    base_url: str,
    username: str,
    password: str,
    role: str,
    message: str,
    *,
    session_title: str,
    use_stream: bool,
) -> bool:
    try:
        import httpx
    except ImportError:
        print("httpx not installed; cannot run live mode.")
        return False

    print(f"\n-> Live mode: POST {base_url}/solar-ai-chat/{'stream' if use_stream else 'query'}")
    with httpx.Client(base_url=base_url, timeout=180.0, follow_redirects=False) as client:
        resp = client.post("/auth/login", data={"username": username, "password": password, "next": "/dashboard"})
        if resp.status_code not in (302, 303):
            print(f"Login failed: {resp.status_code} {resp.text[:200]}")
            return False

        sess_resp = client.post("/solar-ai-chat/sessions", json={"role": role, "title": session_title})
        if sess_resp.status_code >= 400:
            print(f"Create session failed: {sess_resp.status_code} {sess_resp.text[:200]}")
            return False
        session_id = sess_resp.json().get("session_id", "")

        payload = {"role": role, "session_id": session_id, "message": message}

        if not use_stream:
            r = client.post("/solar-ai-chat/query", json=payload)
            if r.status_code >= 400:
                print(f"Query failed: {r.status_code} {r.text[:200]}")
                return False
            body = r.json()
            return _assert_live_payload(body)

        # Streaming
        with client.stream("POST", "/solar-ai-chat/stream", json=payload) as r:
            if r.status_code >= 400:
                print(f"Stream failed: {r.status_code}")
                return False
            buffer = ""
            done_evt: dict[str, Any] | None = None
            for chunk in r.iter_text():
                buffer += chunk
                blocks = buffer.split("\n\n")
                buffer = blocks.pop()
                for block in blocks:
                    line = block.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if evt.get("event") == "done":
                        done_evt = evt
            if done_evt is None:
                print("No done event received.")
                return False
            return _assert_live_payload(done_evt)


def _assert_live_payload(body: dict[str, Any]) -> bool:
    has_any = any(body.get(k) for k in ("data_table", "chart", "kpi_cards"))
    ok = _check("response has at least one viz field", has_any, f"keys={[k for k in ('data_table','chart','kpi_cards') if body.get(k)]}")
    if body.get("data_table"):
        dt = body["data_table"]
        ok &= _check("data_table has rows", bool(dt.get("rows")), f"row_count={dt.get('row_count')}")
    if body.get("chart"):
        ch = body["chart"]
        ok &= _check("chart has plotly_spec.data", isinstance(ch.get("plotly_spec", {}).get("data"), list), f"chart_type={ch.get('chart_type')}")
    return bool(ok)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Solar AI Chat visualization pipeline.")
    p.add_argument("--live", action="store_true", help="Run live validation against a running backend.")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--username", default="admin")
    p.add_argument("--password", default="admin123")
    p.add_argument("--role", default="data_engineer")
    p.add_argument("--message", default="Station daily report")
    p.add_argument("--session-title", default="CLI viz benchmark")
    p.add_argument("--stream", action="store_true", help="In live mode, use the SSE endpoint.")
    p.add_argument("--output-json", default="", help="Dump offline fixture outputs to this path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dump_path = Path(args.output_json) if args.output_json else None

    print("== Offline fixtures ==")
    ok = validate_fixtures(dump_path)

    if args.live:
        print("\n== Live mode ==")
        live_ok = validate_live(
            base_url=args.base_url,
            username=args.username,
            password=args.password,
            role=args.role,
            message=args.message,
            session_title=args.session_title,
            use_stream=args.stream,
        )
        ok = ok and live_ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
