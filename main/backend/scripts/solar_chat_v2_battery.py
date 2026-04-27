"""Solar Chat v2 — ground-truth battery runner.

Runs each YAML case end-to-end against a real LLM + Databricks dispatcher,
captures the same surface the UI receives (answer, key_metrics, chart,
data_table, sources, trace), and asserts:

  • Numeric KPIs match Databricks ground truth (tolerance %)
  • Chart format / mark match expectation
  • Data table row count meets minimum
  • Source attribution matches expected table
  • Answer text avoids known refusal phrases
  • Greeting cases skip tool calls

Usage:
    python scripts/solar_chat_v2_battery.py [--cases tests/eval/v2_battery.yaml]
                                            [--profile <profile_id>]
                                            [--filter <id_substring>]
                                            [--out outputs/v2_battery_<ts>.json]

Exits 0 if every case passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- bootstrap path so we can `from app...` from inside `main/backend` ----
HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Load .env so DATABRICKS_* + SOLAR_CHAT_* are populated
try:
    from dotenv import load_dotenv
    for candidate in (BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env",
                      BACKEND_ROOT.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
except Exception:  # noqa: BLE001
    pass

import yaml  # noqa: E402

from app.core.settings import SolarChatSettings  # noqa: E402
from app.services.solar_ai_chat.llm_client import LLMModelRouter  # noqa: E402
from app.services.solar_ai_chat.model_profile_service import (  # noqa: E402
    get_default_profile_id,
    resolve_profile,
    settings_with_profile_override,
)
from app.services.solar_ai_chat.dispatcher import Dispatcher  # noqa: E402
from app.services.solar_ai_chat.engine import ChatEngine  # noqa: E402


GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _clr(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def _format_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, str) and len(v) > 80:
        return v[:77] + "..."
    return str(v)


def _within_tolerance(actual: float, expected: float, tolerance_pct: float) -> bool:
    if expected == 0:
        return abs(actual) <= max(tolerance_pct / 100.0, 0.001)
    return abs(actual - expected) / abs(expected) <= tolerance_pct / 100.0


def _execute_ground_truth(dispatcher: Dispatcher, sql: str) -> dict[str, Any]:
    """Run a ground-truth SQL via the same dispatcher used by the engine."""
    result = dispatcher.execute("execute_sql", {"sql": sql})
    if not result.ok:
        raise RuntimeError(f"Ground-truth SQL failed: {result.result.get('error') or result.result}")
    rows = result.result.get("rows") or []
    if not rows:
        return {}
    # Single-row aggregate: return the row dict.
    if len(rows) == 1:
        return rows[0]
    # Multi-row: return the row dict + extras
    return {"_rows": rows, "_row_count": len(rows), **rows[0]}


def _check_assertions(
    case: dict[str, Any],
    result_dict: dict[str, Any],
    ground_truth: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failure_reasons)."""
    failures: list[str] = []

    answer = (result_dict.get("answer") or "").lower()
    must = case.get("answer_must_contain") or []
    for s in must:
        if s.lower() not in answer:
            failures.append(f"answer missing required substring: {s!r}")
    must_not = case.get("answer_must_not_contain") or []
    for s in must_not:
        if s.lower() in answer:
            failures.append(f"answer contains forbidden substring: {s!r}")

    # KPI checks against ground truth
    expect_kpi = case.get("expect_kpi") or {}
    if expect_kpi:
        kpis = result_dict.get("key_metrics") or {}
        tol = float(case.get("tolerance_pct") or 1.0)
        for kpi_key, gt_key in expect_kpi.items():
            if kpi_key not in kpis:
                failures.append(f"missing KPI key: {kpi_key} (have: {sorted(kpis.keys())})")
                continue
            actual = kpis[kpi_key]
            if isinstance(gt_key, str) and ground_truth and gt_key in ground_truth:
                expected = ground_truth[gt_key]
                if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                    if not _within_tolerance(float(actual), float(expected), tol):
                        failures.append(
                            f"KPI {kpi_key}={actual} differs from ground-truth {gt_key}={expected} "
                            f"(>{tol}% tolerance)"
                        )

    # Chart shape
    expect_chart = case.get("expect_chart")
    if expect_chart:
        chart = result_dict.get("chart") or {}
        for k, expected in expect_chart.items():
            actual = chart.get(k)
            if actual != expected:
                failures.append(f"chart.{k}={actual!r} expected {expected!r}")

    # Data table presence + min rows
    if case.get("expect_data_table") is True:
        dt = result_dict.get("data_table")
        if not dt:
            failures.append("expected data_table but got none")
        else:
            min_rows = int(case.get("expect_min_rows") or 1)
            actual_rows = int(dt.get("row_count") or 0)
            if actual_rows < min_rows:
                failures.append(f"data_table has {actual_rows} rows (< {min_rows} expected)")
    elif case.get("expect_data_table") is False:
        if result_dict.get("data_table"):
            failures.append("data_table present but not expected")

    # Source attribution
    expect_source = case.get("expect_source_table")
    if expect_source:
        srcs = [s.get("dataset") for s in (result_dict.get("sources") or [])]
        if expect_source not in srcs:
            failures.append(f"missing source: {expect_source} (got: {srcs})")

    # No tool calls (greeting case)
    if case.get("expect_no_tool_calls"):
        steps = result_dict.get("trace_steps") or []
        if steps:
            failures.append(f"expected no tool calls, got {len(steps)}")

    return (len(failures) == 0, failures)


def _result_to_dict(r) -> dict[str, Any]:
    """Convert ChatEngineResult dataclass to plain dict for serialisation/asserts."""
    return {
        "answer": r.answer,
        "model_used": r.model_used,
        "fallback_used": r.fallback_used,
        "key_metrics": dict(r.key_metrics or {}),
        "sources": [
            {"dataset": getattr(s, "dataset", None),
             "layer": getattr(s, "layer", None)}
            for s in (r.sources or [])
        ],
        "chart": r.chart,
        "data_table": r.data_table,
        "trace_steps": list(r.trace_steps or []),
        "error": r.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(BACKEND_ROOT / "tests/eval/v2_battery.yaml"))
    parser.add_argument("--filter", default="", help="Substring filter on case id")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--language-default", default="en")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="0 = run all matching")
    parser.add_argument("--profile", default=None,
                        help="Profile ID to use (e.g. openai-local). Overrides the DEFAULT profile.")
    parser.add_argument("--primary-model", default=None,
                        help="Override the resolved primary model for this run")
    parser.add_argument("--fallback-model", default=None,
                        help="Override the resolved fallback model for this run")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(_clr(f"FATAL: cases file not found: {cases_path}", RED))
        return 2

    raw = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    if args.filter:
        cases = [c for c in cases if args.filter in c.get("id", "")]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        print(_clr("No cases matched.", YELLOW))
        return 1

    base_settings = SolarChatSettings()
    # Mirror routes.get_solar_ai_chat_service: prefer the profile flagged
    # _DEFAULT=true when present, else fall back to legacy SOLAR_CHAT_LLM_*.
    chosen_profile_id = args.profile or get_default_profile_id()
    chosen_profile = (
        resolve_profile(chosen_profile_id, "admin")
        if chosen_profile_id else None
    )
    if chosen_profile is not None:
        settings = settings_with_profile_override(
            base_settings, chosen_profile, model_name=None,
        )
    else:
        settings = base_settings

    if not settings.llm_api_key and not settings.llm_base_url:
        print(_clr("FATAL: no LLM connection configured. Either set a profile "
                   "with _DEFAULT=true or populate SOLAR_CHAT_LLM_* in .env.",
                   RED))
        return 2

    # CLI overrides for one-off model evaluation runs.
    if args.primary_model or args.fallback_model:
        update: dict[str, Any] = {}
        if args.primary_model:
            update["primary_model"] = args.primary_model
        if args.fallback_model:
            update["fallback_model"] = args.fallback_model
        settings = settings.model_copy(update=update)

    router = LLMModelRouter(settings=settings)
    dispatcher = Dispatcher(settings, role_id="admin")
    engine = ChatEngine(router, dispatcher)
    print(_clr(f"\n=== Solar Chat v2 battery — {len(cases)} cases ===", DIM))
    print(_clr(f"Engine version: {settings.engine_version} | "
               f"Primary model: {settings.primary_model} | "
               f"Fallback: {settings.fallback_model}", DIM))

    runs: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    started_all = time.perf_counter()

    for i, case in enumerate(cases, 1):
        cid = case.get("id", f"case_{i}")
        question = case.get("question", "")
        tool_hints = case.get("tool_hints") or []
        force_chart = "visualize" in tool_hints
        difficulty = case.get("difficulty", "medium")

        print(f"\n[{i:>2}/{len(cases)}] {_clr(cid, YELLOW)} ({difficulty})")
        print(f"     Q: {question}")

        # 1. Ground truth (if any)
        ground_truth: dict[str, Any] | None = None
        gt_sql = case.get("ground_truth_sql")
        if gt_sql:
            try:
                ground_truth = _execute_ground_truth(dispatcher, gt_sql)
                if ground_truth:
                    preview = {k: _format_value(v) for k, v in list(ground_truth.items())[:5]
                               if not k.startswith("_")}
                    print(f"     {_clr('ground truth:', DIM)} {preview}")
            except Exception as gt_err:
                print(f"     {_clr(f'ground truth FAILED: {gt_err}', RED)}")

        # 2. Run the engine
        t0 = time.perf_counter()
        try:
            from app.services.solar_ai_chat.engine import _detect_language
            lang = _detect_language(question) or args.language_default
            r = engine.run(
                user_message=question,
                history=[],
                language=lang,
                force_chart=force_chart,
            )
            duration = int((time.perf_counter() - t0) * 1000)
            result_dict = _result_to_dict(r)
        except Exception as run_err:  # noqa: BLE001
            duration = int((time.perf_counter() - t0) * 1000)
            print(f"     {_clr(f'ENGINE CRASH: {run_err}', RED)}  ({duration}ms)")
            n_fail += 1
            runs.append({
                "id": cid, "question": question, "passed": False,
                "duration_ms": duration, "error": str(run_err),
                "ground_truth": ground_truth,
            })
            continue

        # 3. Assertions
        passed, failures = _check_assertions(case, result_dict, ground_truth)

        steps = result_dict.get("trace_steps") or []
        primitives_used = [s.get("primitive") for s in steps]
        kpi_summary = {k: _format_value(v) for k, v in (result_dict.get("key_metrics") or {}).items()}

        status = _clr("PASS", GREEN) if passed else _clr("FAIL", RED)
        print(f"     {status}  {duration}ms  steps={len(steps)} "
              f"primitives={primitives_used}")
        print(f"     answer (first 200 chars): "
              f"{(result_dict['answer'] or '')[:200].strip()!r}")
        if kpi_summary:
            print(f"     KPI: {kpi_summary}")
        chart = result_dict.get("chart")
        if chart:
            print(f"     chart: format={chart.get('format')} mark={chart.get('mark')} "
                  f"rows={chart.get('row_count')}")
        dt = result_dict.get("data_table")
        if dt:
            print(f"     data_table: rows={dt.get('row_count')} "
                  f"cols={[c.get('key') for c in (dt.get('columns') or [])][:6]}")
        if failures:
            for f in failures:
                print(f"       {_clr('✗', RED)} {f}")

        if passed:
            n_pass += 1
        else:
            n_fail += 1
        runs.append({
            "id": cid, "question": question, "passed": passed,
            "duration_ms": duration, "primitives_used": primitives_used,
            "failures": failures, "ground_truth": ground_truth,
            "result": result_dict,
        })

    # Summary
    total_dur = int((time.perf_counter() - started_all) * 1000)
    print()
    print(_clr("=" * 60, DIM))
    summary_color = GREEN if n_fail == 0 else RED
    print(_clr(f"  TOTAL: {n_pass}/{n_pass + n_fail} passed  ({total_dur:,}ms)", summary_color))
    if n_fail:
        print(_clr(f"  FAILED CASES:", RED))
        for r in runs:
            if not r["passed"]:
                print(_clr(f"    • {r['id']}: {r.get('failures') or [r.get('error')]}", RED))
    print(_clr("=" * 60, DIM))

    out_path = Path(args.out) if args.out else (
        BACKEND_ROOT.parent / "outputs" /
        f"v2_battery_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": settings.primary_model,
        "engine_version": settings.engine_version,
        "n_pass": n_pass, "n_fail": n_fail,
        "total_duration_ms": total_dur,
        "runs": runs,
    }, indent=2, default=str), encoding="utf-8")
    print(_clr(f"\nDetailed report: {out_path}", DIM))

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
