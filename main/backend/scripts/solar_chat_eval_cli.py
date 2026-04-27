"""Solar Chat — LLM-judge regression eval harness.

Captures full chat responses (text + KPIs + chart spec + dataset + trace)
as structured JSONL so an evaluator LLM can compare v1 vs v2 behaviour
during the engine refactor.

Subcommands
-----------
- capture: run a question set against the running server, write JSONL
- inspect: single-query mode, pretty-print envelope to stdout
- judge:   pair baseline + candidate by query_id, ask judge LLM for verdict
- report:  aggregate JSONL judgments into a markdown report

Usage
-----
    python solar_chat_eval_cli.py capture --base-url http://127.0.0.1:8000 \
        --username admin --password admin123 \
        --question-set tests/eval/question_sets/regression_v1.yaml \
        --engine v1 --output reports/baseline_v1.jsonl

    python solar_chat_eval_cli.py inspect --base-url http://127.0.0.1:8000 \
        --username admin --password admin123 \
        --query "Trạm nào có tốc độ gió trung bình lớn nhất"

    python solar_chat_eval_cli.py judge \
        --baseline reports/baseline_v1.jsonl \
        --candidate reports/run_v2.jsonl \
        --judge-model gpt-4.1 \
        --output reports/judgment.jsonl

    python solar_chat_eval_cli.py report \
        --judgment reports/judgment.jsonl \
        --output reports/migration_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    import yaml
except ImportError:
    print("FATAL: PyYAML required. Install: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# Force UTF-8 stdout so Vietnamese diacritics render on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

def _login(client: httpx.Client, username: str, password: str) -> None:
    resp = client.post(
        "/auth/login",
        data={"username": username, "password": password, "next": "/dashboard"},
        follow_redirects=False,
    )
    if resp.status_code not in (302, 303):
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text[:300]}")
    if not client.cookies:
        raise RuntimeError("Login succeeded but no auth cookie set.")


def _create_session(client: httpx.Client, role: str, title: str) -> str:
    resp = client.post("/solar-ai-chat/sessions", json={"role": role, "title": title})
    if resp.status_code >= 400:
        raise RuntimeError(f"Create session failed ({resp.status_code}): {resp.text[:300]}")
    sid = str(resp.json().get("session_id", "")).strip()
    if not sid:
        raise RuntimeError("Create session response missing session_id.")
    return sid


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Capture: run question against server, return full envelope
# ---------------------------------------------------------------------------

def _extract_chart(body: dict[str, Any]) -> dict[str, Any] | None:
    chart = body.get("chart")
    if not isinstance(chart, dict):
        return None
    return {
        "chart_type": chart.get("chart_type"),
        "title": chart.get("title"),
        "spec": chart.get("plotly_spec") or chart.get("spec"),
        "source_metric_key": chart.get("source_metric_key"),
    }


def _extract_dataset(body: dict[str, Any]) -> dict[str, Any] | None:
    table = body.get("data_table")
    if not isinstance(table, dict):
        return None
    return {
        "title": table.get("title"),
        "columns": [c.get("key") for c in (table.get("columns") or []) if isinstance(c, dict)],
        "row_count": table.get("row_count", 0),
        "rows_preview": (table.get("rows") or [])[:20],   # cap to avoid huge JSONL
    }


def _extract_trace(body: dict[str, Any]) -> dict[str, Any] | None:
    trace = body.get("thinking_trace")
    if not isinstance(trace, dict):
        return None
    return {
        "summary": trace.get("summary"),
        "trace_id": trace.get("trace_id"),
        "steps": [
            {"step": s.get("step"), "detail": s.get("detail"), "status": s.get("status")}
            for s in (trace.get("steps") or [])
            if isinstance(s, dict)
        ],
    }


def _capture_one(
    client: httpx.Client,
    question: dict[str, Any],
    *,
    engine: str,
    session_id: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": question["query"],
        "role": question.get("role", "admin"),
    }
    if question.get("model_profile_id"):
        payload["model_profile_id"] = question["model_profile_id"]
    if question.get("model_name"):
        payload["model_name"] = question["model_name"]
    if session_id:
        payload["session_id"] = session_id

    started = time.time()
    error: str | None = None
    body: dict[str, Any] = {}
    try:
        resp = client.post(
            "/solar-ai-chat/query",
            json=payload,
            timeout=timeout_seconds,
        )
        if resp.status_code >= 400:
            error = f"HTTP {resp.status_code}: {resp.text[:500]}"
        else:
            body = resp.json()
    except Exception as exc:  # noqa: BLE001 - capture any failure for the report
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = int((time.time() - started) * 1000)

    return {
        "query_id": question["id"],
        "query": question["query"],
        "role": question.get("role", "admin"),
        "language": question.get("language"),
        "engine_version": engine,
        "expectations": question.get("expectations", {}),
        "model_profile_id": body.get("model_profile_id") or question.get("model_profile_id"),
        "model_name": body.get("model_used") or question.get("model_name"),
        "captured_at": _utc_iso_now(),
        "latency_ms": latency_ms,
        "response": {
            "answer": body.get("answer"),
            "topic": body.get("topic"),
            "intent_confidence": body.get("intent_confidence"),
            "fallback_used": body.get("fallback_used"),
            "warning_message": body.get("warning_message"),
        },
        "key_metrics": body.get("key_metrics") or {},
        "sources": body.get("sources") or [],
        "kpi_cards": body.get("kpi_cards"),
        "chart": _extract_chart(body),
        "dataset": _extract_dataset(body),
        "trace": _extract_trace(body),
        "errors": [error] if error else [],
    }


def cmd_capture(args: argparse.Namespace) -> int:
    qset_path = Path(args.question_set)
    if not qset_path.is_file():
        print(f"FATAL: question set not found: {qset_path}", file=sys.stderr)
        return 2
    questions = yaml.safe_load(qset_path.read_text(encoding="utf-8")) or []
    if not isinstance(questions, list):
        print("FATAL: question set must be a YAML list.", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(args.timeout_seconds, connect=10.0)
    base_url = args.base_url.rstrip("/")
    failures = 0

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        _login(client, args.username, args.password)
        session_id = None
        if args.shared_session:
            session_id = _create_session(client, args.role, "eval-cli-shared")
            print(f"  ↳ shared session_id={session_id}")

        with output_path.open("w", encoding="utf-8") as out_fh:
            for q in questions:
                if not isinstance(q, dict) or "id" not in q or "query" not in q:
                    print(f"  ! skip malformed question: {q!r}", file=sys.stderr)
                    continue
                # Per-question session unless shared
                qsid = session_id
                if not qsid:
                    qsid = _create_session(client, q.get("role", args.role), f"eval-{q['id']}")
                envelope = _capture_one(
                    client, q, engine=args.engine, session_id=qsid,
                    timeout_seconds=args.timeout_seconds,
                )
                out_fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                out_fh.flush()
                status = "✗" if envelope["errors"] else "✓"
                print(f"  {status} {q['id']:<40} {envelope['latency_ms']:>6}ms")
                if envelope["errors"]:
                    failures += 1
                    print(f"      → {envelope['errors'][0][:200]}")

    print(f"\nCaptured {len(questions)} queries → {output_path}")
    print(f"Failures: {failures}/{len(questions)}")
    return 0


# ---------------------------------------------------------------------------
# Inspect: single-query, pretty-printed
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(args.timeout_seconds, connect=10.0)
    base_url = args.base_url.rstrip("/")
    question = {
        "id": "inspect",
        "query": args.query,
        "role": args.role,
        "model_profile_id": args.model_profile_id or None,
        "model_name": args.model_name or None,
    }
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        _login(client, args.username, args.password)
        sid = _create_session(client, args.role, "inspect")
        envelope = _capture_one(
            client, question, engine="inspect", session_id=sid,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0 if not envelope["errors"] else 1


# ---------------------------------------------------------------------------
# Hard-expectation checks (run before LLM judge — auto-fail on violations)
# ---------------------------------------------------------------------------

def _check_expectations(envelope: dict[str, Any]) -> list[str]:
    """Return list of human-readable violation strings (empty = pass)."""
    exp = envelope.get("expectations") or {}
    violations: list[str] = []
    answer = (envelope.get("response") or {}).get("answer") or ""
    chart = envelope.get("chart") or {}
    dataset = envelope.get("dataset") or {}
    trace_steps = ((envelope.get("trace") or {}).get("steps") or [])
    # The primitive name lives in step["step"] (e.g. "1. execute_sql"); the
    # detail field only carries args JSON. Scanning detail for primitive
    # names was a bug that flagged every passing v2 run as failing.
    trace_text = " ".join(
        (s.get("step") or "") + " " + (s.get("detail") or "")
        for s in trace_steps
    )

    # must_call_primitive_or_tool
    expected_tools = exp.get("must_call_primitive_or_tool") or []
    if expected_tools:
        if not any(t.lower() in trace_text.lower() for t in expected_tools):
            violations.append(
                f"none of expected tools/primitives appear in trace: {expected_tools}"
            )

    # must_have_chart_type
    expected_charts = exp.get("must_have_chart_type") or []
    if expected_charts:
        actual_chart = (chart.get("chart_type") or "").lower()
        if actual_chart not in [c.lower() for c in expected_charts]:
            violations.append(
                f"chart_type={actual_chart!r} not in {expected_charts}"
            )

    # must_have_dataset_columns
    expected_cols = exp.get("must_have_dataset_columns") or []
    if expected_cols:
        actual_cols = [c.lower() for c in (dataset.get("columns") or [])]
        missing = [c for c in expected_cols if c.lower() not in actual_cols]
        if missing:
            violations.append(f"dataset missing columns: {missing}")

    # min_dataset_rows
    min_rows = exp.get("min_dataset_rows")
    if min_rows is not None and dataset.get("row_count", 0) < int(min_rows):
        violations.append(
            f"dataset row_count={dataset.get('row_count', 0)} < required {min_rows}"
        )

    # forbidden_in_answer
    for phrase in exp.get("forbidden_in_answer") or []:
        if phrase.lower() in answer.lower():
            violations.append(f"forbidden phrase in answer: {phrase!r}")

    # forbidden_error_codes
    err_text = " ".join(envelope.get("errors") or [])
    for code in exp.get("forbidden_error_codes") or []:
        if code in err_text:
            violations.append(f"forbidden error code present: {code}")

    # must_mention (any of these terms must appear in answer)
    must_mention = exp.get("must_mention") or []
    if must_mention:
        if not any(term.lower() in answer.lower() for term in must_mention):
            violations.append(
                f"answer mentions none of: {must_mention}"
            )

    return violations


# ---------------------------------------------------------------------------
# Judge: LLM-based comparison of baseline vs candidate
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are an evaluator for a Solar AI Chat system.
Given the same question answered by two engine versions (baseline = current,
candidate = refactored), decide which response is better and score per rubric.

Output STRICT JSON only — no markdown, no commentary.
Schema:
{
  "verdict": "better" | "worse" | "equivalent",
  "confidence": 0.0-1.0,
  "scores_baseline": {"correctness": 1-5, "completeness": 1-5, "chart_appropriateness": 1-5, "language_quality": 1-5, "data_grounding": 1-5},
  "scores_candidate": {"correctness": 1-5, "completeness": 1-5, "chart_appropriateness": 1-5, "language_quality": 1-5, "data_grounding": 1-5},
  "regression_flags": [string, ...],
  "improvement_flags": [string, ...],
  "rationale": "2-3 sentences"
}

Rubric definitions:
- correctness: does the answer match the data shown?
- completeness: did it answer all parts of the question?
- chart_appropriateness: does the chart type fit the data shape and intent?
- language_quality: native fluency in the question's language? no code-switching?
- data_grounding: are KPIs and citations consistent with the dataset?
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _judge_one(
    judge_client: httpx.Client,
    judge_model: str,
    judge_provider: str,
    judge_api_key: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Send one (baseline, candidate) pair to the judge LLM and parse the JSON verdict."""
    user_payload = {
        "question": baseline.get("query"),
        "language": baseline.get("language"),
        "expectations": baseline.get("expectations"),
        "baseline": _judge_view(baseline),
        "candidate": _judge_view(candidate),
    }
    body = json.dumps(user_payload, ensure_ascii=False)

    if judge_provider == "openai":
        resp = judge_client.post(
            "/chat/completions",
            json={
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": body},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {judge_api_key}"},
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    elif judge_provider == "gemini":
        resp = judge_client.post(
            f"/models/{judge_model}:generateContent",
            json={
                "systemInstruction": {"parts": [{"text": _JUDGE_SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": body}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            headers={"x-goog-api-key": judge_api_key},
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"Unknown judge provider: {judge_provider}")

    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        verdict = {"verdict": "equivalent", "confidence": 0.0,
                   "rationale": f"judge returned non-JSON: {text[:200]}",
                   "regression_flags": [], "improvement_flags": [],
                   "scores_baseline": {}, "scores_candidate": {}}
    return verdict


def _judge_view(envelope: dict[str, Any]) -> dict[str, Any]:
    """Compress envelope into the slice the judge actually needs."""
    return {
        "answer": (envelope.get("response") or {}).get("answer"),
        "topic": (envelope.get("response") or {}).get("topic"),
        "tools_used": [s.get("step") for s in ((envelope.get("trace") or {}).get("steps") or [])],
        "chart_type": (envelope.get("chart") or {}).get("chart_type"),
        "dataset_columns": (envelope.get("dataset") or {}).get("columns"),
        "dataset_row_count": (envelope.get("dataset") or {}).get("row_count"),
        "key_metrics_keys": list((envelope.get("key_metrics") or {}).keys()),
        "errors": envelope.get("errors"),
    }


def cmd_judge(args: argparse.Namespace) -> int:
    baseline = {r["query_id"]: r for r in _read_jsonl(Path(args.baseline))}
    candidate = {r["query_id"]: r for r in _read_jsonl(Path(args.candidate))}

    judge_provider = args.judge_provider
    judge_api_key = os.environ.get(args.judge_api_key_env, "").strip()
    if not judge_api_key:
        print(f"FATAL: env var {args.judge_api_key_env} is empty.", file=sys.stderr)
        return 2

    judge_base_url = args.judge_base_url.rstrip("/")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(60.0, connect=10.0)
    judged = 0
    skipped = 0
    with httpx.Client(base_url=judge_base_url, timeout=timeout) as judge_client, \
         output_path.open("w", encoding="utf-8") as out_fh:
        for qid, base in baseline.items():
            if qid not in candidate:
                skipped += 1
                continue
            cand = candidate[qid]

            # Hard-expectation pre-check (cheaper than calling the judge)
            base_violations = _check_expectations(base)
            cand_violations = _check_expectations(cand)
            try:
                verdict = _judge_one(
                    judge_client, args.judge_model, judge_provider, judge_api_key,
                    base, cand,
                )
            except Exception as exc:  # noqa: BLE001
                verdict = {"verdict": "equivalent", "confidence": 0.0,
                           "rationale": f"judge error: {exc}",
                           "regression_flags": [], "improvement_flags": [],
                           "scores_baseline": {}, "scores_candidate": {}}

            row = {
                "query_id": qid,
                "query": base.get("query"),
                "baseline_violations": base_violations,
                "candidate_violations": cand_violations,
                "verdict": verdict.get("verdict"),
                "confidence": verdict.get("confidence"),
                "scores_baseline": verdict.get("scores_baseline"),
                "scores_candidate": verdict.get("scores_candidate"),
                "regression_flags": verdict.get("regression_flags") or [],
                "improvement_flags": verdict.get("improvement_flags") or [],
                "rationale": verdict.get("rationale"),
                "baseline_latency_ms": base.get("latency_ms"),
                "candidate_latency_ms": cand.get("latency_ms"),
            }
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            judged += 1
            print(f"  {qid:<40} verdict={row['verdict']!s:<12} "
                  f"viol_b={len(base_violations)} viol_c={len(cand_violations)}")

    print(f"\nJudged {judged} pairs (skipped {skipped} missing in candidate) → {output_path}")
    return 0


# ---------------------------------------------------------------------------
# Report: aggregate judgments → markdown
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.judgment))
    if not rows:
        print("FATAL: judgment file is empty.", file=sys.stderr)
        return 2

    total = len(rows)
    verdicts = Counter(r.get("verdict") or "unknown" for r in rows)
    regressions = [r for r in rows if r.get("verdict") == "worse"]
    improvements = [r for r in rows if r.get("verdict") == "better"]

    def _avg_score(rubric: str, side: str) -> float:
        vals = [
            (r.get(side) or {}).get(rubric, 0)
            for r in rows
            if (r.get(side) or {}).get(rubric)
        ]
        return statistics.mean(vals) if vals else 0.0

    rubrics = ["correctness", "completeness", "chart_appropriateness",
               "language_quality", "data_grounding"]

    base_lat = [r.get("baseline_latency_ms") for r in rows if r.get("baseline_latency_ms")]
    cand_lat = [r.get("candidate_latency_ms") for r in rows if r.get("candidate_latency_ms")]

    md_lines = [
        f"# v1 → v2 Migration Report",
        f"",
        f"**Generated:** {_utc_iso_now()}",
        f"**Question count:** {total}",
        f"",
        f"## Verdict distribution",
        f"",
        f"| Verdict | Count | % |",
        f"|---|---|---|",
    ]
    for v in ("better", "equivalent", "worse", "unknown"):
        n = verdicts.get(v, 0)
        md_lines.append(f"| {v} | {n} | {100*n/total:.0f}% |")

    md_lines.append("")
    md_lines.append("## Per-rubric averages")
    md_lines.append("")
    md_lines.append("| Rubric | Baseline | Candidate | Δ |")
    md_lines.append("|---|---|---|---|")
    for r in rubrics:
        b = _avg_score(r, "scores_baseline")
        c = _avg_score(r, "scores_candidate")
        md_lines.append(f"| {r} | {b:.2f} | {c:.2f} | {c-b:+.2f} |")

    md_lines.append("")
    md_lines.append("## Latency")
    md_lines.append("")
    md_lines.append("| | Baseline | Candidate |")
    md_lines.append("|---|---|---|")
    md_lines.append(f"| p50 (ms) | {int(statistics.median(base_lat)) if base_lat else 0} "
                    f"| {int(statistics.median(cand_lat)) if cand_lat else 0} |")
    md_lines.append(f"| mean (ms) | {int(statistics.mean(base_lat)) if base_lat else 0} "
                    f"| {int(statistics.mean(cand_lat)) if cand_lat else 0} |")

    if regressions:
        md_lines.append("")
        md_lines.append(f"## Regressions ({len(regressions)} — must fix before cutover)")
        md_lines.append("")
        for r in regressions:
            md_lines.append(f"### {r['query_id']}")
            md_lines.append(f"- Query: {r.get('query')}")
            md_lines.append(f"- Confidence: {r.get('confidence', 0):.2f}")
            md_lines.append(f"- Flags: {r.get('regression_flags')}")
            md_lines.append(f"- Rationale: {r.get('rationale')}")
            md_lines.append("")

    if improvements:
        md_lines.append(f"## Improvements ({len(improvements)})")
        md_lines.append("")
        flag_counter = Counter()
        for r in improvements:
            for f in r.get("improvement_flags") or []:
                flag_counter[f] += 1
        for flag, n in flag_counter.most_common(10):
            md_lines.append(f"- {flag} ({n}×)")

    md_lines.append("")
    md_lines.append("## Hard-expectation violations")
    md_lines.append("")
    base_viols = sum(1 for r in rows if r.get("baseline_violations"))
    cand_viols = sum(1 for r in rows if r.get("candidate_violations"))
    md_lines.append(f"- Baseline failed expectations: {base_viols}/{total}")
    md_lines.append(f"- Candidate failed expectations: {cand_viols}/{total}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote report → {output_path}")

    # CI gate flags
    if args.fail_on_regressions is not None and len(regressions) > args.fail_on_regressions:
        print(f"FAIL: {len(regressions)} regressions exceed threshold {args.fail_on_regressions}",
              file=sys.stderr)
        return 1
    if args.fail_on_rubric_drop is not None:
        for r in rubrics:
            drop = _avg_score(r, "scores_baseline") - _avg_score(r, "scores_candidate")
            if drop > args.fail_on_rubric_drop:
                print(f"FAIL: rubric {r!r} dropped {drop:.2f} (threshold {args.fail_on_rubric_drop})",
                      file=sys.stderr)
                return 1
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # capture
    p_cap = sub.add_parser("capture", help="Run question set, write JSONL of full envelopes.")
    p_cap.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_cap.add_argument("--username", default="admin")
    p_cap.add_argument("--password", default="admin123")
    p_cap.add_argument("--question-set", required=True, help="YAML file with question list.")
    p_cap.add_argument("--engine", default="auto", help="Tag for engine_version field (e.g. v1, v2).")
    p_cap.add_argument("--role", default="admin", help="Default role when question doesn't specify.")
    p_cap.add_argument("--output", required=True, help="JSONL output path.")
    p_cap.add_argument("--shared-session", action="store_true",
                       help="Reuse one chat session across all questions (default: per-question session).")
    p_cap.add_argument("--timeout-seconds", type=float, default=180.0)
    p_cap.set_defaults(func=cmd_capture)

    # inspect
    p_ins = sub.add_parser("inspect", help="Single-query mode, pretty-print envelope.")
    p_ins.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_ins.add_argument("--username", default="admin")
    p_ins.add_argument("--password", default="admin123")
    p_ins.add_argument("--query", required=True)
    p_ins.add_argument("--role", default="admin")
    p_ins.add_argument("--model-profile-id", default="")
    p_ins.add_argument("--model-name", default="")
    p_ins.add_argument("--timeout-seconds", type=float, default=180.0)
    p_ins.set_defaults(func=cmd_inspect)

    # judge
    p_jdg = sub.add_parser("judge", help="LLM-judge baseline vs candidate JSONL pairs.")
    p_jdg.add_argument("--baseline", required=True)
    p_jdg.add_argument("--candidate", required=True)
    p_jdg.add_argument("--judge-provider", choices=["openai", "gemini"], default="openai")
    p_jdg.add_argument("--judge-base-url", default="https://api.openai.com/v1")
    p_jdg.add_argument("--judge-model", default="gpt-4.1")
    p_jdg.add_argument("--judge-api-key-env", default="SOLAR_CHAT_EVAL_JUDGE_KEY",
                       help="Env var holding the judge API key.")
    p_jdg.add_argument("--output", required=True)
    p_jdg.set_defaults(func=cmd_judge)

    # report
    p_rep = sub.add_parser("report", help="Aggregate judgment JSONL to markdown.")
    p_rep.add_argument("--judgment", required=True)
    p_rep.add_argument("--output", required=True)
    p_rep.add_argument("--fail-on-regressions", type=int, default=None,
                       help="Exit 1 if regression count exceeds N.")
    p_rep.add_argument("--fail-on-rubric-drop", type=float, default=None,
                       help="Exit 1 if any rubric mean drops by more than this.")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
