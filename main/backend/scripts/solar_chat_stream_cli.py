"""CLI tool to test and validate the Solar AI Chat SSE streaming endpoint.

Usage examples:
    # Basic stream test
    python scripts/solar_chat_stream_cli.py --message "Energy performance"

    # Save SSE events as JSON, verify task steps appear
    python scripts/solar_chat_stream_cli.py --message "System overview" --output-json out.json

    # Fail if no thinking_step events emitted
    python scripts/solar_chat_stream_cli.py --message "Pipeline status" --expect-steps

    # Compare latency of stream vs classic /query
    python scripts/solar_chat_stream_cli.py --message "Facility info" --compare-latency

    # Repeat
    python scripts/solar_chat_stream_cli.py --message "Forecast" --repeat 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Solar AI Chat SSE streaming endpoint (/solar-ai-chat/stream).",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Backend base URL.")
    parser.add_argument("--username", default="admin", help="Login username.")
    parser.add_argument("--password", default="admin123", help="Login password.")
    parser.add_argument("--role", default="data_engineer", help="Chat role for benchmark.")
    parser.add_argument("--message", required=True, help="Message to send.")
    parser.add_argument("--session-id", default="", help="Existing session id (auto-created if empty).")
    parser.add_argument("--session-title", default="CLI stream benchmark", help="Title for auto-created session.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of stream requests.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Overall request timeout.")
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0, help="Connection timeout.")
    parser.add_argument(
        "--expect-steps",
        action="store_true",
        help="Fail with exit code 1 if the stream emits zero thinking_step events.",
    )
    parser.add_argument(
        "--compare-latency",
        action="store_true",
        help="After streaming, also call /solar-ai-chat/query and compare latency.",
    )
    parser.add_argument("--print-answer", action="store_true", help="Print final answer to stdout.")
    parser.add_argument("--print-events", action="store_true", help="Print every SSE event to stdout.")
    parser.add_argument("--output-json", default="", help="Write results as JSON to this path.")
    return parser.parse_args()


# -----------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------

def _build_timeout(total: float, connect: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=total, connect=connect)


def login(client: httpx.Client, username: str, password: str) -> None:
    resp = client.post(
        "/auth/login",
        data={"username": username, "password": password, "next": "/dashboard"},
        follow_redirects=False,
    )
    if resp.status_code not in (302, 303):
        raise RuntimeError(f"Login failed with status {resp.status_code}: {resp.text[:300]}")
    if not client.cookies:
        raise RuntimeError("Login succeeded but no auth cookie was set.")


def create_session(client: httpx.Client, role: str, title: str) -> str:
    resp = client.post("/solar-ai-chat/sessions", json={"role": role, "title": title})
    if resp.status_code >= 400:
        raise RuntimeError(f"Create session failed: {resp.status_code} {resp.text[:300]}")
    payload = resp.json()
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise RuntimeError("Create session response missing session_id.")
    return session_id


# -----------------------------------------------------------------------
# SSE stream runner
# -----------------------------------------------------------------------

def stream_once(
    client: httpx.Client,
    role: str,
    session_id: str,
    message: str,
    print_events: bool = False,
) -> dict[str, Any]:
    """POST to /stream, parse SSE events, return result summary."""
    payload = {"role": role, "session_id": session_id, "message": message}

    events: list[dict[str, Any]] = []
    step_events: list[dict[str, Any]] = []
    tool_result_events: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []
    done_event: dict[str, Any] | None = None
    error_event: dict[str, Any] | None = None

    started = time.perf_counter()
    first_step_ts: float | None = None
    done_ts: float | None = None

    with client.stream("POST", "/solar-ai-chat/stream", json=payload) as resp:
        if resp.status_code >= 400:
            body = resp.read()
            raise RuntimeError(f"Stream request failed: {resp.status_code} {body[:300]!r}")

        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
            blocks = buffer.split("\n\n")
            buffer = blocks.pop()  # keep last incomplete block
            for block in blocks:
                line = block.strip()
                if not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                evt_type = evt.get("event", "")
                events.append(evt)
                now = time.perf_counter()

                if print_events:
                    print(f"  [{evt_type:15s}] {json.dumps({k: v for k, v in evt.items() if k != 'event'})}")

                if evt_type == "thinking_step":
                    step_events.append(evt)
                    if first_step_ts is None:
                        first_step_ts = now
                elif evt_type == "tool_result":
                    tool_result_events.append(evt)
                elif evt_type == "status_update":
                    status_events.append(evt)
                elif evt_type == "done":
                    done_event = evt
                    done_ts = now
                elif evt_type == "error":
                    error_event = evt

    elapsed = time.perf_counter() - started
    ttfe_ms = int((first_step_ts - started) * 1000) if first_step_ts else -1
    total_ms = int(elapsed * 1000)

    return {
        "mode": "stream",
        "total_ms": total_ms,
        "time_to_first_event_ms": ttfe_ms,
        "server_latency_ms": int(done_event.get("latency_ms", -1)) if done_event else -1,
        "event_count": len(events),
        "thinking_step_count": len(step_events),
        "tool_result_count": len(tool_result_events),
        "status_count": len(status_events),
        "answer": done_event.get("answer", "") if done_event else "",
        "topic": done_event.get("topic", "") if done_event else "",
        "model_used": done_event.get("model_used", "") if done_event else "",
        "fallback_used": done_event.get("fallback_used", False) if done_event else False,
        "warning_message": done_event.get("warning_message", "") if done_event else "",
        "intent_confidence": done_event.get("intent_confidence", 0.0) if done_event else 0.0,
        "has_done": done_event is not None,
        "has_error": error_event is not None,
        "error_message": error_event.get("message", "") if error_event else "",
        "all_events": [e.get("event") for e in events],
        "step_tools": [e.get("tool_name") for e in step_events],
    }


# -----------------------------------------------------------------------
# Classic /query runner (for latency comparison)
# -----------------------------------------------------------------------

def query_once(
    client: httpx.Client,
    role: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    payload = {"role": role, "session_id": session_id, "message": message}
    started = time.perf_counter()
    resp = client.post("/solar-ai-chat/query", json=payload)
    total_ms = int((time.perf_counter() - started) * 1000)
    if resp.status_code >= 400:
        raise RuntimeError(f"Query failed: {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    return {
        "mode": "query",
        "total_ms": total_ms,
        "latency_ms": int(data.get("latency_ms", -1)),
        "answer": str(data.get("answer", "")),
        "model_used": str(data.get("model_used", "")),
    }


# -----------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------

def _mean(values: list[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    import math
    idx = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[idx]


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    repeat = max(1, int(args.repeat))
    timeout = _build_timeout(args.timeout_seconds, args.connect_timeout_seconds)
    base_url = args.base_url.rstrip("/")

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        login(client=client, username=args.username, password=args.password)
        session_id = args.session_id.strip()
        if not session_id:
            session_id = create_session(client=client, role=args.role, title=args.session_title)

        print(f"base_url={base_url}")
        print(f"session_id={session_id}")
        print(f"repeat={repeat}")

        stream_results: list[dict[str, Any]] = []
        query_results: list[dict[str, Any]] = []
        step_violations = 0

        for run in range(1, repeat + 1):
            print(f"\n── Run {run}/{repeat} ──────────────────────────────────")

            # ---------- stream ----------
            print(f"[stream] POST /solar-ai-chat/stream  message={args.message!r}")
            if args.print_events:
                print("  Events:")
            try:
                sr = stream_once(
                    client=client,
                    role=args.role,
                    session_id=session_id,
                    message=args.message,
                    print_events=args.print_events,
                )
            except Exception as exc:
                print(f"  STREAM ERROR: {exc}")
                step_violations += 1
                continue

            stream_results.append(sr)
            print(
                f"  total_ms={sr['total_ms']}   ttfe_ms={sr['time_to_first_event_ms']}   "
                f"server_latency_ms={sr['server_latency_ms']}"
            )
            print(
                f"  events={sr['event_count']}  thinking_steps={sr['thinking_step_count']}  "
                f"tool_results={sr['tool_result_count']}  model={sr['model_used']}  "
                f"fallback={sr['fallback_used']}"
            )
            print(f"  topic={sr['topic']}  intent_confidence={sr['intent_confidence']:.2f}")
            print(f"  event_types={sr['all_events']}")
            if sr["step_tools"]:
                print(f"  tool_sequence={sr['step_tools']}")
            if sr["has_error"]:
                print(f"  stream_error={sr['error_message']}")
            if sr["warning_message"]:
                print(f"  warning={sr['warning_message']}")

            if args.expect_steps and sr["thinking_step_count"] <= 0:
                step_violations += 1
                print("  step_check=FAILED expected thinking_step events > 0")

            if args.print_answer and sr["answer"]:
                print(f"  answer:\n{sr['answer'][:1000]}")

            # ---------- compare latency (optional) ----------
            if args.compare_latency:
                try:
                    qr = query_once(
                        client=client,
                        role=args.role,
                        session_id=session_id,
                        message=args.message,
                    )
                    query_results.append(qr)
                    print(
                        f"  [query]  total_ms={qr['total_ms']}  server_latency_ms={qr['latency_ms']}  "
                        f"model={qr['model_used']}"
                    )
                    diff = sr["total_ms"] - qr["total_ms"]
                    sign = "+" if diff >= 0 else ""
                    print(f"  latency_delta(stream-query)={sign}{diff}ms")
                except Exception as exc:
                    print(f"  QUERY ERROR: {exc}")

        # ── Summary ──────────────────────────────────────────
        if stream_results:
            print("\n── Summary ─────────────────────────────────────────────")
            total_vals = [r["total_ms"] for r in stream_results]
            ttfe_vals = [r["time_to_first_event_ms"] for r in stream_results if r["time_to_first_event_ms"] >= 0]
            server_vals = [r["server_latency_ms"] for r in stream_results if r["server_latency_ms"] >= 0]
            step_vals = [r["thinking_step_count"] for r in stream_results]

            print(f"  stream_total_avg_ms={_mean(total_vals)}   p95={_p95(total_vals)}")
            if ttfe_vals:
                print(f"  stream_ttfe_avg_ms={_mean(ttfe_vals)}   p95={_p95(ttfe_vals)}")
            if server_vals:
                print(f"  server_latency_avg_ms={_mean(server_vals)}   p95={_p95(server_vals)}")
            print(f"  thinking_step_avg={_mean(step_vals)}")

            has_done = sum(1 for r in stream_results if r["has_done"])
            has_error = sum(1 for r in stream_results if r["has_error"])
            print(f"  has_done={has_done}/{len(stream_results)}   has_error={has_error}/{len(stream_results)}")
            if args.expect_steps:
                print(f"  step_violations={step_violations}")

            if args.compare_latency and query_results:
                q_total = [r["total_ms"] for r in query_results]
                print(f"  query_total_avg_ms={_mean(q_total)}   p95={_p95(q_total)}")

        if args.output_json.strip():
            out_path = Path(args.output_json).expanduser()
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_payload = {
                "base_url": base_url,
                "message": args.message,
                "repeat": repeat,
                "stream_results": stream_results,
                "query_results": query_results,
                "summary": {
                    "stream_total_avg_ms": _mean([r["total_ms"] for r in stream_results]) if stream_results else None,
                    "stream_ttfe_avg_ms": (
                        _mean([r["time_to_first_event_ms"] for r in stream_results
                               if r["time_to_first_event_ms"] >= 0])
                        if stream_results else None
                    ),
                    "step_violations": step_violations if args.expect_steps else None,
                },
            }
            out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nresults_json={out_path}")

    if args.expect_steps and step_violations > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
