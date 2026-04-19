from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Force UTF-8 stdout so Vietnamese diacritics render on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Solar AI Chat response time from CLI.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Backend base URL.")
    parser.add_argument("--username", default="admin", help="Login username.")
    parser.add_argument("--password", default="admin123", help="Login password.")
    parser.add_argument(
        "--mode",
        choices=["chat", "full", "model-only"],
        default="chat",
        help=(
            "Benchmark mode: "
            "chat=/solar-ai-chat/query (matches website), "
            "full=/query/benchmark, model-only=/query/benchmark/model-only"
        ),
    )
    parser.add_argument("--role", default="data_engineer", help="Chat role value in request body.")
    parser.add_argument("--message", required=True, help="Message to send for benchmark.")
    parser.add_argument("--session-id", default="", help="Existing chat session id. If omitted, script will create one.")
    parser.add_argument("--session-title", default="CLI performance benchmark", help="Title used when auto-creating a session.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of benchmark requests.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Overall request timeout.")
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0, help="Connection timeout.")
    parser.add_argument("--print-answer", action="store_true", help="Print assistant answer text.")
    parser.add_argument("--print-metrics", action="store_true", help="Print key_metrics JSON from chat response.")
    parser.add_argument(
        "--print-thinking",
        action="store_true",
        help="Print the full thinking_trace (planner actions, tool calls, reflection).",
    )
    parser.add_argument(
        "--print-sources",
        action="store_true",
        help="Print the list of data sources cited for the answer.",
    )
    parser.add_argument(
        "--expect-thinking-trace",
        action="store_true",
        help="Fail with exit code 1 if response does not include planner thinking_trace steps.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write raw benchmark results as JSON.",
    )
    return parser.parse_args()


def _build_timeout(total_timeout: float, connect_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(
        timeout=total_timeout,
        connect=connect_timeout,
    )


def login(client: httpx.Client, username: str, password: str) -> None:
    response = client.post(
        "/auth/login",
        data={
            "username": username,
            "password": password,
            "next": "/dashboard",
        },
        follow_redirects=False,
    )

    if response.status_code not in (302, 303):
        raise RuntimeError(
            f"Login failed with status {response.status_code}: {response.text[:300]}"
        )

    if not client.cookies:
        raise RuntimeError("Login succeeded but no auth cookie was set.")


def create_session(client: httpx.Client, role: str, title: str) -> str:
    response = client.post(
        "/solar-ai-chat/sessions",
        json={"role": role, "title": title},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Create session failed with status {response.status_code}: {response.text[:300]}"
        )

    payload = response.json()
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise RuntimeError("Create session response missing session_id.")
    return session_id


def _extract_trace_fields(payload: dict[str, Any]) -> tuple[str, str, int]:
    trace = payload.get("thinking_trace")
    if not isinstance(trace, dict):
        return "", "", 0
    summary = str(trace.get("summary", "") or "")
    trace_id = str(trace.get("trace_id", "") or "")
    raw_steps = trace.get("steps")
    step_count = len(raw_steps) if isinstance(raw_steps, list) else 0
    return summary, trace_id, step_count


def benchmark_full_once(
    client: httpx.Client,
    role: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    request_payload = {
        "role": role,
        "session_id": session_id,
        "message": message,
    }

    started = time.perf_counter()
    response = client.post("/solar-ai-chat/query/benchmark", json=request_payload)
    roundtrip_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Benchmark request failed with status {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()
    chat_response = payload.get("response") or {}
    trace_summary, trace_id, trace_step_count = _extract_trace_fields(chat_response)

    return {
        "benchmark_type": str(payload.get("benchmark_type", "full_pipeline")),
        "roundtrip_ms": roundtrip_ms,
        "server_elapsed_ms": int(payload.get("server_elapsed_ms", -1)),
        "service_latency_ms": int(payload.get("service_latency_ms", chat_response.get("latency_ms", -1))),
        "route_overhead_ms": int(payload.get("route_overhead_ms", -1)),
        "answer": str(chat_response.get("answer", "")),
        "topic": str(chat_response.get("topic", "")),
        "model_used": str(chat_response.get("model_used", "")),
        "fallback_used": bool(chat_response.get("fallback_used", False)),
        "warning_message": str(chat_response.get("warning_message", "") or ""),
        "key_metrics": chat_response.get("key_metrics", {}),
        "thinking_trace_summary": trace_summary,
        "thinking_trace_id": trace_id,
        "thinking_step_count": trace_step_count,
    }


def chat_once(
    client: httpx.Client,
    role: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    request_payload = {
        "role": role,
        "session_id": session_id,
        "message": message,
    }

    started = time.perf_counter()
    response = client.post("/solar-ai-chat/query", json=request_payload)
    roundtrip_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Chat request failed with status {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()
    trace_summary, trace_id, trace_step_count = _extract_trace_fields(payload)
    trace_steps_raw: list[dict[str, Any]] = []
    trace = payload.get("thinking_trace")
    if isinstance(trace, dict):
        raw_steps = trace.get("steps")
        if isinstance(raw_steps, list):
            for s in raw_steps:
                if isinstance(s, dict):
                    trace_steps_raw.append({
                        "step": str(s.get("step", "")),
                        "detail": str(s.get("detail", "")),
                        "status": str(s.get("status", "")),
                    })
    return {
        "benchmark_type": "chat",
        "roundtrip_ms": roundtrip_ms,
        "latency_ms": int(payload.get("latency_ms", -1)),
        "answer": str(payload.get("answer", "")),
        "topic": str(payload.get("topic", "")),
        "model_used": str(payload.get("model_used", "")),
        "fallback_used": bool(payload.get("fallback_used", False)),
        "warning_message": str(payload.get("warning_message", "") or ""),
        "intent_confidence": float(payload.get("intent_confidence", 0.0)),
        "key_metrics": payload.get("key_metrics", {}),
        "sources": payload.get("sources", []),
        "thinking_trace_summary": trace_summary,
        "thinking_trace_id": trace_id,
        "thinking_step_count": trace_step_count,
        "thinking_steps": trace_steps_raw,
    }


def benchmark_model_only_once(
    client: httpx.Client,
    role: str,
    message: str,
) -> dict[str, Any]:
    request_payload = {
        "role": role,
        "message": message,
    }

    started = time.perf_counter()
    response = client.post("/solar-ai-chat/query/benchmark/model-only", json=request_payload)
    roundtrip_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Model-only benchmark request failed with status {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()
    model_response = payload.get("response") or {}

    return {
        "benchmark_type": str(payload.get("benchmark_type", "model_only")),
        "roundtrip_ms": roundtrip_ms,
        "server_elapsed_ms": int(payload.get("server_elapsed_ms", -1)),
        "model_generation_ms": int(payload.get("model_generation_ms", -1)),
        "route_overhead_ms": int(payload.get("route_overhead_ms", -1)),
        "error": str(payload.get("error", "")).strip(),
        "answer": str(model_response.get("answer", "")),
        "model_used": str(model_response.get("model_used", "")),
        "fallback_used": bool(model_response.get("fallback_used", False)),
    }


def _mean(values: list[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    # Nearest-rank percentile: ceil(0.95 * N) on 1-indexed positions.
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    args = parse_args()
    repeat = max(1, int(args.repeat))

    timeout = _build_timeout(args.timeout_seconds, args.connect_timeout_seconds)
    base_url = args.base_url.rstrip("/")

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        login(client=client, username=args.username, password=args.password)
        session_id = ""
        if args.mode in {"full", "chat"}:
            session_id = args.session_id.strip()
            if not session_id:
                session_id = create_session(client=client, role=args.role, title=args.session_title)

        print(f"base_url={base_url}")
        print(f"mode={args.mode}")
        if session_id:
            print(f"session_id={session_id}")
        print(f"repeat={repeat}")

        results: list[dict[str, Any]] = []
        trace_violations = 0
        for run_index in range(1, repeat + 1):
            if args.mode == "full":
                result = benchmark_full_once(
                    client=client,
                    role=args.role,
                    session_id=session_id,
                    message=args.message,
                )
                print(
                    "run={run} type={rtype} roundtrip_ms={roundtrip} server_elapsed_ms={server} "
                    "service_latency_ms={service} route_overhead_ms={overhead} model={model} fallback={fallback} "
                    "trace_steps={trace_steps}".format(
                        run=run_index,
                        rtype=result["benchmark_type"],
                        roundtrip=result["roundtrip_ms"],
                        server=result["server_elapsed_ms"],
                        service=result["service_latency_ms"],
                        overhead=result["route_overhead_ms"],
                        model=result["model_used"],
                        fallback=result["fallback_used"],
                        trace_steps=result.get("thinking_step_count", 0),
                    )
                )
                if result.get("warning_message"):
                    print(f"  warning={result['warning_message']}")
            elif args.mode == "chat":
                result = chat_once(
                    client=client,
                    role=args.role,
                    session_id=session_id,
                    message=args.message,
                )
                print(
                    "run={run} type={rtype} roundtrip_ms={roundtrip} latency_ms={latency} "
                    "topic={topic} model={model} fallback={fallback} intent_confidence={intent:.2f} "
                    "source_count={source_count} trace_steps={trace_steps}".format(
                        run=run_index,
                        rtype=result["benchmark_type"],
                        roundtrip=result["roundtrip_ms"],
                        latency=result["latency_ms"],
                        topic=result["topic"],
                        model=result["model_used"],
                        fallback=result["fallback_used"],
                        intent=result["intent_confidence"],
                        source_count=len(result.get("sources", [])),
                        trace_steps=result.get("thinking_step_count", 0),
                    )
                )
                if result.get("warning_message"):
                    print(f"  warning={result['warning_message']}")
            else:
                result = benchmark_model_only_once(
                    client=client,
                    role=args.role,
                    message=args.message,
                )
                print(
                    "run={run} type={rtype} roundtrip_ms={roundtrip} server_elapsed_ms={server} "
                    "model_generation_ms={model_ms} route_overhead_ms={overhead} model={model} fallback={fallback}".format(
                        run=run_index,
                        rtype=result["benchmark_type"],
                        roundtrip=result["roundtrip_ms"],
                        server=result["server_elapsed_ms"],
                        model_ms=result["model_generation_ms"],
                        overhead=result["route_overhead_ms"],
                        model=result["model_used"],
                        fallback=result["fallback_used"],
                    )
                )
                if result["error"]:
                    print(f"  model_error={result['error']}")

            if args.expect_thinking_trace and args.mode in {"chat", "full"}:
                if int(result.get("thinking_step_count", 0)) <= 0:
                    trace_violations += 1
                    print("  trace_check=FAILED expected thinking_trace steps > 0")

            results.append(result)

        roundtrip_values = [int(item.get("roundtrip_ms", 0)) for item in results]
        server_values = [
            int(item.get("server_elapsed_ms", -1))
            for item in results
            if int(item.get("server_elapsed_ms", -1)) >= 0
        ]
        service_values = [
            int(item.get("service_latency_ms", -1))
            for item in results
            if int(item.get("service_latency_ms", -1)) >= 0
        ]
        model_values = [
            int(item.get("model_generation_ms", -1))
            for item in results
            if int(item.get("model_generation_ms", -1)) >= 0
        ]
        chat_latency_values = [
            int(item.get("latency_ms", -1))
            for item in results
            if int(item.get("latency_ms", -1)) >= 0
        ]

        print("summary:")
        print(f"  roundtrip_avg_ms={_mean(roundtrip_values)}")
        print(f"  roundtrip_p95_ms={_p95(roundtrip_values)}")
        if server_values:
            print(f"  server_elapsed_avg_ms={_mean(server_values)}")
            print(f"  server_elapsed_p95_ms={_p95(server_values)}")
        if args.mode == "full" and service_values:
            print(f"  service_latency_avg_ms={_mean(service_values)}")
            print(f"  service_latency_p95_ms={_p95(service_values)}")
        if args.mode == "chat" and chat_latency_values:
            print(f"  chat_latency_avg_ms={_mean(chat_latency_values)}")
            print(f"  chat_latency_p95_ms={_p95(chat_latency_values)}")
        if args.mode == "model-only" and model_values:
            print(f"  model_generation_avg_ms={_mean(model_values)}")
            print(f"  model_generation_p95_ms={_p95(model_values)}")

        if args.mode in {"chat", "full"}:
            trace_present = sum(1 for item in results if int(item.get("thinking_step_count", 0)) > 0)
            print(f"  thinking_trace_present={trace_present}/{len(results)}")
            if args.expect_thinking_trace:
                print(f"  thinking_trace_violations={trace_violations}")

        if args.print_thinking and results:
            steps = results[-1].get("thinking_steps") or []
            print("thinking_trace:")
            if steps:
                for idx, s in enumerate(steps, 1):
                    print(f"  [{idx}] {s.get('step', '')} — {s.get('status', '')}: {s.get('detail', '')}")
            else:
                print("  (no thinking_trace steps returned by backend)")
        if args.print_sources and results:
            srcs = results[-1].get("sources") or []
            print("sources:")
            if srcs:
                for s in srcs:
                    if isinstance(s, dict):
                        layer = s.get("layer", "")
                        dataset = s.get("dataset", "")
                        url = s.get("url", "")
                        tail = f" ({url})" if url else ""
                        print(f"  - {layer}:{dataset}{tail}")
                    else:
                        print(f"  - {s}")
            else:
                print("  (no sources returned)")
        if args.print_answer and results:
            print("answer:")
            print(results[-1]["answer"])
        if args.print_metrics and results:
            print("key_metrics:")
            print(json.dumps(results[-1].get("key_metrics", {}), ensure_ascii=False, indent=2))

        if args.output_json.strip():
            output_path = Path(args.output_json).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            payload = {
                "base_url": base_url,
                "mode": args.mode,
                "repeat": repeat,
                "results": results,
                "summary": {
                    "roundtrip_avg_ms": _mean(roundtrip_values),
                    "roundtrip_p95_ms": _p95(roundtrip_values),
                    "thinking_trace_present": (
                        sum(1 for item in results if int(item.get("thinking_step_count", 0)) > 0)
                        if args.mode in {"chat", "full"}
                        else None
                    ),
                    "thinking_trace_violations": trace_violations if args.expect_thinking_trace else None,
                },
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"results_json={output_path}")

    if args.expect_thinking_trace and args.mode in {"chat", "full"} and trace_violations > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
