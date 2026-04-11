from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Solar AI Chat response time from CLI.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Backend base URL.")
    parser.add_argument("--username", default="admin", help="Login username.")
    parser.add_argument("--password", default="admin123", help="Login password.")
    parser.add_argument(
        "--mode",
        choices=["full", "model-only"],
        default="full",
        help="Benchmark mode: full pipeline or model-only (no RAG/data fetch).",
    )
    parser.add_argument("--role", default="data_engineer", help="Chat role value in request body.")
    parser.add_argument("--message", required=True, help="Message to send for benchmark.")
    parser.add_argument("--session-id", default="", help="Existing chat session id. If omitted, script will create one.")
    parser.add_argument("--session-title", default="CLI performance benchmark", help="Title used when auto-creating a session.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of benchmark requests.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Overall request timeout.")
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0, help="Connection timeout.")
    parser.add_argument("--print-answer", action="store_true", help="Print assistant answer text.")
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
    index = int((len(ordered) - 1) * 0.95)
    return ordered[index]


def main() -> int:
    args = parse_args()
    repeat = max(1, int(args.repeat))

    timeout = _build_timeout(args.timeout_seconds, args.connect_timeout_seconds)
    base_url = args.base_url.rstrip("/")

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        login(client=client, username=args.username, password=args.password)
        session_id = ""
        if args.mode == "full":
            session_id = args.session_id.strip()
            if not session_id:
                session_id = create_session(client=client, role=args.role, title=args.session_title)

        print(f"base_url={base_url}")
        print(f"mode={args.mode}")
        if session_id:
            print(f"session_id={session_id}")
        print(f"repeat={repeat}")

        results: list[dict[str, Any]] = []
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
                    "service_latency_ms={service} route_overhead_ms={overhead} model={model} fallback={fallback}".format(
                        run=run_index,
                        rtype=result["benchmark_type"],
                        roundtrip=result["roundtrip_ms"],
                        server=result["server_elapsed_ms"],
                        service=result["service_latency_ms"],
                        overhead=result["route_overhead_ms"],
                        model=result["model_used"],
                        fallback=result["fallback_used"],
                    )
                )
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

            results.append(result)

        roundtrip_values = [int(item["roundtrip_ms"]) for item in results]
        server_values = [int(item["server_elapsed_ms"]) for item in results if int(item["server_elapsed_ms"]) >= 0]
        service_values = [int(item["service_latency_ms"]) for item in results if int(item.get("service_latency_ms", -1)) >= 0]
        model_values = [int(item["model_generation_ms"]) for item in results if int(item.get("model_generation_ms", -1)) >= 0]

        print("summary:")
        print(f"  roundtrip_avg_ms={_mean(roundtrip_values)}")
        print(f"  roundtrip_p95_ms={_p95(roundtrip_values)}")
        if server_values:
            print(f"  server_elapsed_avg_ms={_mean(server_values)}")
            print(f"  server_elapsed_p95_ms={_p95(server_values)}")
        if args.mode == "full" and service_values:
            print(f"  service_latency_avg_ms={_mean(service_values)}")
            print(f"  service_latency_p95_ms={_p95(service_values)}")
        if args.mode == "model-only" and model_values:
            print(f"  model_generation_avg_ms={_mean(model_values)}")
            print(f"  model_generation_p95_ms={_p95(model_values)}")

        if args.print_answer and results:
            print("answer:")
            print(results[-1]["answer"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
