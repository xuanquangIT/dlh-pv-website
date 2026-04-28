"""In-memory cancellation registry for in-flight chat requests.

Each `handle_query_stream` / `handle_query` call registers a `threading.Event`
keyed by `trace_id`. The engine checks the event between LLM turns and raises
`EngineCancelled`. A `POST /solar-ai-chat/stop` endpoint sets the event so the
loop exits on its next checkpoint.

The registry is process-local and bounded (LRU-evicted at MAX_ACTIVE) so it
won't leak under abnormal termination paths that skip `unregister`.
"""
from __future__ import annotations

import threading
from collections import OrderedDict


MAX_ACTIVE = 256


class EngineCancelled(Exception):
    """Raised inside the engine when the cancellation event is set."""


_LOCK = threading.Lock()
_EVENTS: "OrderedDict[str, threading.Event]" = OrderedDict()


def register(trace_id: str) -> threading.Event:
    ev = threading.Event()
    with _LOCK:
        _EVENTS[trace_id] = ev
        while len(_EVENTS) > MAX_ACTIVE:
            _EVENTS.popitem(last=False)
    return ev


def cancel(trace_id: str) -> bool:
    with _LOCK:
        ev = _EVENTS.get(trace_id)
    if ev is None:
        return False
    ev.set()
    return True


def unregister(trace_id: str) -> None:
    with _LOCK:
        _EVENTS.pop(trace_id, None)


def is_cancelled(trace_id: str) -> bool:
    with _LOCK:
        ev = _EVENTS.get(trace_id)
    return ev is not None and ev.is_set()
