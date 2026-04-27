"""Solar AI Chat — dispatcher.

Maps a function-call name (from the LLM) → primitive call → result envelope.
This replaces the legacy 14-tool ToolExecutor (~700 lines) with ~50 lines.

Usage from chat_service:
    dispatcher = Dispatcher(settings, role_id="admin")
    result = dispatcher.execute(function_name, arguments)
    # result has shape {"function_name": ..., "result": <primitive output>,
    #                    "ok": bool, "duration_ms": int}
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.core.settings import SolarChatSettings
from app.services.solar_ai_chat import primitives
from app.services.solar_ai_chat.databricks_adapter import (
    make_sample_executor,
    make_sql_executor,
)
from app.services.solar_ai_chat.semantic_loader import (
    SemanticLayer,
    load_semantic_layer,
)

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    function_name: str
    ok: bool
    result: dict[str, Any]
    duration_ms: int


class Dispatcher:
    """Dispatches LLM function calls to engine primitives.

    Constructed once per session (per chat request, per role). Holds a
    Databricks executor + the semantic layer.
    """

    KNOWN_PRIMITIVES = (
        "discover_schema",
        "inspect_table",
        "recall_metric",
        "execute_sql",
        "render_visualization",
    )

    def __init__(
        self,
        settings: SolarChatSettings,
        role_id: str,
        *,
        semantic_layer: SemanticLayer | None = None,
        sql_executor: Callable[[str], list[dict[str, Any]]] | None = None,
        sample_executor: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._settings = settings
        self._role_id = role_id
        self._semantic_layer = semantic_layer or load_semantic_layer()
        self._sql_executor = sql_executor or make_sql_executor(settings)
        self._sample_executor = sample_executor or make_sample_executor(settings)

    def execute(
        self,
        function_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> DispatchResult:
        args = arguments or {}
        started = time.time()
        ok = True
        try:
            result = self._dispatch(function_name, args)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "dispatch_error fn=%s args=%s err=%s",
                function_name, args, exc,
            )
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "guidance": "Check arguments and retry. Use discover_schema or inspect_table to confirm names.",
            }
            ok = False

        # Soft-error pass: primitives like inspect_table return
        # {"error": "..."} for unknown tables without raising. Treat those
        # as failed so loop-detection + traces reflect reality (otherwise a
        # model that fans out inspect_table on 5 hallucinated tables sees
        # 5 green checks in the trace UI).
        if ok and isinstance(result, dict) and "error" in result:
            useful_keys = {"rows", "matches", "spec", "format", "tables",
                           "schemas", "fqn", "columns"}
            if not (useful_keys & set(result.keys())):
                ok = False

        duration_ms = int((time.time() - started) * 1000)
        return DispatchResult(
            function_name=function_name,
            ok=ok,
            result=result,
            duration_ms=duration_ms,
        )

    def _dispatch(self, fn: str, args: dict[str, Any]) -> dict[str, Any]:
        if fn == "discover_schema":
            return primitives.discover_schema(
                role_id=self._role_id,
                domain=args.get("domain"),
                semantic_layer=self._semantic_layer,
            )
        if fn == "inspect_table":
            return primitives.inspect_table(
                table_fqn=args["table_fqn"],
                role_id=self._role_id,
                sample_executor=self._sample_executor,
                semantic_layer=self._semantic_layer,
            )
        if fn == "recall_metric":
            return primitives.recall_metric(
                query=args["query"],
                role_id=self._role_id,
                top_k=int(args.get("top_k", 5)),
                semantic_layer=self._semantic_layer,
            )
        if fn == "execute_sql":
            return primitives.execute_sql(
                sql=args["sql"],
                role_id=self._role_id,
                max_rows=int(args.get("max_rows", 1000)),
                sql_executor=self._sql_executor,
                semantic_layer=self._semantic_layer,
            )
        if fn == "render_visualization":
            return primitives.render_visualization(
                spec=args["spec"],
                data=args.get("data") or [],
                title=args.get("title"),
            )

        return {
            "error": f"Unknown function: {fn!r}",
            "available_primitives": list(self.KNOWN_PRIMITIVES),
        }
