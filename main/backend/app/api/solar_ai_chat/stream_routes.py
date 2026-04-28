"""SSE streaming route for Solar AI Chat.

POST /solar-ai-chat/stream
  – Accepts the same body as /query (SolarChatRequest)
  – Returns text/event-stream
  – Each SSE event is a JSON-encoded UiEvent (see schemas/solar_ai_chat/stream.py)
  – Final event is always ``done`` (full response) or ``error``

Auth model: identical to /query — requires authenticated user with one of the
permitted roles.  Session ownership is validated the same way.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import require_role
from app.db.database import AuthUser
from app.schemas.solar_ai_chat import SolarChatRequest
from app.schemas.solar_ai_chat.stream import ErrorEvent

from app.api.solar_ai_chat.routes import (
    _get_history_repository,
    _resolve_chat_service_for_request,
    _resolve_user_chat_role,
)
from app.repositories.solar_ai_chat.base_repository import DatabricksDataUnavailableError
from app.services.solar_ai_chat.cancellation import cancel as _cancel_trace

logger = logging.getLogger(__name__)

stream_router = APIRouter(prefix="/solar-ai-chat", tags=["Solar AI Chat — SSE"])


def _error_sse(message: str, code: str = "stream_error") -> str:
    return "data: " + json.dumps(ErrorEvent(message=message, code=code).model_dump()) + "\n\n"


@stream_router.post("/stream")
def stream_solar_ai_chat(
    request: SolarChatRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
):
    """Stream Solar AI Chat response as Server-Sent Events.

    Event types (``event`` JSON field):
    - ``thinking_step``  — a tool call has started (step index, tool name, human label)
    - ``tool_result``    — a tool call has finished (status, metric keys, duration ms)
    - ``status_update``  — free-form status text (shown in the loading bar)
    - ``text_delta``     — an incremental token chunk (reserved for future LLM streaming)
    - ``done``           — final full response payload; stream ends after this
    - ``error``          — fatal error; stream ends after this
    """
    service = _resolve_chat_service_for_request(request, current_user)
    effective_role = _resolve_user_chat_role(current_user)

    # Validate session ownership (same guard as /query)
    if request.session_id:
        history_repo = _get_history_repository()
        owned = history_repo.session_exists(
            session_id=request.session_id,
            owner_user_id=str(current_user.id),
        )
        if not owned:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    scoped_request = request.model_copy(update={"role": effective_role})

    def event_generator():
        try:
            yield from service.handle_query_stream(scoped_request)
        except DatabricksDataUnavailableError:
            yield _error_sse(
                "Databricks is temporarily unavailable. Please retry shortly.",
                code="databricks_unavailable",
            )
        except PermissionError as pe:
            yield _error_sse(str(pe), code="permission_denied")
        except ValueError as ve:
            yield _error_sse(str(ve), code="bad_request")
        except Exception as exc:
            logger.exception("solar_chat_stream_route_unhandled: %s", exc)
            yield _error_sse(
                "Solar AI Chat is temporarily unavailable. Please retry shortly.",
                code="stream_error",
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering for live streaming
            "Access-Control-Allow-Origin": "*",
        },
    )


class StopRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=64)


class StopResponse(BaseModel):
    cancelled: bool


@stream_router.post("/stop", response_model=StopResponse)
def stop_solar_ai_chat(
    body: StopRequest,
    current_user: AuthUser = Depends(require_role(["admin", "data_engineer", "ml_engineer"])),
) -> StopResponse:
    """Cancel an in-flight chat run by trace_id.

    Sets the cancellation event for the trace. The engine checks the event
    between LLM turns and raises EngineCancelled on the next checkpoint, so
    the actual stop happens when the current LLM call returns (soft stop —
    same UX as ChatGPT's stop button). Returns ``{cancelled: false}`` if
    the trace_id is unknown (already finished or never existed).
    """
    cancelled = _cancel_trace(body.trace_id)
    if cancelled:
        logger.info("solar_chat_stop trace_id=%s by user=%s",
                    body.trace_id, current_user.username)
    return StopResponse(cancelled=cancelled)
