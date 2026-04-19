"""UI feature-visibility flags for the Solar AI Chat surface.

These flags govern *what the client is allowed to render* for a given role —
tool names, tool arguments, execution duration, the thinking-trace accordion,
memory search results, and verbose error detail. They are distinct from the
topic/tool access permissions in `permissions.py`: a role may still *call* a
tool while being prevented from *seeing* its name or arguments in the UI.

The backend resolves the flag set from the user's ChatRole on every response
and returns it as `ui_features` so the frontend can gate rendering without
duplicating the role mapping.
"""
from __future__ import annotations

from app.schemas.solar_ai_chat.enums import ChatRole


class UiFeature(str):
    """String-constant namespace for UI feature flag keys."""

    SHOW_TOOL_NAMES = "show_tool_names"
    SHOW_TOOL_ARGUMENTS = "show_tool_arguments"
    SHOW_TOOL_DURATION_MS = "show_tool_duration_ms"
    SHOW_THINKING_TRACE = "show_thinking_trace"
    SHOW_MEMORY_RESULTS = "show_memory_results"
    SHOW_TOOL_ERROR_DETAIL = "show_tool_error_detail"


ALL_FEATURES: tuple[str, ...] = (
    UiFeature.SHOW_TOOL_NAMES,
    UiFeature.SHOW_TOOL_ARGUMENTS,
    UiFeature.SHOW_TOOL_DURATION_MS,
    UiFeature.SHOW_THINKING_TRACE,
    UiFeature.SHOW_MEMORY_RESULTS,
    UiFeature.SHOW_TOOL_ERROR_DETAIL,
)


ROLE_UI_FEATURES: dict[ChatRole, set[str]] = {
    ChatRole.ADMIN: set(ALL_FEATURES),
    ChatRole.DATA_ENGINEER: {
        UiFeature.SHOW_TOOL_NAMES,
        UiFeature.SHOW_TOOL_DURATION_MS,
        UiFeature.SHOW_THINKING_TRACE,
    },
    ChatRole.ML_ENGINEER: {
        UiFeature.SHOW_TOOL_NAMES,
        UiFeature.SHOW_THINKING_TRACE,
    },
    ChatRole.DATA_ANALYST: {
        UiFeature.SHOW_TOOL_NAMES,
    },
}


def resolve_ui_features(role: ChatRole | None) -> dict[str, bool]:
    """Return a flat flag map (feature_key -> bool) for the given role.

    Unknown roles receive an empty grant (everything False).
    """
    granted = ROLE_UI_FEATURES.get(role, set()) if role is not None else set()
    return {feature: (feature in granted) for feature in ALL_FEATURES}
