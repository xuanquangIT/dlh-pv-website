"""Solar AI Chat — legacy tool declaration stubs (Phase 4 cleanup).

The 14-tool v1 surface (get_system_overview / get_energy_performance / …)
was removed in Phase 4. The v2 engine declares its own tool schemas in
``services/solar_ai_chat/v2/tool_schemas.py`` (6 generic primitives).

These two names are kept as empty exports because:
- ``permissions.py`` still imports ``TOOL_NAME_TO_TOPIC`` to derive the
  optional ``ROLE_TOOL_PERMISSIONS`` map.
- ``schemas/solar_ai_chat/__init__.py`` re-exports both for any external
  caller. The dropdown in the chat UI is now driven from
  ``v2/tool_schemas.py`` directly.

Anything wanting the v2 tool palette should import
``app.services.solar_ai_chat.v2.tool_schemas.V2_TOOL_SCHEMAS`` instead.
"""
from __future__ import annotations

TOOL_DECLARATIONS: list[dict] = []
TOOL_NAME_TO_TOPIC: dict[str, str] = {}
