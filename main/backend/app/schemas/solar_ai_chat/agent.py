"""Pydantic schemas for the planner-first agent architecture.

Includes:
- PlannerOutput: structured action plan produced by LLM planner
- EvidenceItem / EvidenceStore: unified evidence container for retrieved facts
- ToolResultEnvelope: standardised tool result contract
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Planner schemas
# ---------------------------------------------------------------------------

class PlannerAction(BaseModel):
    """A single action requested by the planner."""

    tool: str = Field(
        description=(
            "Tool name to invoke, or 'answer_directly' if no external data is needed."
        )
    )
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", description="Why this action was chosen.")


class PlannerOutput(BaseModel):
    """Structured output from the QueryPlanner LLM call."""

    intent_type: str = Field(
        description="High-level intent: e.g. 'data_query', 'definition', 'comparison', 'general'."
    )
    actions: list[PlannerAction] = Field(
        default_factory=list,
        description="Ordered list of actions to execute.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Planner confidence in this plan.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when the query is too ambiguous to plan reliably.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="If needs_clarification, the follow-up question to ask the user.",
    )


# ---------------------------------------------------------------------------
# Evidence schemas
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """One piece of retrieved evidence from any tool or retrieval path."""

    source: str = Field(description="Dataset name, RAG source file, or web URL.")
    tool: str = Field(description="Which tool or path produced this evidence.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: str | None = Field(default=None, description="ISO-8601 when data was fetched.")
    payload: dict[str, Any] = Field(default_factory=dict, description="The actual data payload.")
    data_source: str = Field(default="databricks")


class EvidenceStore(BaseModel):
    """Accumulates all evidence collected during a single query lifecycle."""

    items: list[EvidenceItem] = Field(default_factory=list)

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def merge_payload(self) -> dict[str, Any]:
        """Merge all evidence payloads into a flat dict (last-write-wins for conflicts)."""
        merged: dict[str, Any] = {}
        for item in self.items:
            merged.update(item.payload)
        return merged

    def to_source_metadata_dicts(self) -> list[dict[str, str]]:
        seen: set[tuple[str, str, str]] = set()
        result = []
        for item in self.items:
            key = (item.tool, item.source, item.data_source)
            if key not in seen:
                seen.add(key)
                result.append(
                    {"layer": item.tool, "dataset": item.source, "data_source": item.data_source}
                )
        return result


# ---------------------------------------------------------------------------
# Tool result envelope
# ---------------------------------------------------------------------------

class ToolResultEnvelope(BaseModel):
    """Standardised contract returned by ToolExecutor for every tool invocation."""

    status: Literal["ok", "error", "partial"] = "ok"
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, str]] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)
    tool_name: str = ""
