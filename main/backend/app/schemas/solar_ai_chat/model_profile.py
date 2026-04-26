"""Public API schemas for Solar Chat LLM model picker.

The picker is admin / ml_engineer only.  See
``services/solar_ai_chat/model_profile_service.py`` for the registry, RBAC,
and env-driven loader.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LLMProfileSummary(BaseModel):
    """Metadata-only view of a selectable provider profile.

    Excludes secrets and infra-only fields so it is safe to expose over HTTP.
    A profile bundles ONE provider configuration with N selectable models.
    """

    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(description="Stable profile identifier; pass back in chat requests.")
    label: str = Field(description="Human-readable name for the picker UI.")
    provider: str = Field(description="LLM API format: openai, anthropic, gemini.")
    primary_model: str = Field(description="Default model when this profile is picked.")
    models: list[str] = Field(description="All models the user may pick within this profile.")
    fallback_model: str | None = None
    is_default: bool = Field(default=False, description="True for the startup default profile.")


class LLMProfileList(BaseModel):
    profiles: list[LLMProfileSummary]
    default_profile_id: str | None = None
    default_model_name: str | None = Field(
        default=None,
        description="The startup default profile's primary_model (convenience for the UI).",
    )
