"""Solar Chat — provider profile registry, RBAC, and settings overrides.

Concept
-------
A **profile** is one provider configuration (wire format + endpoint + api
key) that can serve **multiple models**. Picking from the UI selects a
``(profile_id, model_name)`` pair: the profile supplies the connection,
the model name supplies the LLM identifier sent on the wire.

This avoids the prior 1-profile-per-model bloat where 8 models sharing a
single proxy required 8 near-identical env blocks.

Env layout
----------
Each profile is one numbered block. Indices start at 1 and may have gaps.

  SOLAR_CHAT_PROFILE_1_ID=copilot-local                # required, unique
  SOLAR_CHAT_PROFILE_1_LABEL=Local Copilot proxy        # shown in dropdown
  SOLAR_CHAT_PROFILE_1_PROVIDER=openai                  # openai | anthropic | gemini
  SOLAR_CHAT_PROFILE_1_BASE_URL=http://localhost:3456/v1
  SOLAR_CHAT_PROFILE_1_API_KEY=token_xxx                # raw secret OR
  SOLAR_CHAT_PROFILE_1_API_KEY_ENV=COPILOT_LOCAL_API_KEY # ref to other env var
  SOLAR_CHAT_PROFILE_1_MODELS=gpt-4.1,gpt-4o,gpt-4o-mini # CSV, available models
  SOLAR_CHAT_PROFILE_1_PRIMARY_MODEL=gpt-4.1            # required, must be in MODELS
  SOLAR_CHAT_PROFILE_1_FALLBACK_MODEL=gpt-4o-mini       # optional
  SOLAR_CHAT_PROFILE_1_ALLOWED_ROLES=admin,ml_engineer  # optional, default: both
  SOLAR_CHAT_PROFILE_1_DEFAULT=true                     # optional, server startup default
  SOLAR_CHAT_PROFILE_1_DISABLED=false                   # optional, hide without delete

Permissions
-----------
Picker is restricted to ``PICKER_AUTH_ROLES`` (admin, ml_engineer).
Per-profile ``ALLOWED_ROLES`` can narrow further (e.g. an expensive
direct-API profile limited to admin only).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

from app.core.settings import SolarChatSettings, get_solar_chat_settings

logger = logging.getLogger(__name__)

load_dotenv(override=False)

PICKER_AUTH_ROLES: frozenset[str] = frozenset({"admin", "ml_engineer"})

_MAX_PROFILE_INDEX = 200


@dataclass(frozen=True)
class LLMProfile:
    """A provider connection + its list of selectable models."""

    id: str
    label: str
    provider: str  # "openai" | "anthropic" | "gemini"
    primary_model: str
    models: tuple[str, ...]
    fallback_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_startup_default: bool = False
    allowed_roles: frozenset[str] = field(default_factory=lambda: PICKER_AUTH_ROLES)

    def has_model(self, model_name: str) -> bool:
        return model_name in self.models

    def resolve_api_key(self) -> str | None:
        return (self.api_key or "").strip() or None

    def resolve_base_url(self) -> str | None:
        return (self.base_url or "").strip() or None


def _read_env(key: str) -> str:
    return (os.environ.get(key, "") or "").strip()


def _parse_csv(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(tok.strip() for tok in raw.split(",") if tok.strip())


def _parse_roles(raw: str) -> frozenset[str]:
    if not raw:
        return PICKER_AUTH_ROLES
    parsed = {tok.strip().lower() for tok in raw.split(",") if tok.strip()}
    intersected = parsed & PICKER_AUTH_ROLES
    return frozenset(intersected) if intersected else PICKER_AUTH_ROLES


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_env_var_name(value: str) -> bool:
    """Heuristic: env var names are uppercase letters / digits / underscores,
    typically <= 64 chars, never start with a digit. Anything else (lowercase,
    long, contains punctuation) is almost certainly a raw secret pasted in
    by mistake."""
    if not value or len(value) > 64:
        return False
    if value[0].isdigit():
        return False
    return all(c.isupper() or c.isdigit() or c == "_" for c in value)


def _load_one_profile(index: int) -> LLMProfile | None:
    prefix = f"SOLAR_CHAT_PROFILE_{index}_"

    profile_id = _read_env(prefix + "ID")
    if not profile_id:
        return None

    if _parse_bool(_read_env(prefix + "DISABLED")):
        logger.info("solar_chat_profile_skipped_disabled id=%s index=%d", profile_id, index)
        return None

    primary_model = _read_env(prefix + "PRIMARY_MODEL")
    models = _parse_csv(_read_env(prefix + "MODELS"))

    if not primary_model:
        if models:
            # Forgiving: if the operator forgot PRIMARY_MODEL but provided a
            # MODELS list, default to the first entry instead of silently
            # dropping the profile (which is hard to debug from the UI).
            primary_model = models[0]
            logger.warning(
                "solar_chat_profile_primary_defaulted id=%s index=%d "
                "primary_model=%s — PRIMARY_MODEL was unset, used first MODELS entry.",
                profile_id, index, primary_model,
            )
        else:
            logger.warning(
                "solar_chat_profile_skipped_missing_primary_model id=%s index=%d "
                "set SOLAR_CHAT_PROFILE_%d_PRIMARY_MODEL or _MODELS.",
                profile_id, index, index,
            )
            return None

    if not models:
        # Single-model profile — degenerate to the primary model only.
        models = (primary_model,)
    elif primary_model not in models:
        # Be lenient: prepend the primary so callers always get a valid set.
        models = (primary_model, *models)

    api_key = _read_env(prefix + "API_KEY")
    if not api_key:
        ref_env = _read_env(prefix + "API_KEY_ENV")
        if ref_env:
            api_key = _read_env(ref_env)
            if not api_key:
                # Two common mistakes: (a) the referenced env var is unset
                # in this deployment, (b) the user pasted the raw key here
                # instead of the env-var NAME (env var names are uppercase
                # alphanumeric+underscore, not key-shaped).
                if not _looks_like_env_var_name(ref_env):
                    logger.warning(
                        "solar_chat_profile_skipped id=%s index=%d reason=invalid_api_key_env "
                        "value_preview=%r — _API_KEY_ENV expects the NAME of another env var "
                        "(e.g. MY_PROVIDER_KEY), not the raw key. Either rename it to _API_KEY=<raw> "
                        "or add a separate env var with that name.",
                        profile_id, index, ref_env[:12] + "...",
                    )
                else:
                    logger.warning(
                        "solar_chat_profile_skipped id=%s index=%d reason=api_key_env_unset "
                        "ref_env=%s — declared env var does not exist or is empty.",
                        profile_id, index, ref_env,
                    )
                return None
    if not api_key:
        logger.warning(
            "solar_chat_profile_skipped id=%s index=%d reason=no_api_key "
            "set either SOLAR_CHAT_PROFILE_%d_API_KEY=<raw> or _API_KEY_ENV=<other-env-var-name>.",
            profile_id, index, index,
        )
        return None

    return LLMProfile(
        id=profile_id,
        label=_read_env(prefix + "LABEL") or profile_id,
        provider=_read_env(prefix + "PROVIDER").lower() or "openai",
        primary_model=primary_model,
        models=models,
        fallback_model=_read_env(prefix + "FALLBACK_MODEL") or None,
        base_url=_read_env(prefix + "BASE_URL") or None,
        api_key=api_key,
        is_startup_default=_parse_bool(_read_env(prefix + "DEFAULT")),
        allowed_roles=_parse_roles(_read_env(prefix + "ALLOWED_ROLES")),
    )


def _load_all_profiles() -> tuple[LLMProfile, ...]:
    profiles: list[LLMProfile] = []
    seen_ids: set[str] = set()
    for index in range(1, _MAX_PROFILE_INDEX + 1):
        profile = _load_one_profile(index)
        if profile is None:
            continue
        if profile.id in seen_ids:
            logger.warning(
                "solar_chat_profile_duplicate_id id=%s index=%d (ignored)",
                profile.id, index,
            )
            continue
        seen_ids.add(profile.id)
        profiles.append(profile)
    return tuple(profiles)


_profiles_cache: tuple[LLMProfile, ...] | None = None


def get_enabled_profiles() -> tuple[LLMProfile, ...]:
    global _profiles_cache
    if _profiles_cache is None:
        _profiles_cache = _load_all_profiles()
        logger.info(
            "solar_chat_profiles_loaded count=%d entries=%s",
            len(_profiles_cache),
            [
                {"id": p.id, "models": list(p.models), "default": p.is_startup_default}
                for p in _profiles_cache
            ],
        )
    return _profiles_cache


def invalidate_profile_cache() -> None:
    global _profiles_cache
    _profiles_cache = None
    get_default_profile_id.cache_clear()


def list_profiles_for_role(auth_role_id: str) -> tuple[LLMProfile, ...]:
    if auth_role_id not in PICKER_AUTH_ROLES:
        return ()
    return tuple(p for p in get_enabled_profiles() if auth_role_id in p.allowed_roles)


def find_profile(profile_id: str) -> LLMProfile | None:
    if not profile_id:
        return None
    for profile in get_enabled_profiles():
        if profile.id == profile_id:
            return profile
    return None


def resolve_profile(
    profile_id: str, auth_role_id: str
) -> LLMProfile | None:
    """RBAC + role-allowed filter. Returns None if the user can't use this profile."""
    if not profile_id or auth_role_id not in PICKER_AUTH_ROLES:
        return None
    profile = find_profile(profile_id)
    if profile is None:
        return None
    if auth_role_id not in profile.allowed_roles:
        return None
    return profile


def settings_with_profile_override(
    base_settings: SolarChatSettings,
    profile: LLMProfile,
    model_name: str | None = None,
) -> SolarChatSettings:
    """Return a copy of base settings with this profile's connection +
    chosen model applied. If ``model_name`` is unset or invalid, falls
    back to the profile's ``primary_model``."""
    chosen_model = (
        model_name
        if model_name and profile.has_model(model_name)
        else profile.primary_model
    )
    overrides: dict[str, object] = {
        "llm_api_format": profile.provider,
        "primary_model": chosen_model,
        "fallback_model": profile.fallback_model or chosen_model,
    }
    api_key = profile.resolve_api_key()
    if api_key is not None:
        overrides["llm_api_key"] = api_key
    base_url = profile.resolve_base_url()
    if base_url is not None:
        overrides["llm_base_url"] = base_url
    return base_settings.model_copy(update=overrides)


@lru_cache(maxsize=1)
def get_default_profile_id() -> str | None:
    """Identify the startup default profile. Priority:
    1. Profile with ``_DEFAULT=true`` (first match if multiple)
    2. Profile whose connection matches the legacy SOLAR_CHAT_* defaults
    3. First enabled profile (so the picker is never blank)
    """
    enabled = get_enabled_profiles()
    if not enabled:
        return None
    for profile in enabled:
        if profile.is_startup_default:
            return profile.id

    settings = get_solar_chat_settings()
    base_primary = (settings.primary_model or "").strip().lower()
    base_provider = settings.resolved_llm_api_format
    if base_primary:
        for profile in enabled:
            if (
                profile.primary_model.strip().lower() == base_primary
                and profile.provider == base_provider
            ):
                return profile.id

    return enabled[0].id
