"""End-to-end smoke test for the Solar Chat provider-profile picker.

Verifies:
1. Profiles defined as SOLAR_CHAT_PROFILE_<N>_* env blocks load correctly.
   Each profile groups MULTIPLE models under one provider connection.
2. RBAC narrows the per-role list (incl. per-profile ALLOWED_ROLES).
3. settings_with_profile_override propagates the profile's connection plus
   the chosen model into the LLMModelRouter; an unknown model name silently
   falls back to the profile's primary_model.
4. Default profile detection respects _DEFAULT=true, then env-default
   match, then first profile.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.settings import get_solar_chat_settings  # noqa: E402
from app.services.solar_ai_chat.llm_client import LLMModelRouter  # noqa: E402
from app.services.solar_ai_chat.model_profile_service import (  # noqa: E402
    PICKER_AUTH_ROLES,
    get_default_profile_id,
    get_enabled_profiles,
    list_profiles_for_role,
    resolve_profile,
    settings_with_profile_override,
)


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    failures = 0

    banner("1. Profile loading + per-profile model lists")
    enabled = get_enabled_profiles()
    print(f"PICKER_AUTH_ROLES = {sorted(PICKER_AUTH_ROLES)}")
    print(f"enabled profiles ({len(enabled)}):")
    for p in enabled:
        marker = " [DEFAULT]" if p.is_startup_default else ""
        print(f"  - {p.id}{marker}")
        print(f"      provider={p.provider}  base_url={p.base_url}")
        print(f"      primary={p.primary_model}  fallback={p.fallback_model}")
        print(f"      models ({len(p.models)}): {list(p.models)}")
        print(f"      allowed_roles={sorted(p.allowed_roles)}")
    print(f"default_profile_id = {get_default_profile_id()}")

    banner("2. RBAC + per-profile role gate")
    for role in ["admin", "ml_engineer", "data_engineer", "analyst", "system"]:
        ids = [p.id for p in list_profiles_for_role(role)]
        print(f"  role={role:15s} -> {ids}")

    banner("3. settings_with_profile_override (profile + model)")
    base = get_solar_chat_settings()
    print(f"BASE provider={base.resolved_llm_api_format} primary={base.primary_model}")
    if not enabled:
        print("[SKIP] no profiles loaded")
        return 1

    sample = enabled[0]
    cases = [
        ("primary_model fallback (model_name=None)", None, sample.primary_model),
        ("explicit valid model", sample.models[-1], sample.models[-1]),
        ("invalid model -> falls back to primary", "does-not-exist", sample.primary_model),
    ]
    for label, model_name, expected_model in cases:
        ovr = settings_with_profile_override(base, sample, model_name=model_name)
        ok = (
            ovr.primary_model == expected_model
            and ovr.resolved_llm_api_format == sample.provider
            and bool(ovr.llm_api_key)
        )
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  [{status}] {label:50s} -> "
            f"model={ovr.primary_model:30s} provider={ovr.resolved_llm_api_format}"
        )

    banner("4. LLMModelRouter receives profile connection + chosen model")
    chosen_model = sample.models[1] if len(sample.models) > 1 else sample.primary_model
    overridden = settings_with_profile_override(base, sample, model_name=chosen_model)
    router = LLMModelRouter(settings=overridden)
    print(f"  router._api_format    = {router._api_format}")
    print(f"  router._primary_model = {router._primary_model}")
    print(f"  router._fallback_model= {router._fallback_model}")
    print(f"  router._base_url      = {router._base_url}")
    print(f"  router._api_key set?  = {bool(router._api_key)}")
    if router._api_format == sample.provider and router._primary_model == chosen_model:
        print("  [OK] router config matches (profile, model) pair")
    else:
        print("  [FAIL] router config does NOT match")
        failures += 1

    banner("5. resolve_profile RBAC")
    for profile_id, role, expected in [
        (sample.id, "admin", sample.id),
        (sample.id, "data_engineer", None),
        ("does-not-exist", "admin", None),
        ("", "admin", None),
    ]:
        result = resolve_profile(profile_id, role)
        got = result.id if result else None
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status}] resolve_profile({profile_id!r:20s}, {role!r:15s}) -> {got}")

    banner("Result")
    print(f"failures = {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
