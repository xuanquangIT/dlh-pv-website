# Development Configuration

This folder stores development-only runtime overrides for PV Lakehouse.

## Solar AI Chat Setup

1. Create local environment file:

```powershell
Copy-Item dev/config/.env_example dev/config/.env -Force
```

2. Set valid Gemini API key in `dev/config/.env`:

- `SOLAR_CHAT_GEMINI_API_KEY`

3. Keep all application source code in `main/`.

- `dev/config` must only contain environment-specific configuration.

## Runtime Notes

- Solar AI Chat uses primary model `gemini-2.5-flash-lite` and fallback `gemini-2.5-flash`.
- Data access is read-only and constrained to Silver/Gold curated datasets under `main/sql` by default.
- If Gemini is unavailable, the service returns a safe data-backed summary.
