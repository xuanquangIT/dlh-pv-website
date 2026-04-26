# PV Lakehouse Dashboard

A full-stack analytics portal for solar photovoltaic (PV) energy monitoring, built on a **Medallion Architecture** (Bronze → Silver → Gold) powered by **Databricks**. Includes an agentic Solar AI Chat assistant with multi-provider LLM support.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend (Jinja2)                         │
│  Dashboard · Solar AI Chat · Quality · Forecast · Pipeline   │
│  Training · Registry · Accounts                              │
├──────────────────────────────────────────────────────────────┤
│                  FastAPI Backend (Python)                     │
│  Auth (JWT) · REST API · Solar AI Chat (agentic loop)        │
├─────────────┬──────────────────────────┬─────────────────────┤
│  Neon       │  Databricks SQL          │  LLM Provider       │
│  PostgreSQL │  Warehouse(s)            │  (Profile picker:   │
│  Auth/Chat/ │  Silver/Gold analytics   │  OpenAI · Gemini ·  │
│  pgvector   │  + Solar Chat warehouse  │  OpenRouter · local)│
└─────────────┴──────────────────────────┴─────────────────────┘
```

**Key components:**

- **Neon PostgreSQL** — User auth, chat sessions/messages, RAG documents (pgvector)
- **Databricks SQL Warehouse** — Energy readings, weather, air quality, forecasts, model monitoring (catalog `pv`, schemas `silver` and `gold`). A separate isolated warehouse can be configured for Solar Chat workloads.
- **LLM Providers** — Multi-provider, env-driven profile system: OpenAI-compatible, Gemini, Anthropic, or OpenRouter free models. Admin/ML engineers can switch provider+model at runtime via a UI picker.

## Project Structure

```
dlh-pv-website/
├── .env                              # Environment config (gitignored)
├── .env.example                      # Template — copy to .env and fill in
├── requirements.txt
├── README.md
└── main/
    ├── 002-create-lakehouse-tables.sql  # PostgreSQL bootstrap
    ├── 005-app_setup.py
    ├── backend/
    │   └── app/
    │       ├── main.py
    │       ├── api/
    │       │   ├── auth/
    │       │   ├── solar_ai_chat/       # query, stream, sessions, profiles, admin
    │       │   ├── data_pipeline/
    │       │   ├── data_quality/
    │       │   ├── forecast/
    │       │   ├── ml_training/
    │       │   ├── model_registry/
    │       │   └── frontend.py
    │       ├── core/                    # Settings (settings.py)
    │       ├── schemas/
    │       │   └── solar_ai_chat/
    │       │       ├── tools.py         # 14 tool definitions
    │       │       ├── chat.py          # Request/response models
    │       │       └── model_profile.py # Profile picker schemas
    │       ├── services/
    │       │   └── solar_ai_chat/
    │       │       ├── chat_service.py          # Agentic orchestration loop
    │       │       ├── llm_client.py            # Multi-provider LLM client
    │       │       ├── model_profile_service.py # Env-driven provider registry
    │       │       ├── tool_executor.py
    │       │       ├── intent_service.py
    │       │       ├── prompt_builder.py
    │       │       ├── embedding_client.py      # Primary + fallback key support
    │       │       └── ...
    │       ├── repositories/
    │       │   └── solar_ai_chat/
    │       │       ├── report_repository.py     # Daily + hourly reports (range mode)
    │       │       ├── postgres_history_repository.py
    │       │       ├── tool_usage_repository.py
    │       │       └── vector_repository.py
    │       └── scripts/                 # CLI tools for testing
    └── frontend/
        ├── templates/platform_portal/
        └── static/
            ├── css/
            │   ├── platform_portal.css
            │   └── solar_chat_premium.css
            └── js/
                ├── platform_portal/
                │   ├── solar_chat_page.js
                │   ├── chatbot_widget.js
                │   └── common.js
                └── components/
                    ├── model_picker.js   # Provider/model selector (admin/ml_eng)
                    └── tool_picker.js
```

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/dashboard` | KPI cards, energy charts, system overview |
| Solar AI Chat | `/solar-chat` | Full-page agentic AI assistant |
| Data Quality | `/quality` | Per-facility quality scores, issue tracking |
| Forecast | `/forecast` | 72-hour energy forecast with confidence intervals |
| Pipeline | `/pipeline` | Medallion pipeline stage progress & diagnostics |
| ML Training | `/training` | Model training metrics & evaluation |
| Model Registry | `/registry` | Model version registry & comparison |
| Accounts | `/settings/accounts` | User management (admin only) |

## Solar AI Chat

An agentic loop where the LLM calls backend tools to retrieve live Databricks data.

### Tools (14 total)

| Tool | Returns |
|---|---|
| `get_system_overview` | Production MWh, R², quality score, facility count |
| `get_energy_performance` | Top/bottom facilities, peak hours, capacity factors, forecast |
| `get_ml_model_info` | Model name/version, R², skill score, NRMSE |
| `get_pipeline_status` | Bronze/Silver/Gold stage progress, alerts |
| `get_data_quality_issues` | Per-facility quality scores, likely causes |
| `get_forecast_72h` | 3-day forecast with confidence intervals |
| `get_facility_info` | Facility metadata (location, capacity, timezone) |
| `get_station_daily_report` | Per-station daily energy/weather for a date or date range |
| `get_station_hourly_report` | Hourly trend for one day **or** AVG per-hour-of-day across a date range |
| `get_extreme_aqi` | AQI record values |
| `get_extreme_energy` | Energy output records |
| `get_extreme_weather` | Weather metric records |
| `query_gold_kpi` | Dynamic KPI mart query |
| `search_documents` | RAG search (opt-in, disabled by default) |

### Model Picker

Admin and `ml_engineer` users see a provider/model dropdown in the chat toolbar. Configuration is entirely env-driven — add a new `SOLAR_CHAT_PROFILE_<N>_*` block to `.env` to expose a new provider without any code change.

```
SOLAR_CHAT_PROFILE_1_PROVIDER=openai        # wire format: openai | gemini | anthropic
SOLAR_CHAT_PROFILE_1_BASE_URL=http://...    # any OpenAI-compatible endpoint
SOLAR_CHAT_PROFILE_1_MODELS=gpt-4.1,gpt-4o # CSV of selectable models
SOLAR_CHAT_PROFILE_1_PRIMARY_MODEL=gpt-4.1
SOLAR_CHAT_PROFILE_1_DEFAULT=true
```

### Key Features

- **Bilingual** — Vietnamese and English
- **Contextual follow-ups** — scope guard bypassed when conversation already established a domain context
- **Strict RBAC** — tool set is structurally isolated per role; LLM physically cannot invoke tools outside the user's permission domain
- **Multi-provider LLM** — profile-based routing with automatic primary→fallback failover
- **Embedding key fallback** — primary + fallback Gemini embedding key with auto fail-over on quota/auth errors
- **Tool usage telemetry** — every tool call logged to `chat_tool_usage`; admin aggregate at `GET /solar-ai-chat/admin/tool-stats?days=7`
- **Dev-only affordances** — model picker and thinking-trace panel visible to admin/ml_engineer or when `APP_ENV=dev` / `?debug=1`

## Prerequisites

- Python 3.11+
- Neon PostgreSQL (or local Postgres with pgvector)
- Databricks workspace + SQL Warehouse + PAT token
- At least one LLM API key (see `.env.example` for provider options)

## Quick Start

### 1. Virtual environment

```powershell
cd dlh-pv-website
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment configuration

```powershell
Copy-Item .env.example .env
# Then edit .env with your real values
```

Minimum required fields:

```env
DATABASE_URL=postgresql://...
DATABRICKS_HOST=https://...
DATABRICKS_TOKEN=dapi...
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/...
UC_CATALOG=pv

# At least one LLM profile:
SOLAR_CHAT_PROFILE_1_ID=my-provider
SOLAR_CHAT_PROFILE_1_PROVIDER=openai
SOLAR_CHAT_PROFILE_1_BASE_URL=https://api.openai.com/v1
SOLAR_CHAT_PROFILE_1_API_KEY=sk-...
SOLAR_CHAT_PROFILE_1_PRIMARY_MODEL=gpt-4o
SOLAR_CHAT_PROFILE_1_DEFAULT=true

AUTH_SECRET_KEY=<generate: openssl rand -base64 64>
```

### 3. Bootstrap database

```powershell
python -c "
import os; from pathlib import Path; import psycopg2; from dotenv import load_dotenv
load_dotenv('.env'); dsn=os.getenv('DATABASE_URL')
sql=Path('main/002-create-lakehouse-tables.sql').read_text(encoding='utf-8')
conn=psycopg2.connect(dsn); conn.autocommit=True
cur=conn.cursor(); cur.execute(sql); cur.close(); conn.close()
print('bootstrap_ok')
"
```

Creates: `auth_roles`, `auth_users`, `chat_sessions`, `chat_messages`, `chat_tool_usage`, `rag_documents`

### 4. Run the server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir main/backend
```

### 5. Open the app

- Login: `http://127.0.0.1:8000/login`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Solar AI Chat: `http://127.0.0.1:8000/solar-chat`

## Testing

```powershell
# Unit tests
python -m pytest main/backend/tests/unit/ -q

# Integration tests with coverage gate
python -m pytest main/backend/tests/integration/ --cov=app --cov-fail-under=90 -q

# Latency benchmark
python main/backend/scripts/solar_chat_perf_cli.py \
  --base-url http://127.0.0.1:8000 \
  --username admin --password admin123 \
  --message "Give me a quick PV Lakehouse overview" --print-answer

# Accuracy regression suite (bilingual)
python main/backend/scripts/solar_chat_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --username admin --password admin123 --role admin --strict-exit

# Model picker smoke test
python main/backend/scripts/solar_chat_model_picker_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --username admin --password admin123

# Hourly range mode smoke test
python main/backend/scripts/solar_chat_hourly_range_smoke.py
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named app` | Wrong working directory | Run from `dlh-pv-website/` with `--app-dir main/backend` |
| Port already in use | Stale server process | `Get-NetTCPConnection -LocalPort 8000 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| `.env` parse error | Non-comment plain text line | Prefix section headers with `#` |
| Profile not appearing in picker | Missing `PRIMARY_MODEL` | Add `SOLAR_CHAT_PROFILE_<N>_PRIMARY_MODEL=...` or check server log for `profile_skipped` warning |
| OpenRouter 404 "No endpoints available" | Privacy guardrail blocking free models | Go to openrouter.ai/settings/privacy → enable free providers |
| Embedding key quota exceeded | Primary key rate-limited | Set `SOLAR_CHAT_EMBEDDING_FALLBACK_API_KEY` — client auto-switches on 429 |
| Tool-calling errors | Model doesn't support function calling | Switch to a profile whose provider supports tool calling (GPT-4+, Gemini Flash, Claude Haiku) |
| Follow-up question refused by scope guard | Replied in a new session (no history) | History-based bypass only works within the same session |

## Security

- Never commit `.env` — it is gitignored; use `.env.example` as the template
- Rotate API keys immediately if exposed in logs or chat history
- Set `AUTH_COOKIE_SECURE=true` in production (requires HTTPS)
- All SQL queries use parameterised statements — no string interpolation
