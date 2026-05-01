# PV Lakehouse Dashboard

A full-stack analytics portal for solar photovoltaic (PV) energy monitoring, built on a **Medallion Architecture** (Bronze → Silver → Gold) powered by **Databricks**. Includes a Solar AI Chat assistant that composes SQL over a YAML semantic layer.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend (Jinja2)                         │
│  Dashboard · Solar AI Chat · Quality · Forecast · Pipeline   │
│  Training · Registry · Accounts                              │
├──────────────────────────────────────────────────────────────┤
│                  FastAPI Backend (Python)                     │
│  Auth (JWT) · REST API · Solar AI Chat (5 primitives loop)   │
├─────────────┬──────────────────────────┬─────────────────────┤
│  Neon       │  Databricks SQL          │  LLM Provider       │
│  PostgreSQL │  Warehouse(s)            │  (Profile picker:   │
│  Auth /     │  Silver/Gold analytics   │  OpenAI · Gemini ·  │
│  Chat       │  + Solar Chat warehouse  │  Anthropic · local) │
└─────────────┴──────────────────────────┴─────────────────────┘
```

**Key components:**

- **Neon PostgreSQL** — User auth, chat sessions / messages, tool-usage telemetry.
- **Databricks SQL Warehouse** — Energy readings, weather, air quality, forecasts, model monitoring (catalog `pv`, schemas `silver` and `gold`). A separate isolated warehouse can be configured for Solar Chat workloads.
- **LLM Providers** — Multi-provider, env-driven profile system: OpenAI-compatible, Gemini, Anthropic. Admin / ML engineers can switch provider+model at runtime via a UI picker.

## Project Structure

```
dlh-pv-website/
├── .env                              # Environment config (gitignored)
├── .env.example                      # Template — copy to .env and fill in
├── requirements.txt
├── README.md
└── main/
    ├── 002-create-lakehouse-tables.sql  # PostgreSQL bootstrap (auth + chat tables)
    ├── 005-app_setup.py
    ├── backend/
    │   └── app/
    │       ├── main.py                  # FastAPI app factory
    │       ├── api/
    │       │   ├── auth/
    │       │   ├── solar_ai_chat/
    │       │   │   ├── routes.py        # /query, /sessions, /llm-profiles, /admin
    │       │   │   └── stream_routes.py # SSE /stream
    │       │   ├── data_pipeline/
    │       │   ├── data_quality/
    │       │   ├── forecast/
    │       │   ├── ml_training/
    │       │   ├── model_registry/
    │       │   └── frontend.py
    │       ├── core/                    # Settings (settings.py)
    │       ├── schemas/
    │       │   └── solar_ai_chat/       # Pydantic models for chat / stream / viz
    │       ├── services/
    │       │   └── solar_ai_chat/
    │       │       ├── chat_service.py          # HTTP plumbing + persistence
    │       │       ├── engine.py                # ChatEngine — agentic loop
    │       │       ├── dispatcher.py            # function_name → primitive call
    │       │       ├── primitives.py            # 5 primitive implementations
    │       │       ├── tool_schemas.py          # TOOL_SCHEMAS + SYSTEM_PROMPT
    │       │       ├── semantic_loader.py       # Load metrics.yaml, RBAC filter
    │       │       ├── databricks_adapter.py    # BaseRepository → callable adapter
    │       │       ├── llm_client.py            # Multi-provider router
    │       │       ├── model_profile_service.py # Env-driven provider registry
    │       │       ├── permissions.py
    │       │       └── semantic/
    │       │           └── metrics.yaml         # Tables + metrics + roles
    │       └── repositories/
    │           └── solar_ai_chat/
    │               ├── base_repository.py       # Databricks SQL connection
    │               ├── chat_repository.py       # Thin facade
    │               ├── postgres_history_repository.py
    │               └── tool_usage_repository.py
    ├── scripts/                        # CLI utilities
    │   ├── solar_chat_eval_cli.py            # capture / inspect / judge / report
    │   ├── solar_chat_perf_cli.py            # Latency benchmark
    │   ├── solar_chat_model_picker_smoke.py
    │   ├── solar_chat_warehouse_smoke.py
    │   └── validate_metrics_yaml.py          # Static + live metrics validation
    ├── tests/
    │   ├── unit/                       # 600+ tests — engine, dispatcher, primitives,
    │   │                                # SQL safety, semantic layer, LLM client
    │   ├── integration/                # Routes (auth + RBAC + query/sessions/stream)
    │   └── eval/question_sets/         # regression_v1.yaml + phase3_smoke.yaml
    └── frontend/
        ├── templates/platform_portal/
        └── static/
            ├── css/
            └── js/
                ├── platform_portal/
                │   └── solar_chat_page.js
                └── components/
                    ├── chart_renderer.js  # Vega-Lite + Leaflet dispatcher
                    ├── model_picker.js    # Provider/model selector (admin/ml_eng)
                    ├── data_table.js
                    └── kpi_cards.js
```

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/dashboard` | KPI cards, energy charts, system overview |
| Solar AI Chat | `/solar-chat` | Full-page agentic AI assistant |
| Data Quality | `/quality` | Per-facility quality scores, issue tracking |
| Forecast | `/forecast` | 7-day energy forecast (D+1, D+3, D+5, D+7) with confidence intervals |
| Pipeline | `/pipeline` | Medallion pipeline stage progress & diagnostics |
| Model Registry | `/registry` | Model version registry & comparison |
| Accounts | `/settings/accounts` | User management (admin only) |

## Solar AI Chat

A single stable agentic engine. The LLM composes 5 generic primitives over a YAML semantic layer instead of being routed by hardcoded tools.

### Five primitives

| Primitive | Purpose |
|---|---|
| `discover_schema(domain?)` | List available tables, optionally filtered by topic |
| `inspect_table(table_fqn)` | Show columns, types, sample rows |
| `recall_metric(query)` | Semantic search over canonical SQL templates in `metrics.yaml` |
| `execute_sql(sql, max_rows?)` | Read-only SELECT with sqlglot-style safety + auto-LIMIT |
| `render_visualization(spec, data, title?)` | Emit a Vega-Lite chart spec |

The semantic layer (`services/solar_ai_chat/semantic/metrics.yaml`) has 19 canonical tables + 24+ metrics + 4 role policies. Edit YAML to add a new metric — no Python change required. Loader cached via `lru_cache`.

### Forecast model (D+1, D+3, D+5, D+7)

The ML pipeline registers four horizon-specific models in MLflow:

| Model | Horizon | Used for |
|---|---|---|
| `pv.gold.daily_forecast_d1` | D+1 (next day) | Same-day operational decisions |
| `pv.gold.daily_forecast_d3` | D+3 | 72-hour lookahead |
| `pv.gold.daily_forecast_d5` | D+5 | Mid-week scheduling |
| `pv.gold.daily_forecast_d7` | D+7 (1 week) | Weekly planning |

`pv.gold.forecast_daily` stores rows for each `(facility_id, forecast_date, forecast_horizon)` triple. Per-horizon accuracy metrics (R², RMSE, NRMSE, skill score) live in `pv.gold.model_monitoring_daily` and feed the canonical `model_metadata` metric so the chat can answer "what model is in production / what's the current accuracy?".

### Engine guards

Built into `engine.py`:

- **Off-topic regex** — generic chitchat / math / code-help / sport queries refuse before tool calls fire.
- **Conceptual question regex** — definitional questions ("Performance Ratio là gì") return text-only without SQL.
- **Per-tool persistent-loop ban** — `recall_metric` / `discover_schema` threshold = 4 turns; default = 3.
- **Auto-execute fallback** — when `recall_metric` is banned and there's a top match, the engine renders its SQL template so weak models still produce data-grounded answers.
- **Post-hoc hedge replacer** — if model returns "I can't fetch that…" but rows are present, swap in a deterministic draft.
- **Reasoning-model CoT scrubber** — strips `<think>...</think>` blocks AND inline English meta-prose paragraphs (Minimax M2.7, DeepSeek-R1, Qwen-QwQ pattern).
- **Missing-column suppression** — when answer says "không có data for X" the engine drops chart + data_table so the user doesn't see irrelevant numbers next to the refusal.

### Model Picker

Admin and `ml_engineer` users see a provider/model dropdown in the chat toolbar. Configuration is entirely env-driven — add a new `SOLAR_CHAT_PROFILE_<N>_*` block to `.env` to expose a new provider without any code change.

```env
SOLAR_CHAT_PROFILE_1_PROVIDER=openai        # wire format: openai | gemini | anthropic
SOLAR_CHAT_PROFILE_1_BASE_URL=https://api.openai.com/v1
SOLAR_CHAT_PROFILE_1_MODELS=gpt-4.1,gpt-4o  # CSV of selectable models
SOLAR_CHAT_PROFILE_1_PRIMARY_MODEL=gpt-4.1
SOLAR_CHAT_PROFILE_1_DEFAULT=true
```

### Eval / regression CLI

`scripts/solar_chat_eval_cli.py` captures full chat envelopes, optionally judges them with a separate LLM, and writes a markdown report:

```powershell
# Capture against a question set
python main/backend/scripts/solar_chat_eval_cli.py capture `
  --base-url http://127.0.0.1:8000 `
  --username admin --password admin123 `
  --question-set main/backend/tests/eval/question_sets/regression_v1.yaml `
  --output reports/regression.jsonl

# Force a specific model profile
python main/backend/scripts/solar_chat_eval_cli.py capture `
  --question-set main/backend/tests/eval/question_sets/phase3_smoke.yaml `
  --model-profile-id minimax --model-name MiniMax-M2.7 `
  --output reports/smoke_minimax.jsonl
```

Available question sets:
- `regression_v1.yaml` — 62 questions across 13 categories
- `phase3_smoke.yaml` — 12-question CI smoke set

### Key Features

- **Bilingual** — Vietnamese and English with code-switch detection.
- **Strict RBAC** — per-role table + metric allowlists in `metrics.yaml` `roles:` section.
- **Multi-provider LLM** — profile-based routing with automatic primary→fallback failover.
- **Tool usage telemetry** — every primitive call logged to `chat_tool_usage`; admin aggregate at `GET /solar-ai-chat/admin/tool-stats?days=7`.
- **Vega-Lite + Leaflet** — chart_renderer.js dispatches by `payload.format`.
- **Dev-only affordances** — model picker and thinking-trace panel visible to admin/ml_engineer or when `APP_ENV=dev` / `?debug=1`.

## Prerequisites

- Python 3.11+
- Neon PostgreSQL (or local Postgres)
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

Creates: `auth_roles`, `auth_users`, `chat_sessions`, `chat_messages`, `chat_tool_usage`.

### 4. Run the server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir main/backend

.\.venv\Scripts\Activate.ps1 ; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir main/backend --reload
```

### 5. Open the app

- Login: `http://127.0.0.1:8000/login`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Solar AI Chat: `http://127.0.0.1:8000/solar-chat`

## Testing

```powershell
# Full test suite (600+ unit + integration)
python -m pytest main/backend/tests/ -q

# Just unit tests with coverage gate
python -m pytest main/backend/tests/unit/ --cov=app --cov-fail-under=85 -q

# Static metrics.yaml validation (no DB)
python main/backend/scripts/validate_metrics_yaml.py --static

# Live metrics validation (DESCRIBE TABLE + EXPLAIN per metric)
python main/backend/scripts/validate_metrics_yaml.py --live

# Latency benchmark
python main/backend/scripts/solar_chat_perf_cli.py `
  --base-url http://127.0.0.1:8000 `
  --username admin --password admin123 `
  --message "Give me a quick PV Lakehouse overview" --print-answer

# Eval CLI smoke (12-question set)
python main/backend/scripts/solar_chat_eval_cli.py capture `
  --base-url http://127.0.0.1:8000 `
  --username admin --password admin123 `
  --question-set main/backend/tests/eval/question_sets/phase3_smoke.yaml `
  --output reports/smoke.jsonl

# Model picker smoke test
python main/backend/scripts/solar_chat_model_picker_smoke.py `
  --base-url http://127.0.0.1:8000 `
  --username admin --password admin123

# Warehouse connectivity smoke
python main/backend/scripts/solar_chat_warehouse_smoke.py
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named app` | Wrong working directory | Run from `dlh-pv-website/` with `--app-dir main/backend` |
| Port already in use | Stale server process | `Get-NetTCPConnection -LocalPort 8000 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| `.env` parse error | Non-comment plain text line | Prefix section headers with `#` |
| Profile not appearing in picker | Missing `PRIMARY_MODEL` | Add `SOLAR_CHAT_PROFILE_<N>_PRIMARY_MODEL=...` or check server log for `profile_skipped` warning |
| OpenRouter 404 "No endpoints available" | Privacy guardrail blocking free models | Go to openrouter.ai/settings/privacy → enable free providers |
| Tool-calling errors | Model doesn't support function calling | Switch to a profile whose provider supports tool calling (GPT-4+, Gemini Flash, Claude Haiku, MiniMax-M2.7) |
| Chat shows English chain-of-thought | Reasoning model leaking CoT | Already filtered for `<think>` tags + inline English prose; if a new pattern leaks, extend `_INLINE_COT_OPENERS` in `engine.py` |
| Forecast question returns wrong rows | Old null-horizon backfill | All forecast metrics now filter `forecast_horizon IS NOT NULL` (D+1/D+3/D+5/D+7) |

## Security

- Never commit `.env` — it is gitignored; use `.env.example` as the template
- Rotate API keys immediately if exposed in logs or chat history
- Set `AUTH_COOKIE_SECURE=true` in production (requires HTTPS)
- All SQL queries use parameterised statements + sqlglot-style validation — no string interpolation
