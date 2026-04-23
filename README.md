# PV Lakehouse Dashboard

A full-stack analytics dashboard for solar photovoltaic (PV) energy monitoring, built on a **Medallion Architecture** (Bronze → Silver → Gold) powered by **Databricks**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Jinja2)                     │
│  Dashboard · Solar Chat · Quality · Forecast · Pipeline │
│  Training · Registry · Accounts                         │
├─────────────────────────────────────────────────────────┤
│                FastAPI Backend (Python)                  │
│  Auth (JWT) · REST API · Solar AI Chat Service          │
├──────────┬──────────────────────────┬───────────────────┤
│  Neon    │  Databricks SQL          │  LLM Provider     │
│  Postgres│  Warehouse               │  (GPT-4o/Gemini)  │
│  (Auth,  │  (Silver/Gold layers)    │  (Tool-calling)   │
│  Chat)   │                          │                   │
└──────────┴──────────────────────────┴───────────────────┘
```

**Key components:**

- **Neon PostgreSQL** — User auth, chat sessions/messages, RAG documents
- **Databricks SQL Warehouse** — Energy readings, weather, air quality, forecasts, model monitoring (catalog `pv`, schemas `silver` and `gold`)
- **LLM Provider** — GPT-4o (OpenAI), Gemini, or Anthropic for agentic tool-calling chat

## Project Structure

```
dlh-pv-website/
├── .env                              # Environment configuration
├── requirements.txt                  # Python dependencies
├── README.md
└── main/
    ├── 002-create-lakehouse-tables.sql  # PostgreSQL bootstrap
    ├── 005-app_setup.py                 # Initial setup script
    ├── backend/
    │   └── app/
    │       ├── main.py                  # FastAPI app entry point
    │       ├── api/                     # Route handlers
    │       │   ├── auth/                # Login/logout, JWT sessions
    │       │   ├── dashboard/           # Dashboard data endpoints
    │       │   ├── solar_ai_chat/       # Chat API (query, sessions)
    │       │   ├── data_pipeline/       # Pipeline monitoring API
    │       │   ├── data_quality/        # Quality metrics API
    │       │   ├── forecast/            # Forecast data API
    │       │   ├── ml_training/         # Training metrics API
    │       │   ├── model_registry/      # Model registry API
    │       │   └── frontend.py          # Template rendering routes
    │       ├── core/                    # Settings, config
    │       ├── db/                      # Database connections
    │       ├── schemas/                 # Pydantic models & tool schemas
    │       ├── services/
    │       │   ├── auth/                # Auth service
    │       │   ├── dashboard/           # Dashboard service
    │       │   ├── databricks_service.py # Databricks SQL connector
    │       │   └── solar_ai_chat/       # AI Chat module
    │       │       ├── chat_service.py      # Main agentic loop
    │       │       ├── llm_client.py        # Multi-provider LLM client
    │       │       ├── tool_executor.py     # Tool dispatch
    │       │       ├── intent_service.py    # Intent classification
    │       │       ├── prompt_builder.py    # System prompt builder
    │       │       ├── nlp_parser.py        # Date/entity extraction
    │       │       ├── permissions.py       # RBAC for tools
    │       │       └── ...                   # web_search_client.py removed in Phase 1
    │       └── repositories/
    │           ├── auth/                # User/role queries
    │           └── solar_ai_chat/       # Data access layer
    │               ├── base_repository.py       # Shared Databricks logic
    │               ├── topic_repository.py      # Per-topic metrics queries
    │               ├── report_repository.py     # Station daily reports
    │               ├── extreme_repository.py    # Extreme value queries
    │               ├── chat_repository.py       # Facility info
    │               ├── postgres_history_repository.py  # Chat history (Postgres only — Task 1.1)
    │               ├── tool_usage_repository.py       # Tool-call telemetry (Task 0.1)
    │               └── vector_repository.py     # RAG vector search
    └── frontend/
        ├── templates/
        │   └── platform_portal/
        │       ├── base.html            # Layout template
        │       ├── dashboard.html       # Main dashboard
        │       ├── solar_chat.html      # AI Chat interface
        │       ├── quality.html         # Data quality dashboard
        │       ├── forecast.html        # Forecast dashboard
        │       ├── pipeline.html        # Pipeline monitoring
        │       ├── training.html        # ML training dashboard
        │       ├── registry.html        # Model registry
        │       ├── accounts.html        # User management
        │       ├── login.html           # Login page
        │       └── components/          # Shared UI components
        └── static/
            ├── css/
            │   ├── platform_portal.css       # Main layout & components
            │   ├── solar_chat_premium.css     # Chat UI enhancements
            │   ├── chatbot-bubble.css         # Floating chatbot widget
            │   ├── app.css                    # Global styles
            │   ├── data_pipeline.css          # Pipeline page styles
            │   ├── platform_accounts.css      # Accounts page styles
            │   └── platform_auth.css          # Auth pages styles
            └── js/platform_portal/
                ├── solar_chat_page.js    # Full-page AI chat client
                ├── chatbot_widget.js     # Floating chatbot widget
                ├── chatbot-bubble.js     # Bubble animations
                ├── common.js             # Shared utilities & page init
                ├── charts.js             # Chart.js helpers
                ├── data_pipeline.js      # Pipeline page logic
                ├── data_quality.js       # Quality page logic
                ├── forecast.js           # Forecast page logic
                ├── training.js           # Training page logic
                ├── registry.js           # Registry page logic
                └── accounts_page.js      # User accounts logic
```

## Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/platform/dashboard` | KPI cards, energy charts, system overview |
| Solar AI Chat | `/platform/solar-chat` | Full-page agentic AI assistant with tool-calling |
| Data Quality | `/platform/quality` | Per-facility quality scores, issue tracking |
| Forecast | `/platform/forecast` | 72-hour energy forecasts with confidence intervals |
| Pipeline | `/platform/pipeline` | Medallion pipeline stage progress & diagnostics |
| ML Training | `/platform/training` | Model training metrics & evaluation |
| Model Registry | `/platform/registry` | Model version registry & comparison |
| Accounts | `/platform/accounts` | User management (admin only) |

## Solar AI Chat — Agentic Tool-Calling

The chat module implements an **agentic loop** where the LLM can call backend tools to retrieve live data from Databricks:

**Available tools:**

| Tool | Returns |
|---|---|
| `get_system_overview` | Production MWh, R², quality score, facility count, latest data timestamp |
| `get_energy_performance` | Top/bottom facilities, peak hours, capacity factors, forecast |
| `get_ml_model_info` | Model name/version, R², skill score, NRMSE |
| `get_pipeline_status` | Bronze/Silver/Gold stage progress, alerts |
| `get_data_quality_issues` | Per-facility quality scores, likely causes, latest data timestamp |
| `get_forecast_72h` | 3-day forecast with confidence intervals |
| `get_station_daily_report` | Per-station daily data (energy, radiation, weather) for a specific date; supports `station_name` filtering |
| `get_facility_info` | Facility metadata (location, capacity, timezone) |
| `get_extreme_*` | Record values for AQI, energy, weather |

> Phase 1 cuts: the `web_search` tool and its Tavily client were removed
> in favour of staying on internal data only (`solar-chat-upgrade-plan`
> Task 1.2). Other removals in the same phase: Databricks chat-history
> backend (Postgres-only now, Task 1.1) and the legacy regex router
> fallback (Task 1.3).

**Key features:**

- **Intent pre-fetch**: Automatically detects topic from user message and pre-loads relevant data
- **Date-aware queries**: Extracts dates and applies timezone-aware query logic
- **Strict RBAC Tool Isolation**: Roles (Data Engineer, ML Engineer, Data Analyst, Admin) are structurally isolated; the LLM engine physically cannot see or invoke tools outside the authenticated user's permission domain.
- **Dynamic Gold KPI Querying**: Parses and searches dynamic Gold-layer schema attributes safely, effectively replacing hardcoded rules.
- **Multi-provider LLM**: Supports OpenAI, Gemini, and Anthropic APIs with automatic fallback
- **Bilingual**: Vietnamese and English
- **Tool usage telemetry** (Task 0.1): every tool call writes one row to
  `chat_tool_usage`. Admins can query aggregates via
  `GET /solar-ai-chat/admin/tool-stats?days=7`.
- **Dev-only affordances** (Task 1.4/1.5): the widget's role picker and
  the thinking-trace panel are hidden unless `APP_ENV=dev`, the viewing
  user is `admin`/`ml_engineer`/`data_engineer`, or `?debug=1` is on the
  URL. Backend RBAC is unchanged.

## Prerequisites

- Python 3.11+
- Neon PostgreSQL connection string
- Databricks workspace + SQL Warehouse + PAT token
- LLM API key (OpenAI recommended)

## Quick Start

### 1. Virtual environment

```powershell
cd dlh-pv-website
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment configuration

Create/edit `.env` in the project root:

```env
# Database
DATABASE_URL=postgresql://...
POSTGRES_SSLMODE=require
POSTGRES_CHANNEL_BINDING=require
# Chat history is always Postgres; SOLAR_CHAT_HISTORY_BACKEND,
# SOLAR_CHAT_WEBSEARCH_*, and SOLAR_AI_LEGACY_ROUTER_ENABLED are no longer read.

# Optional: force dev affordances on (role picker, trace panel for analyst)
# APP_ENV=dev

# Optional: expose RAG (search_documents) tool to the agent. Default off;
# enable only when docs are actually ingested, otherwise the tool clutters
# the agent palette and can drift synthesis on unrelated queries.
# SOLAR_CHAT_RAG_ENABLED=1

# Databricks
DATABRICKS_HOST=https://...
DATABRICKS_TOKEN=dapi...
DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/...
UC_CATALOG=pv
UC_SILVER_SCHEMA=silver
UC_GOLD_SCHEMA=gold

# LLM (OpenAI recommended)
SOLAR_CHAT_LLM_API_FORMAT=openai
SOLAR_CHAT_LLM_API_KEY=sk-...
SOLAR_CHAT_PRIMARY_MODEL=gpt-4o
SOLAR_CHAT_FALLBACK_MODEL=gpt-4o-mini

# Auth
SECRET_KEY=your-secret-key
```

> **Note:** Non-comment plain text lines in `.env` must be `KEY=VALUE` format. Use `#` for section headers.

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

Tables created: `auth_roles`, `auth_users`, `chat_sessions`, `chat_messages`, `rag_documents`

### 4. Run the server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir main/backend
```

### 5. Open the app

- Login: `http://127.0.0.1:8000/login`
- Dashboard: `http://127.0.0.1:8000/platform/dashboard`
- Solar AI Chat: `http://127.0.0.1:8000/platform/solar-chat`

## Testing

### Chatbot latency benchmark

```powershell
python main/backend/scripts/solar_chat_perf_cli.py \
  --base-url http://127.0.0.1:8000 \
  --mode full \
  --username admin --password admin123 \
  --role data_engineer \
  --message "Give me a quick PV Lakehouse overview" \
  --repeat 1 --print-answer
```

### Accuracy regression suite

```powershell
python main/backend/scripts/solar_chat_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --username admin --password admin123 \
  --role admin --strict-exit
```

Reports: `main/backend/test_reports/solar_chat_accuracy/`

### Unit tests

```powershell
python -m pytest main/backend/tests/unit/ -q
```

### Web pytest suite with coverage gate (CI/CD)

```powershell
python -m pytest \
  main/backend/tests/integration/test_frontend_pages.py \
  main/backend/tests/integration/test_auth_login_flow.py \
  main/backend/tests/integration/test_permission_matrix.py \
  main/backend/tests/integration/test_auth_admin_routes.py \
  --cov=app.api.frontend \
  --cov=app.api.auth.routes \
  --cov=app.main \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=90 \
  -q
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named app` | Wrong working directory | Run from `dlh-pv-website/` with `--app-dir main/backend` |
| Port already in use | Stale server process | `Get-NetTCPConnection -LocalPort 8000 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| `.env` parse error | Non-comment plain text line | Prefix section headers with `#` |
| Tool-calling errors | Model doesn't support function calling | Switch to GPT-4o or enable fallback |
| Chat returns "future date" error | LLM doesn't know today's date | Ensure `prompt_builder.py` injects current date (already implemented) |

## Security

- Never commit `.env` or real API keys
- Rotate tokens if exposed in logs or chat history
- All SQL queries use sanitized parameters to prevent injection
