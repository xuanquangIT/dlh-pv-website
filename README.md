# PV Lakehouse Web - Local Run Guide

This guide is for the local web app runtime in this folder.

Runtime architecture:

- Neon PostgreSQL stores web app metadata (auth, chat history, RAG documents).
- Databricks SQL Warehouse provides analytics data for the chatbot (catalog `pv`, schemas `silver` and `gold`).
- Local default flow does not require Docker or Trino.

## 1) Prerequisites

- Python 3.11+
- PowerShell (Windows) or Bash (Linux/WSL)
- Neon PostgreSQL connection string
- Databricks workspace, SQL Warehouse, and PAT token

## 2) Create a virtual environment

Run from this folder:

```powershell
cd dlh-pv-website
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 3) Configure environment

Main runtime file: [.env](.env)

Required variables:

- `DATABASE_URL=postgresql://...`
- `POSTGRES_SSLMODE=require`
- `POSTGRES_CHANNEL_BINDING=require`
- `SOLAR_CHAT_HISTORY_BACKEND=postgres`
- `DATABRICKS_HOST=https://...`
- `DATABRICKS_TOKEN=...`
- `DATABRICKS_SQL_HTTP_PATH=/sql/1.0/warehouses/...`
- `DATABRICKS_WAREHOUSE_ID=...` (optional alternative to `DATABRICKS_SQL_HTTP_PATH`)
- `UC_CATALOG=pv`
- `UC_SILVER_SCHEMA=silver`
- `UC_GOLD_SCHEMA=gold`

Recommended LLM variables for GPT-4o tool-calling:

- `SOLAR_CHAT_LLM_API_FORMAT=openai`
- `SOLAR_CHAT_LLM_API_KEY=...`
- `SOLAR_CHAT_PRIMARY_MODEL=gpt-4o`
- `SOLAR_CHAT_FALLBACK_MODEL=gpt-4o-mini`
- `SOLAR_CHAT_LLM_BASE_URL=https://api.openai.com/v1` (optional if using default OpenAI endpoint)

Important:

- Any plain text line in `.env` must start with `#` if it is a section title.
- Example: use `# Solar AI Chat`, not `Solar AI Chat`.

## 4) Bootstrap metadata tables in Neon

Bootstrap SQL: [main/002-create-lakehouse-tables.sql](main/002-create-lakehouse-tables.sql)

Quick run (without `psql` CLI):

```powershell
cd dlh-pv-website
.\.venv\Scripts\python.exe -c "import os; from pathlib import Path; import psycopg2; from dotenv import load_dotenv; load_dotenv('.env'); dsn=os.getenv('DATABASE_URL'); sql=Path('main/002-create-lakehouse-tables.sql').read_text(encoding='utf-8'); conn=psycopg2.connect(dsn); conn.autocommit=True; cur=conn.cursor(); cur.execute(sql); cur.close(); conn.close(); print('bootstrap_ok')"
```

Main tables created:

- `auth_roles`
- `auth_users`
- `chat_sessions`
- `chat_messages`
- `rag_documents`

## 5) Run backend API (correct command usage)

Option A: from repo root

```powershell
Set-Location dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --app-dir main/backend
```

Option B: when you are already in `dlh-pv-website`

```powershell
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --app-dir main/backend
```

Do not run `Set-Location dlh-pv-website` again if your current directory is already `dlh-pv-website`.

## 6) Open the web app

After the server starts:

- Login page: `http://127.0.0.1:8001/login`
- Solar chat page: `http://127.0.0.1:8001/solar-chat`

## 7) Quick smoke tests

1. Login behavior:

- `GET /` returns `303` to `/login` when unauthenticated.
- `POST /auth/login` returns `303` on successful login redirect.

2. Verify chat sessions are written to Neon:

```powershell
cd dlh-pv-website
.\.venv\Scripts\python.exe -c "import os, psycopg2; from dotenv import load_dotenv; load_dotenv('.env'); conn=psycopg2.connect(os.getenv('DATABASE_URL')); cur=conn.cursor(); cur.execute('select count(*) from chat_sessions'); print('chat_sessions', cur.fetchone()[0]); cur.close(); conn.close()"
```

## 8) Troubleshooting

### A) `ModuleNotFoundError: No module named app`

Cause: wrong working directory or missing `--app-dir`.

Fix: use one of the commands in section 5 exactly.

### B) `Set-Location` path not found (`...\dlh-pv-website\dlh-pv-website`)

Cause: running `Set-Location dlh-pv-website` while already in `dlh-pv-website`.

Fix: skip `Set-Location` and run only the `python -m uvicorn ...` command.

### C) `Python-dotenv could not parse statement ...`

Cause: invalid `.env` syntax (usually a non-comment plain text line).

Fix: convert section headers to comments (prefix with `#`) and keep lines in `KEY=VALUE` format.

### D) Tool-calling model errors (for example `tool_use_failed`)

Some OpenAI-compatible local models do not support function calling reliably.

Recommended actions:

- Switch to a model with reliable tool-calling support.
- Keep fallback behavior enabled so the app still returns data-backed summaries.

## 9) Benchmark chatbot latency from CLI

Script: [main/backend/scripts/solar_chat_perf_cli.py](main/backend/scripts/solar_chat_perf_cli.py)

Full pipeline benchmark:

```powershell
cd dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe main/backend/scripts/solar_chat_perf_cli.py --base-url http://127.0.0.1:8001 --mode full --username admin --password admin123 --role data_engineer --message "Give me a quick PV Lakehouse overview" --repeat 1 --print-answer
```

Chat endpoint benchmark (same route as website UI):

```powershell
cd dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe main/backend/scripts/solar_chat_perf_cli.py --base-url http://127.0.0.1:8001 --mode chat --username admin --password admin123 --role data_engineer --message "Compare 2 largest facilities" --repeat 1 --print-answer --print-metrics
```

Model-only benchmark (no tools / no RAG):

```powershell
cd dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe main/backend/scripts/solar_chat_perf_cli.py --base-url http://127.0.0.1:8001 --mode model-only --username admin --password admin123 --role data_engineer --message "Summarize the current system status" --repeat 1 --print-answer
```

Key metrics in output:

- `roundtrip_ms`: client-side total latency
- `server_elapsed_ms`: endpoint processing time
- `service_latency_ms`: `SolarAIChatService` time (full mode)
- `model_generation_ms`: model generation time (model-only mode)
- `route_overhead_ms`: endpoint overhead above service/model time

## 10) Accuracy and regression testing

End-to-end bilingual + Databricks verification suite:

```powershell
cd dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe main/backend/scripts/solar_chat_accuracy_suite.py --base-url http://127.0.0.1:8001 --username admin --password admin123 --role admin --strict-exit
```

Output reports are written under:

- `main/backend/test_reports/solar_chat_accuracy/solar_chat_accuracy_latest.md`
- `main/backend/test_reports/solar_chat_accuracy/solar_chat_accuracy_latest.json`

Targeted backend regression tests for routing + fallback guards:

```powershell
cd dlh-pv-website
d:/University/HK8/dlh-pv/.venv/Scripts/python.exe -m pytest main/backend/tests/unit/test_solar_chat_intent_service.py main/backend/tests/unit/test_facility_fallback_guard.py main/backend/tests/unit/test_prompt_builder_energy_kpis.py main/backend/tests/unit/test_solar_ai_chat_service_energy_kpis.py main/backend/tests/unit/test_solar_ai_chat_service_facility_websearch.py -q
```

## 11) Security notes

- Never commit real secrets.
- Rotate tokens and API keys if they were exposed in logs or chat history.
