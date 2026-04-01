# PV Lakehouse - Local Setup Guide

This guide explains how to set up a local Python virtual environment, install required libraries, and run Docker services (PostgreSQL and Trino).

## Prerequisites

- Python 3.11+
- Docker Desktop (with Docker Compose)
- PowerShell (Windows)

## 1) Create and activate a virtual environment

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2) Install Python dependencies

From repository root (with virtual environment activated):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Prepare Docker environment variables

Docker env files are located in [main/docker](main/docker).

Create env file from template if needed:

```powershell
Copy-Item main/docker/.env_example main/docker/.env -Force
```

Update values in [main/docker/.env](main/docker/.env) if required.

## 4) Start Docker services

Run from repository root:

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env up -d
```

Current stack includes:

- PostgreSQL container: `dlhpv_fresh_postgres`
- Trino container: `dlhpv_fresh_trino`

## 5) Create database tables and load CSV data

Run from repository root after Docker services are up.

Create lakehouse tables (safe to run again on existing container — uses `CREATE TABLE IF NOT EXISTS`):

```powershell
docker cp main/docker/postgres/002-create-lakehouse-tables.sql dlhpv_fresh_postgres:/tmp/002-create-lakehouse-tables.sql ; docker exec dlhpv_fresh_postgres psql -U pvlakehouse -d pvlakehouse -f /tmp/002-create-lakehouse-tables.sql
```

Load all CSV data into PostgreSQL (also runs the DDL step automatically):

```powershell
.\main\docker\scripts\load-csv-data.ps1
```

Expected output per table:

```
Copying lh_silver_clean_hourly_energy ...
  -> 153672 rows loaded into lh_silver_clean_hourly_energy
...
All CSV data loaded successfully.
```

Tables created:

- `lh_silver_clean_hourly_energy`
- `lh_silver_clean_hourly_weather`
- `lh_silver_clean_hourly_air_quality`
- `lh_gold_fact_solar_environmental`
- `lh_gold_dim_facility`

## 7) Check service status

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env ps
```

Optional health checks:

```powershell
powershell -ExecutionPolicy Bypass -File main/docker/scripts/stack-health.ps1
```

## 8) Stop services

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env down
```

To also remove volumes (delete local database data):

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env down -v
```

## 9) Common commands

Rebuild and restart services:

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env up -d --force-recreate
```

View logs:

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env logs -f
```

Deactivate virtual environment:

```powershell
deactivate
```

## Notes

- Keep real secrets only in `.env` files that are ignored by Git.
- Keep templates and documentation tracked in Git (`.env_example`, `.env.requirements.md`).

## Solar AI Chat (Vietnamese Natural Language)

Solar AI Chat endpoint is available at:

- `POST /solar-ai-chat/query`

### Request body

```json
{
	"message": "Cho toi tong quan he thong va san luong",
	"role": "viewer"
}
```

Supported role values:

- `data_engineer`
- `ml_engineer`
- `data_analyst`
- `viewer`
- `admin`

### Model routing

- Primary model: `gemini-2.5-flash-lite`
- Fallback model: `gemini-2.5-flash`

If the primary model is unavailable, the service automatically retries with the fallback model.
If both models are unavailable, the service returns a safe data-backed summary with warning metadata.

### Development environment variables

Solar AI Chat environment templates are in `dev/config`:

- `dev/config/.env_example`
- `dev/config/.env.requirements.md`

Create local runtime values with:

```powershell
Copy-Item dev/config/.env_example dev/config/.env -Force
```

## Basic Frontend Pages

FastAPI now serves a basic web UI for module navigation and chatbot testing:

- Home page with 8 module cards: `GET /`
- Solar AI Chat page: `GET /solar-ai-chat`

Run locally:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir main/backend
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/solar-ai-chat`
