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

## 5) Check service status

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env ps
```

Optional health checks:

```powershell
powershell -ExecutionPolicy Bypass -File main/docker/scripts/stack-health.ps1
```

## 6) Stop services

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env down
```

To also remove volumes (delete local database data):

```powershell
docker compose -f main/docker/docker-compose.yml --env-file main/docker/.env down -v
```

## 7) Common commands

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
