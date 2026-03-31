# PV Lakehouse Docker Setup (Core)

This directory contains a minimal Docker stack for the PV Lakehouse core query layer.

## Included Services
- PostgreSQL
- Trino

## Files
- docker-compose.yml: Core service orchestration.
- .env: Local runtime variables (not for Git).
- .env_example: Committed environment template.
- .env.requirements.md: Variable documentation.
- postgres/: Initialization scripts for PostgreSQL.
- trino/catalog/: Trino catalog configuration.
- scripts/: Health check utilities.

## Quick Start
1. Copy environment template.
2. Update secrets.
3. Start services.

```bash
cd main/docker
cp .env_example .env
docker compose up -d
```

## Validate
```bash
./scripts/health-check.sh
```

## Stop
```bash
docker compose down
```