#!/bin/bash
set -euo pipefail

echo "Checking required containers..."
for service in postgres trino; do
  if docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
    echo "[PASS] ${service} is running"
  else
    echo "[FAIL] ${service} is not running"
    exit 1
  fi
done

echo "Checking PostgreSQL readiness..."
if docker exec postgres pg_isready -U "${POSTGRES_USER:-pvlakehouse}" >/dev/null 2>&1; then
  echo "[PASS] PostgreSQL is ready"
else
  echo "[FAIL] PostgreSQL is not ready"
  exit 1
fi

echo "Checking Trino catalog..."
if docker exec trino trino --execute "SHOW CATALOGS" 2>/dev/null | grep -qi "postgresql"; then
  echo "[PASS] Trino catalog 'postgresql' is available"
else
  echo "[FAIL] Trino catalog 'postgresql' is missing"
  exit 1
fi

echo "Health check passed."