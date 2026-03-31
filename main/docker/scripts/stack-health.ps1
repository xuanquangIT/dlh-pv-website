#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"
$failureCount = 0

Write-Host "Checking required containers..." -ForegroundColor Yellow
$containers = docker ps --format "{{.Names}}"

foreach ($service in @("postgres", "trino")) {
    if ($containers -contains $service) {
        Write-Host "[PASS] $service is running" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $service is not running" -ForegroundColor Red
        $failureCount++
    }
}

Write-Host "Checking PostgreSQL readiness..." -ForegroundColor Yellow
try {
    docker exec postgres pg_isready -U pvlakehouse | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[PASS] PostgreSQL is ready" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] PostgreSQL is not ready" -ForegroundColor Red
        $failureCount++
    }
} catch {
    Write-Host "[FAIL] PostgreSQL readiness check failed" -ForegroundColor Red
    $failureCount++
}

Write-Host "Checking Trino catalog..." -ForegroundColor Yellow
try {
    $catalogs = docker exec trino trino --execute "SHOW CATALOGS" 2>$null
    if ($catalogs -match "postgresql") {
        Write-Host "[PASS] Trino catalog 'postgresql' is available" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Trino catalog 'postgresql' is missing" -ForegroundColor Red
        $failureCount++
    }
} catch {
    Write-Host "[FAIL] Trino catalog check failed" -ForegroundColor Red
    $failureCount++
}

if ($failureCount -eq 0) {
    Write-Host "Health check passed." -ForegroundColor Green
    exit 0
}

Write-Host "Health check failed with $failureCount issue(s)." -ForegroundColor Red
exit 1