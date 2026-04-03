# Load CSV data into PostgreSQL running in Docker.
# Run from the repository root:
#   .\main\docker\scripts\load-csv-data.ps1

$container = "dlhpv_fresh_postgres"
$user      = "pvlakehouse"
$db        = "pvlakehouse"
$sqlDir    = Join-Path $PSScriptRoot "..\..\sql" | Resolve-Path

$ddlFile = Join-Path $PSScriptRoot "..\postgres\002-create-lakehouse-tables.sql" | Resolve-Path
Write-Host "Ensuring tables exist ..."
docker cp $ddlFile "${container}:/tmp/002-create-lakehouse-tables.sql"
docker exec $container psql -U $user -d $db -f /tmp/002-create-lakehouse-tables.sql | Out-Null

$tables = @(
    "lh_silver_clean_hourly_energy",
    "lh_silver_clean_hourly_weather",
    "lh_silver_clean_hourly_air_quality",
    "lh_gold_fact_solar_environmental",
    "lh_gold_dim_facility"
)

foreach ($table in $tables) {
    $csv = Join-Path $sqlDir "$table.csv"
    if (-not (Test-Path $csv)) {
        Write-Host "SKIP: $csv not found"
        continue
    }

    Write-Host "Copying $table ..."
    docker cp $csv "${container}:/tmp/${table}.csv"

    Write-Host "Loading $table ..."
    docker exec $container psql -U $user -d $db -c "TRUNCATE TABLE $table;"
    docker exec $container psql -U $user -d $db `
        -c "\COPY $table FROM '/tmp/${table}.csv' WITH (FORMAT csv, HEADER true, NULL '')"

    $count = docker exec $container psql -U $user -d $db -t -c "SELECT COUNT(*) FROM $table;"
    Write-Host "  -> $($count.Trim()) rows loaded into $table"
}

Write-Host "All CSV data loaded successfully."
