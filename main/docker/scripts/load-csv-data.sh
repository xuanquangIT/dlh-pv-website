#!/bin/bash
set -euo pipefail

# Load CSV data from main/sql/ into PostgreSQL lakehouse tables.
# Run from the repository root:
#   bash main/docker/scripts/load-csv-data.sh
#
# Requires: psql, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT

: "${POSTGRES_USER:=pvlakehouse}"
: "${POSTGRES_PASSWORD:=pvlakehouse}"
: "${POSTGRES_DB:=pvlakehouse}"
: "${POSTGRES_HOST:=localhost}"
: "${POSTGRES_PORT:=5432}"

export PGPASSWORD="$POSTGRES_PASSWORD"
PSQL_CMD="psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB"

CSV_DIR="$(cd "$(dirname "$0")/../../sql" && pwd)"

load_csv() {
    local table="$1"
    local csv_file="$2"

    if [ ! -f "$csv_file" ]; then
        echo "SKIP: $csv_file not found"
        return
    fi

    echo "Loading $table from $csv_file ..."
    $PSQL_CMD -c "TRUNCATE TABLE $table;"
    $PSQL_CMD -c "\COPY $table FROM '$csv_file' WITH (FORMAT csv, HEADER true, NULL '')"
    local count
    count=$($PSQL_CMD -t -c "SELECT COUNT(*) FROM $table;")
    echo "  -> $count rows loaded into $table"
}

load_csv "lh_silver_clean_hourly_energy"      "$CSV_DIR/lh_silver_clean_hourly_energy.csv"
load_csv "lh_silver_clean_hourly_weather"      "$CSV_DIR/lh_silver_clean_hourly_weather.csv"
load_csv "lh_silver_clean_hourly_air_quality"  "$CSV_DIR/lh_silver_clean_hourly_air_quality.csv"
load_csv "lh_gold_fact_solar_environmental"    "$CSV_DIR/lh_gold_fact_solar_environmental.csv"
load_csv "lh_gold_dim_facility"                "$CSV_DIR/lh_gold_dim_facility.csv"

echo "All CSV data loaded successfully."
