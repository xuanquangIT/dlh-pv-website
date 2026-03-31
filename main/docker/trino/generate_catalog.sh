#!/bin/bash
set -e

# Generate Trino catalog configuration from template
echo "Generating Trino PostgreSQL catalog configuration..."

envsubst < /etc/trino/catalog/postgresql.properties.template > /etc/trino/catalog/postgresql.properties

echo "Trino catalog configuration generated!"
cat /etc/trino/catalog/postgresql.properties
