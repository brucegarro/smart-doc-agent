#!/bin/bash
# Initialize database by running all SQL files in order

set -e

echo "=========================================="
echo "Smart Doc Agent - Database Initialization"
echo "=========================================="

# Database connection settings
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-doc}
DB_PASSWORD=${DB_PASSWORD:-doc}
DB_NAME=${DB_NAME:-docdb}

export PGPASSWORD=$DB_PASSWORD

echo ""
echo "Connecting to PostgreSQL..."
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo ""

# Check if SQL directory exists
SQL_DIR="$(dirname "$0")/../sql"
if [ ! -d "$SQL_DIR" ]; then
    echo "Error: SQL directory not found at $SQL_DIR"
    exit 1
fi

# Run each SQL file in order
echo "Running SQL migrations:"
for sql_file in "$SQL_DIR"/*.sql; do
    if [ -f "$sql_file" ]; then
        filename=$(basename "$sql_file")
        echo "  Running $filename..."
        
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" > /dev/null 2>&1; then
            echo "  ✓ $filename completed"
        else
            echo "  ✗ Error in $filename"
            exit 1
        fi
    fi
done

echo ""
echo "Verifying installation:"

# Check pgvector extension
echo "  Checking pgvector extension..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
    "SELECT '  ✓ pgvector v' || extversion FROM pg_extension WHERE extname = 'vector';"

# List tables
echo "  Checking tables..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
    "SELECT '    - ' || tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"

# List views
echo "  Checking views..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
    "SELECT '    - ' || viewname FROM pg_views WHERE schemaname = 'public' ORDER BY viewname;"

echo ""
echo "=========================================="
echo "✓ Database initialization completed!"
echo "=========================================="
