# Database Initialization

## Automatic Schema Setup

The PostgreSQL database is automatically initialized with the complete schema when first started. This uses PostgreSQL's built-in initialization mechanism via `docker-entrypoint-initdb.d/`.

### How It Works

1. **Custom Docker Image**: We build a custom image based on `pgvector/pgvector:pg16` that includes our SQL migration files.

2. **Automatic Execution**: On first container startup (when the database is empty), PostgreSQL automatically executes all `.sql` files in `/docker-entrypoint-initdb.d/` in alphabetical order.

3. **Idempotent**: All SQL files use `IF NOT EXISTS` clauses, making them safe to re-run. However, they only run once automatically on first startup.

### What Gets Created

**Extensions:**
- `vector` (v0.8.1) - pgvector for similarity search
- `uuid-ossp` (v1.1) - UUID generation

**Tables:**
- `documents` - PDF metadata, processing status, UDR JSON
- `chunks` - Text segments with 384-dimensional embeddings
- `artifacts_index` - Tables, figures, equations extracted from papers
- `eval_results` - Metrics (accuracy, F1-score, etc.) for comparison queries

**Indexes:**
- HNSW vector index on `chunks.embedding` for fast similarity search
- Full-text search indexes on content fields
- B-tree indexes for efficient filtering and joins

**Views:**
- `document_stats` - Overview of document processing
- `metric_comparison` - Cross-paper metric comparison

## Usage

### First Time Setup

```bash
# Build all services (including custom DB image)
docker compose build

# Start services (DB will auto-initialize)
docker compose up -d
```

The database will be ready with all tables, indexes, and extensions configured automatically.

### Resetting the Database

If you need to start fresh:

```bash
# Stop and remove the database container and volume
docker compose stop db
docker compose rm -f db
docker volume rm smart-doc-agent_pgdata

# Restart - will auto-initialize again
docker compose up -d db
```

### Manual Schema Updates

If you need to run additional SQL after initial setup:

```bash
# Execute SQL file
docker exec -i doc_db psql -U doc -d docdb < new_migration.sql

# Or run SQL directly
docker exec doc_db psql -U doc -d docdb -c "SELECT * FROM documents;"
```

## Files

- `docker/db/Dockerfile` - Custom PostgreSQL image definition
- `sql/01_init.sql` - Extensions setup
- `sql/02_documents.sql` - Documents table
- `sql/03_chunks.sql` - Chunks table with vector embeddings
- `sql/04_artifacts.sql` - Artifacts index table
- `sql/05_eval_results.sql` - Evaluation results table
- `sql/06_views.sql` - Helpful views

## Verification

Check that everything was created:

```bash
# List tables
docker exec doc_db psql -U doc -d docdb -c "\dt"

# Check extensions
docker exec doc_db psql -U doc -d docdb -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp');"

# Verify vector index
docker exec doc_db psql -U doc -d docdb -c "\d chunks"

# List views
docker exec doc_db psql -U doc -d docdb -c "\dv"
```

## Notes

- The initialization scripts run **only once** when the volume is first created
- Subsequent container restarts do **not** re-run the scripts
- All migrations are **idempotent** (safe to run multiple times)
- If you need to modify the schema, either:
  - Delete the volume and recreate (loses all data)
  - Run manual migrations on the existing database
