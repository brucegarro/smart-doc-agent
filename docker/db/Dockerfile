FROM pgvector/pgvector:pg16

# Copy initialization scripts
# Files in /docker-entrypoint-initdb.d/ are executed in alphabetical order
# on first container startup (when database is empty)
COPY sql/*.sql /docker-entrypoint-initdb.d/

# These scripts run automatically when the container starts for the first time
# They are idempotent (safe to re-run) due to IF NOT EXISTS checks
