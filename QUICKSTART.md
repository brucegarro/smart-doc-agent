# Quick Start Guide

## Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- Git
- 8GB+ RAM recommended
- 20GB+ free disk space (for models and data)

## First-Time Setup

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/brucegarro/smart-doc-agent.git
cd smart-doc-agent

# Copy and customize environment file
cp .env.example .env

# Edit .env for your hardware (optional)
# Most important: EMBEDDER_DEVICE (cpu|mps|cuda)
nano .env
```

### 2. Adjust for Your Hardware

Edit `.env` and set `EMBEDDER_DEVICE` based on your system:

**Apple Silicon (M1/M2/M3)**:
```bash
EMBEDDER_DEVICE=mps
```

**Intel/AMD (no GPU)**:
```bash
EMBEDDER_DEVICE=cpu
WORKER_CONCURRENCY=1
WORKER_CPU_LIMIT=2
WORKER_MEMORY_LIMIT=4G
```

**NVIDIA GPU**:
```bash
EMBEDDER_DEVICE=cuda
```

### 3. Build and Start Services

```bash
# Build all images (this will take 10-15 minutes first time)
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps
```

### 4. Download Ollama Models

```bash
# Enter the Ollama container
docker exec -it doc_ollama bash

# Pull the text LLM
ollama pull qwen2.5:7b-instruct-q4_K_M

# Pull the vision-language model
ollama pull qwen2-vl:7b-instruct-q4_K_M

# Exit container
exit
```

### 5. Initialize Database

```bash
# Run migrations (once the app container is ready)
docker exec -it doc_app python -m agent.db.migrate

# Verify database
docker exec -it doc_db psql -U doc -d docdb -c "\dt"
```

## Access Points

Once running, you can access:

- **Jupyter Lab**: http://localhost:8888 (no password by default)
- **MinIO Console**: http://localhost:9001 (login: minio/minio123)
- **PostgreSQL**: localhost:5432 (user: doc, password: doc, db: docdb)
- **Ollama API**: http://localhost:11434

## Usage Examples

### CLI (Interactive)

```bash
# Enter the app container
docker exec -it doc_app bash

# Process a document
python -m agent.cli ingest /data/sample.pdf

# Query documents
python -m agent.cli query "What is the conclusion of the paper?"

# List processed documents
python -m agent.cli list
```

### Jupyter Notebook

1. Open http://localhost:8888
2. Navigate to `/work/` folder
3. Create new notebook or open examples
4. Source code is available at `/home/jovyan/src/` (read-only)

### Python API (in notebook or script)

```python
from agent.ingestion import DocumentProcessor
from agent.query import QueryEngine

# Initialize
processor = DocumentProcessor()
query_engine = QueryEngine()

# Ingest a document
doc_id = processor.ingest_pdf("/data/sample.pdf")

# Query
results = query_engine.query(
    "What are the key findings?",
    top_k=5
)

for result in results:
    print(f"Page {result.page}: {result.text}")
```

## Common Commands

```bash
# View logs
docker compose logs -f app
docker compose logs -f worker

# Restart a service
docker compose restart app

# Stop all services
docker compose down

# Stop and remove volumes (CAUTION: deletes data)
docker compose down -v

# Rebuild after code changes (if not using volume mounts)
docker compose build app && docker compose up -d app

# Check resource usage
docker stats
```

## Troubleshooting

### Services not starting
```bash
# Check logs
docker compose logs

# Check individual service
docker compose logs db
docker compose logs ollama
```

### Out of memory
```bash
# Reduce worker concurrency in .env
WORKER_CONCURRENCY=1
WORKER_MEMORY_LIMIT=4G

# Restart
docker compose down && docker compose up -d
```

### Models not loading
```bash
# Check Ollama models
docker exec -it doc_ollama ollama list

# Check embedder cache
docker exec -it doc_app ls -lh /root/.cache/huggingface
```

### Database connection errors
```bash
# Wait for database to be ready
docker compose logs db | grep "ready to accept connections"

# Manually check database
docker exec -it doc_db psql -U doc -d docdb
```

## Development Workflow

1. **Code changes**: Edit files in `src/` - changes are immediately reflected (volume mount)
2. **Dependency changes**: Edit `docker/*/requirements.txt` and rebuild:
   ```bash
   docker compose build app worker
   docker compose up -d
   ```
3. **Database schema changes**: Create migration and apply:
   ```bash
   docker exec -it doc_app alembic revision --autogenerate -m "description"
   docker exec -it doc_app alembic upgrade head
   ```

## Next Steps

- Read the [Architecture Documentation](docs/architecture.md)
- Explore [Example Notebooks](notebooks/examples/)
- Check [API Reference](docs/api.md)
- Review [Evaluation Metrics](docs/evaluation.md)

## Getting Help

- Check logs: `docker compose logs -f`
- Review [Troubleshooting Guide](docs/troubleshooting.md)
- Open an issue on GitHub
