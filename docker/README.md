# Docker Configuration

This directory contains the Docker configuration for the Smart Doc Agent services.

## Services

### App Service (`docker/app/`)
**Purpose**: CLI interface and future API server for document processing

**Key Features**:
- Full-featured CLI with Typer and Rich
- PDF processing (PyMuPDF, pdfplumber, pypdf)
- OCR support (Tesseract + PaddleOCR)
- Embeddings (sentence-transformers with MPS support)
- Future API endpoints

**Image Details**:
- Base: `python:3.11-slim-bookworm`
- Torch: MPS support for Apple Silicon
- Pre-cached models: BAAI/bge-small-en-v1.5, PaddleOCR English

### Worker Service (`docker/worker/`)
**Purpose**: Background processing for document ingestion, extraction, and indexing

**Key Features** (LEAN):
- Async job processing via Redis Queue (RQ)
- PDF extraction and OCR
- Embeddings generation
- Database indexing
- Minimal dependencies (no CLI/API frameworks)

**Image Details**:
- Base: `python:3.11-slim-bookworm`
- Torch: MPS support for Apple Silicon
- Pre-cached models: BAAI/bge-small-en-v1.5, PaddleOCR English
- CPU-optimized PaddleOCR

## Hardware Configuration

The services support multiple compute backends via the `EMBEDDER_DEVICE` environment variable:

### Apple Silicon (M1/M2/M3)
```bash
EMBEDDER_DEVICE=mps
WORKER_CPU_LIMIT=4
WORKER_MEMORY_LIMIT=8G
```

### Intel/AMD CPU (No GPU)
```bash
EMBEDDER_DEVICE=cpu
WORKER_CPU_LIMIT=2
WORKER_MEMORY_LIMIT=4G
```

### NVIDIA GPU
```bash
EMBEDDER_DEVICE=cuda
WORKER_CPU_LIMIT=8
WORKER_MEMORY_LIMIT=16G
# Note: Requires nvidia-docker runtime
```

## Building Images

### Build all services
```bash
docker compose build
```

### Build specific service
```bash
docker compose build app
docker compose build worker
```

### Build with no cache (force rebuild)
```bash
docker compose build --no-cache
```

## Model Pre-caching

Both Dockerfiles pre-download models during the build process to improve startup performance:

1. **Sentence Transformers**: `BAAI/bge-small-en-v1.5` (~133MB)
2. **PaddleOCR**: English detection and recognition models (~20MB)

Models are cached in:
- HuggingFace: `/root/.cache/huggingface` (shared volume)
- PaddleOCR: `/root/.paddleocr` (shared volume)

## Development Mode

In development, source code is mounted as a volume for hot reloading:

```yaml
volumes:
  - ./src:/app:cached
```

This means:
- ✅ Code changes are immediately reflected
- ✅ No rebuild needed for code updates
- ❌ Dependency changes still require rebuild

## Production Considerations

For production deployment:

1. **Remove volume mounts**: Copy code into image instead
2. **Use secrets management**: Don't use default credentials
3. **Enable Redis persistence**: Switch from in-memory to AOF/RDB
4. **Add authentication**: Enable Jupyter token, add API auth
5. **Resource limits**: Adjust CPU/memory based on workload
6. **Multi-stage builds**: Consider adding build/runtime stages
7. **Security scanning**: Run `docker scan` before deployment

## Troubleshooting

### Models not downloading
If models fail to download during build:
```bash
# Check network connectivity in build
docker compose build --progress=plain app
```

### Out of memory during build
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or build without model pre-caching and download at runtime
```

### MPS device not found
```bash
# Fallback to CPU if MPS not available
EMBEDDER_DEVICE=cpu docker compose up
```

### Layer caching issues
```bash
# Clear build cache
docker builder prune -a
```
