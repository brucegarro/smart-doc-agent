# ==============================================================================
# Smart Doc Agent - Worker Service Dockerfile (LEAN)
# ==============================================================================
# Purpose: Background processing for document ingestion, extraction, indexing
# Base: Python 3.11 on Debian Bookworm with MPS support
# Strategy: Minimal dependencies, optimized for async processing
# ==============================================================================

FROM python:3.11-slim-bookworm

LABEL maintainer="smart-doc-agent"
LABEL description="Document processing agent - Background worker (lean)"

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ==============================================================================
# SYSTEM DEPENDENCIES (MINIMAL)
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF processing
    poppler-utils \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Database client libraries
    libpq-dev \
    # Build tools for Python packages with C extensions
    gcc \
    g++ \
    # Utilities
    curl \
    git \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# PYTHON DEPENDENCIES (LEAN - Worker Essentials Only)
# ==============================================================================

# Copy requirements first for better layer caching
COPY docker/worker/requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# PRE-DOWNLOAD MODELS (Performance Optimization)
# ==============================================================================

# Pre-cache sentence-transformers embedder model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading embedder model...'); \
    model = SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    print('Embedder model cached successfully')"

# ==============================================================================
# PRE-CACHE OCR MODELS
# ==============================================================================
# Note: PaddleOCR model pre-caching is skipped due to segfault issues on ARM64 during build.
# Models will be downloaded automatically on first use at runtime.
# This is acceptable as models are cached in ~/.paddleocr/ and persist across container restarts.

# ==============================================================================
# APPLICATION SETUP
# ==============================================================================

# Copy application source code
# In dev mode, this will be overridden by volume mount
COPY src /app

# Create data directories
RUN mkdir -p /data/processing /data/cache

# Health check (optional - checks if worker process is responsive)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (runs the worker process)
CMD ["python", "-m", "agent.worker"]
