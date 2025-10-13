# ==============================================================================
# Smart Doc Agent - Application Service Dockerfile
# ==============================================================================
# Purpose: CLI interface and future API server for document processing
# Base: Python 3.11 on Debian Bookworm with MPS support for Apple Silicon
# ==============================================================================

FROM python:3.11-slim-bookworm

LABEL maintainer="smart-doc-agent"
LABEL description="Document processing agent - CLI/API service"

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ==============================================================================
# SYSTEM DEPENDENCIES
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
# PYTHON DEPENDENCIES
# ==============================================================================

# Copy requirements first for better layer caching
COPY docker/app/requirements.txt /tmp/requirements.txt

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
COPY scripts /app/scripts

# Create data directories
RUN mkdir -p /data/uploads /data/exports /data/cache

# Expose port for future API server
EXPOSE 8080

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "agent.cli"]
