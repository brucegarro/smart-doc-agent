# Project Structure

```
smart-doc-agent/
│
├── .env                          # Environment configuration (local dev settings)
├── .env.example                  # Environment template for teammates
├── .dockerignore                 # Docker build optimization
├── .gitignore                    # Git ignore patterns
├── docker-compose.yml            # Multi-service orchestration
├── README.md                     # Project overview (to be updated)
├── QUICKSTART.md                 # Setup and usage guide
│
├── docker/                       # Docker configuration
│   ├── README.md                 # Docker documentation
│   ├── SETUP_COMPLETE.md         # Infrastructure completion summary
│   ├── app/
│   │   ├── Dockerfile            # CLI/API service image
│   │   └── requirements.txt      # Full dependencies (CLI, PDF, OCR, NLP)
│   └── worker/
│       ├── Dockerfile            # Background worker image (lean)
│       └── requirements.txt      # Minimal dependencies (no CLI/UI)
│
├── sample_papers/                # Test dataset (~30 research papers)
│   ├── download_papers.txt       # Paper identifiers
│   ├── download_all.sh           # Download script
│   └── *.pdf                     # PDF files for testing
│
├── src/                          # Application source code (TO BE CREATED)
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── cli.py                # CLI entry point (Typer)
│   │   ├── worker.py             # Background worker entry point
│   │   ├── config.py             # Settings and configuration
│   │   │
│   │   ├── db/                   # Database layer
│   │   │   ├── __init__.py
│   │   │   ├── models.py         # SQLAlchemy models (documents, chunks, etc.)
│   │   │   ├── connection.py    # Database connection management
│   │   │   └── migrations/       # Alembic migrations
│   │   │
│   │   ├── ingestion/            # Document ingestion pipeline
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py    # PDF extraction (PyMuPDF, pdfplumber)
│   │   │   ├── ocr.py            # OCR engine (PaddleOCR, Tesseract)
│   │   │   ├── udr.py            # Unified Document Representation
│   │   │   └── processor.py     # Orchestration
│   │   │
│   │   ├── embedding/            # Embedding and indexing
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py      # Sentence-transformers wrapper
│   │   │   ├── chunker.py       # Text chunking strategies
│   │   │   └── indexer.py       # Vector storage (pgvector)
│   │   │
│   │   ├── retrieval/            # RAG retrieval layer
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py     # Vector similarity search
│   │   │   └── reranker.py      # Optional re-ranking
│   │   │
│   │   ├── query/                # Query processing and generation
│   │   │   ├── __init__.py
│   │   │   ├── engine.py        # Query orchestration
│   │   │   ├── llm.py           # Ollama LLM interface
│   │   │   ├── vlm.py           # Vision-Language Model (Qwen-VL)
│   │   │   └── prompts.py       # Prompt templates
│   │   │
│   │   ├── storage/              # Object storage abstraction
│   │   │   ├── __init__.py
│   │   │   ├── s3.py            # MinIO/S3 client
│   │   │   └── artifacts.py     # Artifact management
│   │   │
│   │   ├── evaluation/           # Evaluation and metrics
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py       # F1, precision, recall, nDCG
│   │   │   ├── judge.py         # LLM-as-judge
│   │   │   └── runner.py        # Test suite runner
│   │   │
│   │   └── utils/                # Shared utilities
│   │       ├── __init__.py
│   │       ├── logging.py       # Structured logging
│   │       └── retry.py         # Retry logic with tenacity
│   │
│   └── tests/                    # Test suite (TO BE CREATED)
│       ├── __init__.py
│       ├── conftest.py           # Pytest fixtures
│       ├── test_ingestion.py
│       ├── test_embedding.py
│       ├── test_retrieval.py
│       └── test_query.py
│
└── docs/                         # Documentation (TO BE CREATED)
    ├── architecture.md           # System architecture
    ├── api.md                    # API reference
    ├── udr_schema.md             # UDR schema specification
    ├── evaluation.md             # Evaluation methodology
    └── troubleshooting.md        # Common issues and solutions
```

## Current Status

### ✅ Completed
- Docker infrastructure (all services)
- Environment configuration
- Dockerfiles with model pre-caching
- Requirements files (app + worker)
- Setup documentation

### 🔄 Next Steps
1. Create `src/` directory structure
2. Implement database models and migrations
3. Build document ingestion pipeline
4. Develop UDR schema
5. Implement embedding and indexing
6. Create CLI commands
7. Build worker job handlers
8. Develop query engine
9. Create evaluation harness

### 📦 Test Data Available
- **30 research papers** in `sample_papers/`
- Ready for ingestion pipeline testing
- Variety of formats, layouts, and content types

## Development Workflow

1. **Code in `src/`** - mounted as volume (hot reload)
2. **Run via CLI** - `docker exec -it doc_app python -m agent.cli`
3. **Background jobs** - worker processes Redis queues
4. **Evaluate** - metrics tracked in `eval_results` table

## Infrastructure Ready

All services are configured and ready to start:

```bash
docker compose up -d
```

**Next: Begin implementing the application code!** 🚀
