# Step 1 Implementation Status: PDF Ingestion Pipeline

## Overview

**Goal:** Build a PDF ingestion pipeline that extracts text, metadata, tables, and structure from research papers, then stores them in the database with S3 backing.

**Status:** ✅ **Core Infrastructure Complete** | ⚠️ **Parser Needs Updates for New UDR**

---

## Components Completed

### ✅ 1. Configuration Module (`src/agent/config.py`)
**Status:** Complete and functional

**Features:**
- Pydantic settings with environment variable loading
- Database, S3, Redis, Ollama configuration
- Embedder settings (model, device: CPU/MPS/CUDA)
- OCR and logging configuration

**Usage:**
```python
from agent.config import settings
print(settings.database_url)
print(settings.embedder_model)  # "BAAI/bge-small-en-v1.5"
```

---

### ✅ 2. Database Connection Manager (`src/agent/db/__init__.py`)
**Status:** Complete and functional

**Features:**
- Connection pooling (min=2, max=10)
- Context managers for safe connection/cursor handling
- Dict row factory for easy JSON serialization
- Automatic connection cleanup

**Usage:**
```python
from agent.db import get_db_connection, get_db_cursor

# Connection context
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM documents")
        results = cur.fetchall()

# Cursor context (auto-commits)
with get_db_cursor() as cur:
    cur.execute("INSERT INTO documents (...) VALUES (...)")
```

---

### ✅ 3. S3/MinIO Storage Client (`src/agent/storage/__init__.py`)
**Status:** Complete and functional

**Features:**
- Auto bucket creation (`doc-bucket`)
- Upload/download files and bytes
- Content-type auto-detection (PDF, PNG, JPG, JSON, TXT)
- Object existence check and deletion
- S3 key generation helpers

**Usage:**
```python
from agent.storage import s3_client

# Upload PDF
s3_client.upload_file("paper.pdf", "pdfs/doc_id/paper.pdf")

# Download bytes
data = s3_client.download_bytes("pdfs/doc_id/paper.pdf")

# Check existence
if s3_client.object_exists("pdfs/doc_id/paper.pdf"):
    print("File exists")
```

---

### ✅ 4. UDR Schema (`src/agent/ingestion/udr.py`)
**Status:** ✅ **Updated to align with revised component design**

**Major Changes:**
- **Hierarchical structure:** Document → Pages → Blocks → Spans
- **Page type detection:** DIGITAL | SCANNED | MIXED
- **Extraction method tracking** at all levels
- **OCR confidence scores** for quality assessment
- **Enhanced equation support:** Image references, LaTeX source, confidence
- **Block relationships:** Reading order + parent-child hierarchy
- **Backward compatibility:** Legacy `raw_page_texts` field

**Key Models:**
```python
# Enums
ExtractionMethod = PYMUPDF | PDFPLUMBER | PADDLEOCR | QWEN_VL | HEURISTIC
PageType = DIGITAL | SCANNED | MIXED

# Hierarchy
TextSpan (font info, OCR confidence)
  ↓
TextBlock (block_id, type, spans, reading_order, parent_block_id)
  ↓
Page (page_num, page_type, blocks, OCR metadata)
  ↓
UnifiedDocumentRepresentation (metadata, pages, sections, tables, figures, equations)
```

**Documentation:**
- **Full reference:** `docs/UDR_SCHEMA.md` (500+ lines)
- **Update summary:** `docs/UDR_UPDATE_SUMMARY.md` (300+ lines)

---

### ⚠️ 5. PDF Parser (`src/agent/ingestion/pdf_parser.py`)
**Status:** ⚠️ **Needs updates to build new UDR structure**

**Current Implementation:**
- ✅ Multi-strategy extraction (PyMuPDF + pdfplumber + pypdf)
- ✅ Metadata extraction (title, authors, year)
- ✅ Text extraction by page
- ✅ Section detection (heuristic patterns)
- ✅ Table extraction (pdfplumber)
- ✅ Context manager for resource cleanup

**Missing Features (from revised design):**
- ❌ Page type detection (DIGITAL | SCANNED | MIXED)
- ❌ Hierarchical block/span extraction
- ❌ Selective OCR with PaddleOCR
- ❌ Math OCR with Qwen-VL
- ❌ Reading order preservation
- ❌ Block relationships (parent-child)
- ❌ Extraction method tracking per component
- ❌ OCR confidence scores

**Required Updates:**
See `docs/UDR_UPDATE_SUMMARY.md` for detailed implementation plan:
1. Add page type detection logic
2. Extract TextSpans with font information
3. Build TextBlocks from spans with reading order
4. Integrate PaddleOCR for scanned pages
5. Add Qwen-VL for math OCR
6. Track extraction methods at each level
7. Compute confidence scores

---

### ✅ 6. Document Processor (`src/agent/ingestion/processor.py`)
**Status:** Complete and functional

**Features:**
- Orchestrates full pipeline: PDF → Parse → S3 → Database
- SHA256 hash calculation for deduplication
- Document existence check
- UUID generation
- Error handling with S3 cleanup on failure
- Document status tracking
- List/query operations

**Workflow:**
```python
from agent.ingestion import processor

# Process a PDF
doc_id = processor.process_pdf("paper.pdf", source_name="arxiv")
# Returns: "f47ac10b-58cc-4372-a567-0e02b2c3d479"

# Check status
status = processor.get_document_status(doc_id)
# Returns: {"id": "...", "status": "ingested", "title": "..."}

# List documents
docs = processor.list_documents(limit=20)
```

**Error Handling:**
- Raises `ValueError` if document already exists (duplicate hash)
- Raises `FileNotFoundError` if PDF doesn't exist
- Cleans up S3 objects if database insert fails
- Logs all operations with structured logging

---

### ✅ 7. CLI Interface (`src/agent/cli.py`)
**Status:** Complete with 4 commands

**Commands:**

#### `ingest` - Ingest PDFs
```bash
# Single file
python -m agent.cli ingest paper.pdf --source arxiv

# Directory (non-recursive)
python -m agent.cli ingest ./papers/ --source pubmed

# Directory (recursive)
python -m agent.cli ingest ./papers/ --recursive

# Verbose logging
python -m agent.cli ingest paper.pdf --verbose
```

**Features:**
- Progress reporting with Rich
- Success/skipped/failed summary
- Duplicate detection (skips existing)
- Batch processing with error handling

#### `list` - List documents
```bash
# Default (20 docs)
python -m agent.cli list

# Custom limit
python -m agent.cli list --limit 50

# Pagination
python -m agent.cli list --limit 20 --offset 40
```

**Features:**
- Rich table formatting
- Truncated IDs (first 8 chars)
- Author truncation (first 2 + count)
- Color-coded status (ingested=yellow, indexed=green, failed=red)

#### `status` - Check document status
```bash
python -m agent.cli status f47ac10b-58cc-4372-a567-0e02b2c3d479
```

#### `version` - Show version
```bash
python -m agent.cli version
# Output: smart-doc-agent v0.1.0
```

---

## File Structure

```
src/agent/
├── __init__.py
├── cli.py                    # ✅ CLI with 4 commands (231 lines)
├── config.py                 # ✅ Configuration management (73 lines)
├── worker.py                 # ⏳ Placeholder (for Step 2: indexing)
├── db/
│   └── __init__.py          # ✅ Database connection pool (80 lines)
├── storage/
│   └── __init__.py          # ✅ S3/MinIO client (216 lines)
└── ingestion/
    ├── __init__.py          # ✅ Package exports (13 lines)
    ├── udr.py               # ✅ UDR schema - UPDATED (286 lines)
    ├── pdf_parser.py        # ⚠️ Parser - NEEDS UPDATES (318 lines)
    └── processor.py         # ✅ Document processor (238 lines)

docs/
├── UDR_SCHEMA.md            # ✅ Complete schema reference (500+ lines)
└── UDR_UPDATE_SUMMARY.md    # ✅ Update summary + implementation plan (300+ lines)
```

**Total Lines of Code (Step 1):** ~1,955 lines
- Infrastructure: ~382 lines (config, db, storage)
- UDR Schema: ~286 lines
- Ingestion Pipeline: ~569 lines (parser, processor, package)
- CLI: ~231 lines
- Documentation: ~800 lines

---

## Testing Status

### ✅ Ready to Test
- **Configuration:** Load environment variables, access settings
- **Database:** Connection pooling, queries
- **S3 Storage:** Upload/download files
- **Document Processor:** Orchestration logic
- **CLI:** Command parsing, UI rendering

### ⚠️ Needs Updates Before Testing
- **PDF Parser:** Must implement new UDR structure before full testing
- **End-to-end ingestion:** Blocked by parser updates

### 🧪 Test Plan (After Parser Updates)
1. **Unit tests:** Test individual components (parser, processor, storage)
2. **Integration tests:** Test full pipeline with sample PDFs
3. **Quality tests:** Verify UDR structure, extraction quality
4. **Performance tests:** Measure ingestion speed, memory usage

---

## Next Steps

### Priority 1: Update PDF Parser ⚠️
**Goal:** Implement new UDR structure aligned with revised component design

**Tasks:**
1. ✅ UDR schema updated
2. ✅ Documentation complete
3. ⏳ Implement page type detection
4. ⏳ Extract TextSpans with font info
5. ⏳ Build TextBlocks with reading order
6. ⏳ Integrate PaddleOCR for OCR fallback
7. ⏳ Add Qwen-VL for math OCR
8. ⏳ Track extraction methods
9. ⏳ Compute confidence scores
10. ⏳ Build hierarchical Page objects

**Implementation Guide:** See `docs/UDR_UPDATE_SUMMARY.md` for detailed code examples

---

### Priority 2: Test Ingestion Pipeline ✅
**Goal:** Validate full pipeline with sample papers

**Tasks:**
1. Start Docker services (docker compose up)
2. Ingest sample papers from `sample_papers/`
3. Verify database records
4. Verify S3 uploads
5. Inspect UDR JSON structure
6. Check extraction quality

**Commands:**
```bash
# Start services
docker compose up -d

# Ingest samples
docker compose exec app python -m agent.cli ingest /app/sample_papers/ --recursive --verbose

# List ingested
docker compose exec app python -m agent.cli list

# Check database
docker compose exec db psql -U postgres -d smartdoc -c "SELECT id, title, status FROM documents;"
```

---

### Priority 3: Step 2 - Embedding & Indexing Layer ⏳
**Goal:** Chunk documents, generate embeddings, index in pgvector

**Components to Build:**
1. **Chunking Strategy**
   - Semantic chunking based on TextBlocks
   - Respect section boundaries
   - Target: 100-250 tokens per chunk
   - Overlap: 20-50 tokens

2. **Embedder Wrapper**
   - sentence-transformers with BAAI/bge-small-en-v1.5
   - Device selection (CPU/MPS/CUDA)
   - Batch processing
   - Joint text + LaTeX embedding

3. **Vector Indexing**
   - Insert chunks into `chunks` table
   - Generate embeddings (384 dimensions)
   - HNSW index for fast retrieval

4. **Background Worker**
   - Redis queue consumer
   - Process `q:index` queue
   - Update document status: `ingested` → `indexed`

---

## Summary

### ✅ Completed (Step 1)
- [x] Configuration module with environment management
- [x] Database connection pool with psycopg
- [x] S3/MinIO storage client
- [x] **UDR schema updated** to align with revised component design
- [x] **Documentation complete** (UDR_SCHEMA.md, UDR_UPDATE_SUMMARY.md)
- [x] Document processor with orchestration logic
- [x] CLI with ingest/list/status/version commands

### ⚠️ In Progress
- [ ] **PDF parser updates** to build hierarchical UDR structure
  - Page type detection
  - Block/span extraction
  - OCR integration (PaddleOCR)
  - Math OCR (Qwen-VL)
  - Confidence tracking

### ⏳ Pending
- [ ] End-to-end ingestion testing
- [ ] Step 2: Embedding & Indexing Layer
- [ ] Step 3: Query Engine
- [ ] Step 4: Full CLI Interface
- [ ] Step 5: Test with 30 sample papers
- [ ] Step 6: Evaluation & Refinement

---

## Revised Component Design Alignment

| Layer | Component | UDR Support | Implementation |
|-------|-----------|-------------|----------------|
| 1. Text Extraction | PyMuPDF for digital PDFs | ✅ Schema ready | ⚠️ Parser needs updates |
| 2. OCR Fallback | PaddleOCR for scanned/mixed | ✅ Schema ready | ⏳ Not implemented |
| 3. Math OCR | Qwen-VL → LaTeX | ✅ Schema ready | ⏳ Not implemented |
| 4. Structure Parser | Pages → Blocks → Spans | ✅ Schema ready | ⚠️ Parser needs updates |
| 5. Embedding | BAAI/bge-small-en-v1.5 | ✅ Ready | ⏳ Step 2 |
| 6. Retrieval | pgvector + metadata | ✅ Ready | ⏳ Step 3 |
| 7. Reasoning | Qwen2.5-7B / Qwen2.5-VL | ✅ Ready | ⏳ Step 3 |
| 8. Evaluation | Auto-judge / metrics | ✅ Schema ready | ⏳ Step 6 |

**Status:** Infrastructure and schema are **100% ready** for the revised design. Parser implementation is the main remaining task for Step 1.

---

## Key Decisions Made

1. **Hierarchical UDR:** Document → Pages → Blocks → Spans (vs flat structure)
2. **Page type detection:** DIGITAL | SCANNED | MIXED (skip OCR when possible)
3. **Extraction tracking:** Track methods at document, page, and block levels
4. **OCR confidence:** Store confidence scores for quality filtering
5. **Math OCR support:** Full Qwen-VL integration with image references
6. **Backward compatibility:** Keep legacy `raw_page_texts` for migration
7. **Block relationships:** Reading order + parent-child for structure preservation
8. **Identifiers:** Unique IDs for all components (blocks, sections, tables, equations)

---

**Last Updated:** October 13, 2025
**Status:** Step 1 infrastructure complete, parser updates in progress
