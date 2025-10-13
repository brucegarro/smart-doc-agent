# Unified Document Representation (UDR) Schema

## Overview

The UDR schema defines the complete structured representation of a processed research paper. It aligns with the multi-layer extraction pipeline:

1. **Text Extraction** - PyMuPDF for digital PDFs
2. **OCR Fallback** - PaddleOCR for scanned/mixed layouts
3. **Math OCR** - Qwen-VL for equation images → LaTeX conversion
4. **Structure Parser** - Hierarchical parsing: pages → blocks → spans → relations
5. **Embedding/Retrieval** - Downstream processing (BAAI/bge-small-en-v1.5)

## Hierarchical Structure

```
UnifiedDocumentRepresentation (Document)
├── metadata: DocumentMetadata
├── pages: List[Page]
│   └── Page
│       ├── page_num: int
│       ├── page_type: PageType (DIGITAL | SCANNED | MIXED)
│       ├── blocks: List[TextBlock]
│       │   └── TextBlock
│       │       ├── block_id: str
│       │       ├── block_type: str (paragraph | heading | list | caption)
│       │       ├── spans: List[TextSpan]
│       │       │   └── TextSpan
│       │       │       ├── text: str
│       │       │       ├── font_size, font_name, is_bold, is_italic
│       │       │       └── confidence: float (OCR)
│       │       ├── reading_order: int
│       │       ├── parent_block_id: Optional[str]
│       │       └── extraction_method: ExtractionMethod
│       ├── extraction_method: ExtractionMethod
│       └── ocr_applied: bool
├── sections: List[Section]
├── tables: List[Table]
├── figures: List[Figure]
├── equations: List[Equation]
└── references: List[Reference]
```

## Key Design Decisions

### 1. **Hierarchical Text Structure (Pages → Blocks → Spans)**

**Why?**
- Preserves layout information for structure-aware retrieval
- Enables semantic chunking based on blocks rather than arbitrary token windows
- Supports parent-child relationships (e.g., list items, nested sections)
- Maintains reading order for proper context

**Example:**
```python
Page(
    page_num=1,
    blocks=[
        TextBlock(
            block_id="p1_b1",
            block_type="heading",
            text="Introduction",
            spans=[
                TextSpan(text="Introduction", font_size=18.0, is_bold=True)
            ],
            reading_order=0
        ),
        TextBlock(
            block_id="p1_b2",
            block_type="paragraph",
            text="This paper presents...",
            spans=[...],
            reading_order=1,
            parent_block_id="p1_b1"  # Belongs to Introduction section
        )
    ]
)
```

### 2. **Page Type Detection**

**Three types:**
- `DIGITAL` - Text-based PDF (use PyMuPDF, skip OCR)
- `SCANNED` - Image-based PDF (use PaddleOCR)
- `MIXED` - Hybrid (selective OCR for image regions)

**Why?**
- Optimize extraction: Skip expensive OCR when not needed
- Track processing methods for quality assessment
- Enable targeted reprocessing if needed

**Detection logic** (to be implemented in parser):
```python
def detect_page_type(page) -> PageType:
    # If page has extractable text and good layout
    if has_text(page) and text_coverage > 0.7:
        return PageType.DIGITAL
    # If page is mostly images
    elif image_coverage > 0.8:
        return PageType.SCANNED
    else:
        return PageType.MIXED
```

### 3. **Extraction Method Tracking**

**Why?**
- Debug extraction quality issues
- Compare extraction methods (PyMuPDF vs. pdfplumber vs. OCR)
- Enable A/B testing of extraction strategies
- Support reprocessing with different methods

**Tracked at multiple levels:**
- Document level: `extraction_methods_used: List[ExtractionMethod]`
- Page level: `extraction_method: ExtractionMethod`
- Block level: `extraction_method: ExtractionMethod`
- Equation level: `latex_source: Optional[ExtractionMethod]`

### 4. **Enhanced Equation Support**

**New fields:**
- `equation_id` - Unique identifier
- `is_inline` - Inline vs. display equation
- `image_s3_key` - S3 path if extracted as image
- `latex_source` - How LaTeX was obtained (QWEN_VL, parsing, etc.)
- `confidence` - LaTeX conversion confidence

**Use case:**
```python
# Display equation extracted as image, converted via Qwen-VL
Equation(
    equation_id="eq_1",
    latex=r"\sum_{i=1}^{n} x_i",
    text="sum of x_i from i=1 to n",
    page=3,
    is_inline=False,
    image_s3_key="pdfs/doc_id/equations/eq_1.png",
    extraction_method=ExtractionMethod.PYMUPDF,
    latex_source=ExtractionMethod.QWEN_VL,
    confidence=0.95
)
```

### 5. **OCR Confidence Scores**

**Why?**
- Filter low-quality OCR results
- Prioritize high-confidence blocks during retrieval
- Identify pages that need manual review
- Compute document-level quality metrics

**Tracked at:**
- Span level: `TextSpan.confidence`
- Block level: `TextBlock.confidence` (average of spans)
- Page level: `Page.ocr_confidence` (average of blocks)

### 6. **Block Relationships**

**Two types:**
1. **Reading Order** - Sequential position (`reading_order: int`)
2. **Hierarchical** - Parent-child relationships (`parent_block_id: Optional[str]`)

**Why?**
- Preserve document structure for context-aware retrieval
- Enable section-aware chunking
- Support queries like "What are the subsections of Methods?"
- Improve RAG by including parent context

**Example:**
```python
# Parent heading
TextBlock(block_id="p3_b1", block_type="heading", text="3. Methods", reading_order=5)

# Child paragraph
TextBlock(
    block_id="p3_b2", 
    block_type="paragraph", 
    text="We use the following approach...",
    reading_order=6,
    parent_block_id="p3_b1"  # Links to Methods heading
)
```

### 7. **Backward Compatibility**

**Legacy field maintained:**
- `raw_page_texts: List[str]` - Simple list of page texts

**Why?**
- Support existing code during migration
- Fallback for simple queries
- Quick full-text search without parsing structure

## Data Model Classes

### Core Models

1. **ExtractionMethod** (Enum)
   - PYMUPDF, PDFPLUMBER, PADDLEOCR, QWEN_VL, HEURISTIC

2. **PageType** (Enum)
   - DIGITAL, SCANNED, MIXED

3. **BoundingBox**
   - `x0, y0, x1, y1, page`

4. **TextSpan** (Lowest level)
   - Text with font styling
   - OCR confidence
   - Superscript/subscript support

5. **TextBlock** (Mid level)
   - Contains multiple spans
   - Block type (paragraph, heading, list, etc.)
   - Reading order and parent relationships
   - Extraction metadata

6. **Page** (High level)
   - Page type detection
   - List of blocks
   - OCR metadata

### Content Models

7. **Section**
   - Title, level, text
   - Page range
   - Block IDs (references to TextBlocks)
   - Parent section ID (for nested sections)

8. **Table**
   - 2D array of cells
   - Caption, bbox
   - Extraction method and confidence

9. **Figure**
   - Caption, bbox
   - S3 key for image

10. **Equation**
    - LaTeX representation
    - Image reference (if applicable)
    - Inline vs. display
    - Conversion confidence

11. **Reference**
    - Parsed citation fields
    - DOI, URL support

### Metadata

12. **DocumentMetadata**
    - Title, authors, abstract
    - Publication info (year, venue, DOI)
    - Page counts (total, digital, scanned, mixed)

13. **UnifiedDocumentRepresentation** (Root)
    - All content organized hierarchically
    - Processing metadata (methods used, OCR pages)
    - Extra fields for future extensions

## Usage Examples

### Creating a UDR

```python
from agent.ingestion.udr import (
    UnifiedDocumentRepresentation,
    DocumentMetadata,
    Page,
    PageType,
    TextBlock,
    TextSpan,
    ExtractionMethod
)

udr = UnifiedDocumentRepresentation(
    metadata=DocumentMetadata(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N.", "..."],
        abstract="We propose a new...",
        publication_year=2017,
        venue="NeurIPS",
        num_pages=11,
        num_digital_pages=11
    ),
    pages=[
        Page(
            page_num=1,
            page_type=PageType.DIGITAL,
            width=612.0,
            height=792.0,
            text="Combined page text...",
            blocks=[
                TextBlock(
                    block_id="p1_b1",
                    block_type="heading",
                    text="Attention Is All You Need",
                    spans=[
                        TextSpan(
                            text="Attention Is All You Need",
                            font_size=18.0,
                            is_bold=True
                        )
                    ],
                    reading_order=0,
                    extraction_method=ExtractionMethod.PYMUPDF
                )
            ],
            extraction_method=ExtractionMethod.PYMUPDF,
            ocr_applied=False
        )
    ],
    extraction_methods_used=[ExtractionMethod.PYMUPDF, ExtractionMethod.PDFPLUMBER]
)

# Serialize to JSON for database storage
udr_json = udr.model_dump_json()
```

### Accessing UDR Data

```python
# Get all headings
headings = [
    block for page in udr.pages 
    for block in page.blocks 
    if block.block_type == "heading"
]

# Get pages that required OCR
ocr_pages = [page for page in udr.pages if page.ocr_applied]

# Find equations with high-confidence LaTeX
confident_equations = [
    eq for eq in udr.equations 
    if eq.confidence and eq.confidence > 0.9
]

# Get blocks in reading order
all_blocks = [
    block 
    for page in udr.pages 
    for block in sorted(page.blocks, key=lambda b: b.reading_order)
]
```

## Future Extensions

### Potential Additions

1. **Cross-references**
   - Link citations to reference entries
   - Link figure/table mentions to actual figures/tables

2. **Math symbol extraction**
   - Extract variables and their definitions
   - Build symbol glossary

3. **Code blocks**
   - Extract code snippets
   - Preserve syntax highlighting

4. **Multilingual support**
   - Language detection per block
   - Translation metadata

5. **Quality metrics**
   - Per-page quality score
   - Extraction completeness

6. **Version tracking**
   - UDR schema version
   - Reprocessing history

## Database Storage

The UDR is stored as JSONB in PostgreSQL:

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    ...
    udr_data JSONB NOT NULL,  -- Full UDR stored here
    ...
);

-- Query examples
-- Get all digital documents
SELECT id, title 
FROM documents 
WHERE udr_data->'metadata'->>'num_digital_pages' = 
      udr_data->'metadata'->>'num_pages';

-- Get documents that used OCR
SELECT id, title, jsonb_array_length(udr_data->'ocr_pages') as num_ocr_pages
FROM documents
WHERE jsonb_array_length(udr_data->'ocr_pages') > 0;
```

## Migration Path

For existing documents with old UDR schema:

1. Keep `raw_page_texts` for backward compatibility
2. Gradually migrate to hierarchical structure
3. Use feature flags to enable new features
4. Reprocess documents in background

```python
def migrate_old_udr(old_udr: dict) -> UnifiedDocumentRepresentation:
    """Convert old UDR format to new hierarchical format."""
    # Extract metadata
    metadata = DocumentMetadata(**old_udr["metadata"])
    
    # Convert flat page texts to Page objects
    pages = []
    for i, text in enumerate(old_udr.get("pages", []), start=1):
        pages.append(Page(
            page_num=i,
            page_type=PageType.DIGITAL,  # Assume digital for old docs
            text=text,
            blocks=[],  # Empty blocks, can reprocess later
            extraction_method=ExtractionMethod.PYMUPDF
        ))
    
    # Build new UDR
    return UnifiedDocumentRepresentation(
        metadata=metadata,
        pages=pages,
        raw_page_texts=old_udr.get("pages", []),  # Preserve legacy
        sections=old_udr.get("sections", []),
        tables=old_udr.get("tables", []),
        ...
    )
```
