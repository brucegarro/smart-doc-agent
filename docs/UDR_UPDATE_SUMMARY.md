# UDR Schema Review Summary

## What Changed

The UDR (Unified Document Representation) schema has been **significantly enhanced** to align with the revised component design:

### Before (Simple Schema)
- Flat structure: `pages: List[str]` (just raw text)
- Basic models: TextBlock, Section, Table, Figure, Equation, Reference
- No extraction method tracking
- No page type detection
- No OCR support
- No hierarchical relationships

### After (Hierarchical Schema)
- **Hierarchical structure:** Document → Pages → Blocks → Spans
- **Page type detection:** DIGITAL | SCANNED | MIXED
- **Extraction method tracking** at document, page, and block levels
- **OCR confidence scores** for quality assessment
- **Block relationships:** Reading order + parent-child hierarchy
- **Enhanced equation support:** Image references, LaTeX source tracking, confidence
- **Backward compatibility:** Legacy `raw_page_texts` field maintained

---

## New Data Models

### 1. **ExtractionMethod** (Enum)
Tracks which tool extracted each component:
- `PYMUPDF` - Primary text extraction
- `PDFPLUMBER` - Table detection
- `PADDLEOCR` - OCR for scanned pages
- `QWEN_VL` - Math OCR (equation images → LaTeX)
- `HEURISTIC` - Rule-based parsing

### 2. **PageType** (Enum)
Classifies each page:
- `DIGITAL` - Text-based PDF (skip OCR)
- `SCANNED` - Image-based PDF (requires OCR)
- `MIXED` - Hybrid content (selective OCR)

### 3. **TextSpan** (NEW)
Lowest-level text unit with consistent formatting:
```python
TextSpan(
    text="This is bold text",
    font_size=12.0,
    font_name="Arial-Bold",
    is_bold=True,
    confidence=0.98  # OCR confidence if applicable
)
```

### 4. **TextBlock** (ENHANCED)
Logical text unit (paragraph, heading, list):
```python
TextBlock(
    block_id="p1_b1",
    block_type="heading",  # paragraph | heading | list | caption
    text="Combined text from spans",
    spans=[TextSpan(...)],
    reading_order=0,  # Sequential position
    parent_block_id=None,  # Hierarchical relationship
    extraction_method=ExtractionMethod.PYMUPDF,
    confidence=0.95
)
```

### 5. **Page** (NEW)
Page-level representation with type detection:
```python
Page(
    page_num=1,
    page_type=PageType.DIGITAL,
    width=612.0,
    height=792.0,
    text="Combined text from all blocks",
    blocks=[TextBlock(...)],
    extraction_method=ExtractionMethod.PYMUPDF,
    ocr_applied=False,
    ocr_confidence=None
)
```

### 6. **Section** (ENHANCED)
Now includes block references and parent relationships:
```python
Section(
    section_id="sec_methods",
    title="Methods",
    level=1,
    text="Section content...",
    page_start=3,
    page_end=5,
    block_ids=["p3_b1", "p3_b2", "p4_b1"],  # NEW: Links to blocks
    parent_section_id=None  # NEW: For nested sections
)
```

### 7. **Equation** (ENHANCED)
Full support for image-based equations:
```python
Equation(
    equation_id="eq_1",
    latex=r"\sum_{i=1}^{n} x_i",
    text="sum of x_i from i=1 to n",
    page=3,
    is_inline=False,  # NEW: Display vs inline
    image_s3_key="pdfs/doc_id/equations/eq_1.png",  # NEW: Image reference
    extraction_method=ExtractionMethod.PYMUPDF,
    latex_source=ExtractionMethod.QWEN_VL,  # NEW: LaTeX source
    confidence=0.95  # NEW: Conversion confidence
)
```

### 8. **DocumentMetadata** (ENHANCED)
Added page type distribution:
```python
DocumentMetadata(
    title="Paper Title",
    authors=["Author 1", "Author 2"],
    num_pages=10,
    num_digital_pages=8,  # NEW
    num_scanned_pages=1,  # NEW
    num_mixed_pages=1     # NEW
)
```

### 9. **UnifiedDocumentRepresentation** (ROOT - ENHANCED)
Complete document with processing metadata:
```python
UnifiedDocumentRepresentation(
    metadata=DocumentMetadata(...),
    pages=[Page(...)],  # NEW: Hierarchical structure
    raw_page_texts=["..."],  # LEGACY: Backward compatibility
    sections=[Section(...)],
    tables=[Table(...)],
    figures=[Figure(...)],
    equations=[Equation(...)],
    references=[Reference(...)],
    extraction_methods_used=[  # NEW: Track all methods used
        ExtractionMethod.PYMUPDF,
        ExtractionMethod.PDFPLUMBER,
        ExtractionMethod.PADDLEOCR
    ],
    ocr_pages=[2, 7],  # NEW: Which pages needed OCR
    extracted_at=datetime.utcnow()
)
```

---

## Key Benefits

### 1. **Multi-Strategy Extraction**
Each component tracks its extraction method:
- **Debug quality issues:** "Was this text from PyMuPDF or OCR?"
- **Compare strategies:** "Does pdfplumber extract tables better than PyMuPDF?"
- **Enable A/B testing:** Try different extraction methods and compare results

### 2. **OCR Optimization**
Page type detection enables intelligent processing:
- **Skip unnecessary OCR:** Save time on digital PDFs
- **Target OCR precisely:** Only process scanned/mixed pages
- **Track quality:** OCR confidence scores identify low-quality extractions

### 3. **Hierarchical Structure**
Document → Pages → Blocks → Spans:
- **Semantic chunking:** Chunk by blocks (paragraphs, sections) instead of arbitrary token windows
- **Structure-aware retrieval:** Find relevant blocks with context
- **Relationship preservation:** Parent-child links maintain document structure

### 4. **Math OCR Integration**
Full support for Qwen-VL math conversion:
- **Image-based equations:** Extract equation images, convert to LaTeX
- **Confidence tracking:** Filter low-quality conversions
- **Inline vs display:** Preserve equation formatting

### 5. **Quality Assessment**
Multiple confidence metrics:
- **Span level:** Individual OCR confidence
- **Block level:** Average span confidence
- **Page level:** Average block confidence
- **Equation level:** LaTeX conversion confidence

### 6. **Reading Order Preservation**
Sequential and hierarchical relationships:
- **Reading order:** `reading_order` field maintains sequence
- **Parent-child:** `parent_block_id` links related blocks
- **Context-aware RAG:** Include parent context in retrieval

---

## Alignment with Revised Component Design

| Layer | Component | UDR Support |
|-------|-----------|-------------|
| 1. Text Extraction | PyMuPDF for digital PDFs | ✅ `ExtractionMethod.PYMUPDF`, `PageType.DIGITAL` |
| 2. OCR Fallback | PaddleOCR for scanned/mixed | ✅ `ExtractionMethod.PADDLEOCR`, `ocr_applied`, `ocr_confidence` |
| 3. Math OCR | Qwen-VL → LaTeX | ✅ `Equation.latex_source = QWEN_VL`, `image_s3_key`, `confidence` |
| 4. Structure Parser | Pages → Blocks → Spans | ✅ `Page`, `TextBlock`, `TextSpan` hierarchy with relationships |
| 5. Embedding | BAAI/bge-small-en-v1.5 | ✅ Text + LaTeX ready for joint embedding (downstream) |
| 6. Retrieval | pgvector + metadata filters | ✅ Structured blocks enable semantic chunking |
| 7. Reasoning | Qwen2.5-7B / Qwen2.5-VL | ✅ Rich context from hierarchical structure |
| 8. Evaluation | Auto-judge / metrics | ✅ Confidence scores enable quality filtering |

---

## Next Steps for PDF Parser

The PDF parser (`pdf_parser.py`) needs updates to build the new UDR structure:

### 1. **Page Type Detection**
```python
def detect_page_type(page) -> PageType:
    # Check if page has extractable text
    text = page.get_text()
    if len(text.strip()) > 100:
        return PageType.DIGITAL
    
    # Check for images
    images = page.get_images()
    if len(images) > 0:
        return PageType.SCANNED
    
    return PageType.MIXED
```

### 2. **Build Hierarchical Structure**
```python
def extract_blocks_with_spans(page) -> List[TextBlock]:
    blocks = []
    for block_num, block in enumerate(page.get_text("dict")["blocks"]):
        spans = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(TextSpan(
                    text=span["text"],
                    font_size=span.get("size"),
                    font_name=span.get("font"),
                    is_bold="Bold" in span.get("font", ""),
                    is_italic="Italic" in span.get("font", "")
                ))
        
        blocks.append(TextBlock(
            block_id=f"p{page.number}_b{block_num}",
            block_type="paragraph",  # Could be smarter
            text=" ".join([s.text for s in spans]),
            spans=spans,
            reading_order=block_num,
            extraction_method=ExtractionMethod.PYMUPDF
        ))
    
    return blocks
```

### 3. **Selective OCR**
```python
def apply_ocr_if_needed(page, page_type) -> Tuple[List[TextBlock], bool]:
    if page_type == PageType.DIGITAL:
        # Skip OCR, use PyMuPDF
        return extract_blocks_with_spans(page), False
    
    elif page_type in [PageType.SCANNED, PageType.MIXED]:
        # Apply PaddleOCR
        ocr_results = paddleocr.ocr(page_image)
        blocks = convert_ocr_to_blocks(ocr_results)
        return blocks, True
```

### 4. **Math OCR Integration**
```python
def extract_equations_with_math_ocr(page, page_num) -> List[Equation]:
    equations = []
    
    # Find equation regions (heuristic or model-based)
    eq_regions = detect_equation_regions(page)
    
    for eq_idx, region in enumerate(eq_regions):
        # Extract image
        eq_image = extract_region_image(page, region)
        
        # Upload to S3
        s3_key = f"pdfs/{doc_id}/equations/p{page_num}_eq{eq_idx}.png"
        s3_client.upload_bytes(eq_image, s3_key)
        
        # Convert to LaTeX using Qwen-VL
        latex, confidence = qwen_vl_convert(eq_image)
        
        equations.append(Equation(
            equation_id=f"eq_p{page_num}_{eq_idx}",
            latex=latex,
            text=latex_to_text_fallback(latex),
            page=page_num,
            bbox=region.bbox,
            is_inline=region.is_inline,
            image_s3_key=s3_key,
            extraction_method=ExtractionMethod.PYMUPDF,
            latex_source=ExtractionMethod.QWEN_VL,
            confidence=confidence
        ))
    
    return equations
```

### 5. **Build Complete UDR**
```python
def parse(self) -> UnifiedDocumentRepresentation:
    pages = []
    all_equations = []
    extraction_methods = set()
    ocr_pages = []
    
    for page_num, fitz_page in enumerate(self.fitz_doc, start=1):
        # Detect page type
        page_type = detect_page_type(fitz_page)
        
        # Extract blocks (with selective OCR)
        blocks, ocr_applied = apply_ocr_if_needed(fitz_page, page_type)
        
        if ocr_applied:
            ocr_pages.append(page_num)
            extraction_methods.add(ExtractionMethod.PADDLEOCR)
        else:
            extraction_methods.add(ExtractionMethod.PYMUPDF)
        
        # Extract equations with math OCR
        equations = extract_equations_with_math_ocr(fitz_page, page_num)
        all_equations.extend(equations)
        if equations:
            extraction_methods.add(ExtractionMethod.QWEN_VL)
        
        # Build Page object
        pages.append(Page(
            page_num=page_num,
            page_type=page_type,
            width=fitz_page.rect.width,
            height=fitz_page.rect.height,
            text="\n".join([b.text for b in blocks]),
            blocks=blocks,
            extraction_method=ExtractionMethod.PADDLEOCR if ocr_applied else ExtractionMethod.PYMUPDF,
            ocr_applied=ocr_applied,
            ocr_confidence=compute_avg_confidence(blocks) if ocr_applied else None
        ))
    
    # Extract tables with pdfplumber
    tables = self._extract_tables()
    extraction_methods.add(ExtractionMethod.PDFPLUMBER)
    
    # Build metadata
    metadata = self._extract_metadata()
    metadata.num_digital_pages = sum(1 for p in pages if p.page_type == PageType.DIGITAL)
    metadata.num_scanned_pages = sum(1 for p in pages if p.page_type == PageType.SCANNED)
    metadata.num_mixed_pages = sum(1 for p in pages if p.page_type == PageType.MIXED)
    
    return UnifiedDocumentRepresentation(
        metadata=metadata,
        pages=pages,
        raw_page_texts=[p.text for p in pages],  # Backward compatibility
        tables=tables,
        equations=all_equations,
        extraction_methods_used=list(extraction_methods),
        ocr_pages=ocr_pages
    )
```

---

## Documentation

Full documentation available in:
- **`docs/UDR_SCHEMA.md`** - Complete schema reference with examples
- **`src/agent/ingestion/udr.py`** - Pydantic models with docstrings

---

## Summary

✅ **UDR Schema Updated** to support multi-layer extraction pipeline  
✅ **Hierarchical Structure** (Document → Pages → Blocks → Spans)  
✅ **Page Type Detection** (DIGITAL | SCANNED | MIXED)  
✅ **Extraction Method Tracking** for debugging and comparison  
✅ **OCR Confidence Scores** for quality assessment  
✅ **Math OCR Support** (Qwen-VL integration ready)  
✅ **Block Relationships** (reading order + parent-child)  
✅ **Backward Compatible** (legacy `raw_page_texts` maintained)  
✅ **Documentation Complete** (UDR_SCHEMA.md)

**Next:** Update PDF parser to build this hierarchical structure.
