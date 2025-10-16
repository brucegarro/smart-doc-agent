"""Unified Document Representation (UDR) schema using Pydantic.

Based on revised component design:
- Layer 1: Text Extraction (PyMuPDF for digital PDFs)
- Layer 2: OCR fallback (PaddleOCR for scanned/mixed)
- Layer 3: Math OCR (Qwen-VL for equation images → LaTeX)
- Layer 4: Structure Parser (pages → blocks → spans → relations)
- Layer 5+: Embedding, Retrieval, Reasoning (downstream)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    """Method used to extract content."""
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    PADDLEOCR = "paddleocr"
    QWEN_VL = "qwen-vl"
    HEURISTIC = "heuristic"


class PageType(str, Enum):
    """Type of page content."""
    DIGITAL = "digital"  # Text-based PDF
    SCANNED = "scanned"  # Image-based PDF requiring OCR
    MIXED = "mixed"      # Hybrid (text + images)


class BoundingBox(BaseModel):
    """Bounding box coordinates for layout elements."""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class TextSpan(BaseModel):
    """
    Lowest-level text unit with styling information.
    
    A span is a contiguous piece of text with consistent formatting.
    Multiple spans make up a block.
    """
    text: str
    bbox: Optional[BoundingBox] = None
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    is_superscript: bool = False
    is_subscript: bool = False
    confidence: Optional[float] = None  # OCR confidence (0-1)


class TextBlock(BaseModel):
    """
    A logical block of text (paragraph, heading, list item, etc.).
    
    Blocks contain one or more spans and have layout/semantic information.
    """
    block_id: str  # Unique identifier within document
    block_type: str = "paragraph"  # paragraph, heading, list, caption, etc.
    text: str  # Combined text from all spans
    spans: List[TextSpan] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    reading_order: int = 0  # Position in reading sequence
    parent_block_id: Optional[str] = None  # For hierarchical structures
    
    # Extraction metadata
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    confidence: Optional[float] = None  # Overall confidence for this block


class Page(BaseModel):
    """
    Page-level representation with type detection and blocks.
    
    Structure: Document → Pages → Blocks → Spans
    """
    page_num: int
    page_type: PageType = PageType.DIGITAL
    width: float
    height: float
    
    # Text content
    text: str  # Combined text from all blocks
    blocks: List[TextBlock] = Field(default_factory=list)
    
    # Extraction metadata
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    ocr_applied: bool = False
    ocr_confidence: Optional[float] = None  # Average OCR confidence if applied


class Section(BaseModel):
    """A document section (abstract, introduction, methods, etc.)."""
    section_id: str  # Unique identifier
    title: str
    level: int  # Heading level (1=h1, 2=h2, etc.)
    text: str  # Combined text from all blocks
    page_start: int
    page_end: int
    block_ids: List[str] = Field(default_factory=list)  # References to TextBlock IDs
    parent_section_id: Optional[str] = None  # For nested sections


class Table(BaseModel):
    """Extracted table structure."""
    table_id: str  # Unique identifier
    caption: Optional[str] = None
    page: int
    bbox: Optional[BoundingBox] = None
    data: List[List[str]] = Field(default_factory=list)  # 2D array of cells
    text: str  # Fallback text representation
    subtype: Optional[str] = None  # e.g., "data", "layout", "matrix"
    confidence: Optional[float] = None
    caption_id: Optional[str] = None
    header_rows: Optional[int] = None
    header_cols: Optional[int] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)  # e.g., image_uri, csv_uri
    
    # Extraction metadata
    extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER


class Figure(BaseModel):
    """Figure or diagram reference."""
    figure_id: str  # Unique identifier
    caption: Optional[str] = None
    page: int
    bbox: Optional[BoundingBox] = None
    s3_key: Optional[str] = None  # S3 path to extracted image
    subtype: Optional[str] = None  # photo | chart | diagram | vector
    confidence: Optional[float] = None
    caption_id: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    
    # Extraction metadata
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF


class Equation(BaseModel):
    """
    Mathematical equation with LaTeX conversion.
    
    Supports both inline and display equations.
    If image-based, Qwen-VL is used to convert to LaTeX.
    """
    equation_id: str  # Unique identifier
    latex: Optional[str] = None  # LaTeX representation (from Qwen-VL or parsing)
    text: str  # Plain text fallback
    page: int
    bbox: Optional[BoundingBox] = None
    is_inline: bool = False  # True for inline equations, False for display
    
    # Image-based equation metadata
    image_s3_key: Optional[str] = None  # S3 path if extracted as image
    
    # Extraction metadata
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    latex_source: Optional[ExtractionMethod] = None  # How LaTeX was obtained (QWEN_VL, etc.)
    confidence: Optional[float] = None  # LaTeX conversion confidence


class Reference(BaseModel):
    """Bibliography reference."""
    reference_id: str  # Unique identifier
    text: str  # Full reference text
    authors: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    publication_year: Optional[int] = None
    venue: Optional[str] = None  # Journal/Conference
    doi: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    num_pages: int
    num_words: Optional[int] = None
    
    # Page type distribution
    num_digital_pages: int = 0
    num_scanned_pages: int = 0
    num_mixed_pages: int = 0


class UnifiedDocumentRepresentation(BaseModel):
    """
    Unified Document Representation (UDR) - Complete structured document.
    
    Hierarchical structure: Document → Pages → Blocks → Spans
    
    This is the canonical representation of a processed document,
    stored as JSONB in the database.
    
    Design aligns with multi-layer extraction pipeline:
    1. Text Extraction (PyMuPDF for digital PDFs)
    2. OCR fallback (PaddleOCR for scanned/mixed)
    3. Math OCR (Qwen-VL for equation images → LaTeX)
    4. Structure Parser (pages → blocks → spans → relations)
    """
    
    # Metadata
    metadata: DocumentMetadata
    
    # Hierarchical content structure
    pages: List[Page] = Field(default_factory=list)  # Page objects with blocks
    
    # Legacy: Raw text per page (kept for backward compatibility)
    raw_page_texts: List[str] = Field(default_factory=list)
    
    # Structured content
    sections: List[Section] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    equations: List[Equation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    
    # Processing metadata
    extraction_methods_used: List[ExtractionMethod] = Field(default_factory=list)
    ocr_pages: List[int] = Field(default_factory=list)  # Pages where OCR was applied
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional data
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani et al."],
                    "abstract": "We propose a new...",
                    "publication_year": 2017,
                    "venue": "NeurIPS",
                    "num_pages": 11,
                    "num_digital_pages": 11,
                    "num_scanned_pages": 0
                },
                "pages": [
                    {
                        "page_num": 1,
                        "page_type": "digital",
                        "width": 612.0,
                        "height": 792.0,
                        "text": "Full page text...",
                        "blocks": [
                            {
                                "block_id": "p1_b1",
                                "block_type": "heading",
                                "text": "Attention Is All You Need",
                                "spans": [
                                    {
                                        "text": "Attention Is All You Need",
                                        "font_size": 18.0,
                                        "is_bold": True
                                    }
                                ],
                                "reading_order": 0
                            }
                        ],
                        "extraction_method": "pymupdf",
                        "ocr_applied": False
                    }
                ],
                "sections": [
                    {
                        "section_id": "sec_abstract",
                        "title": "Abstract",
                        "level": 1,
                        "text": "We propose...",
                        "page_start": 1,
                        "page_end": 1,
                        "block_ids": ["p1_b2"]
                    }
                ],
                "extraction_methods_used": ["pymupdf", "pdfplumber"]
            }
        }

