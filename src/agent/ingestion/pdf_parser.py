"""PDF parsing with multi-strategy extraction.

Implements revised component design:
- Layer 1: Text Extraction (PyMuPDF for digital PDFs)
- Layer 2: OCR fallback (PaddleOCR for scanned/mixed) - stub for now
- Layer 3: Math OCR (Qwen-VL for equations → LaTeX) - stub for now
- Layer 4: Structure Parser (pages → blocks → spans → relations)
"""

import io
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pdfplumber
from PIL import Image
from pypdf import PdfReader

from agent.ingestion.udr import (
    BoundingBox,
    DocumentMetadata,
    Equation,
    ExtractionMethod,
    Figure,
    Page,
    PageType,
    Reference,
    Section,
    Table,
    TextBlock,
    TextSpan,
    UnifiedDocumentRepresentation,
)

logger = logging.getLogger(__name__)

# Page type detection thresholds
DIGITAL_TEXT_THRESHOLD = 200  # Minimum text length for digital page
DIGITAL_IMAGE_COVERAGE_THRESHOLD = 0.3  # Maximum image coverage for digital page
SCANNED_TEXT_THRESHOLD = 50  # Maximum text length for scanned page
SCANNED_IMAGE_COVERAGE_THRESHOLD = 0.5  # Minimum image coverage for scanned page
IMAGE_AREA_ESTIMATE = 0.1  # Rough estimate of image area as fraction of page

CAPTION_REGEX = re.compile(r'(?i)\b(fig(?:ure)?|tbl|table)\s*([0-9]+[A-Za-z]?)')
CAPTION_MAX_DISTANCE = 120  # Points distance to associate captions with regions


@dataclass
class CaptionCandidate:
    """Detected caption text for figures or tables."""

    page: int
    text: str
    caption_type: str  # figure | table | unknown
    number: Optional[str]
    rect: fitz.Rect


@dataclass
class LayoutDetection:
    """Result from vision-based layout analysis."""

    page: int
    label: str
    bbox: fitz.Rect
    confidence: float


@dataclass
class FigureCandidate:
    """Potential figure region before final aggregation."""

    page: int
    bbox: fitz.Rect
    source: str  # image | layout | caption
    confidence: float
    subtype: Optional[str] = None
    caption: Optional[CaptionCandidate] = None


@dataclass
class TableCandidate:
    """Potential table region before final aggregation."""

    page: int
    bbox: fitz.Rect
    data: Optional[List[List[str]]]
    extraction_method: ExtractionMethod
    confidence: float
    caption: Optional[CaptionCandidate] = None
    header_rows: Optional[int] = None
    header_cols: Optional[int] = None


def _rect_iou(rect_a: fitz.Rect, rect_b: fitz.Rect) -> float:
    """Compute Intersection-over-Union for two rectangles."""
    if rect_a is None or rect_b is None:
        return 0.0

    inter_rect = rect_a & rect_b
    if inter_rect.is_empty:
        return 0.0

    inter_area = inter_rect.get_area()
    union_area = rect_a.get_area() + rect_b.get_area() - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _horizontal_overlap(rect_a: fitz.Rect, rect_b: fitz.Rect) -> float:
    """Return horizontal overlap ratio between two rectangles."""
    overlap = max(0.0, min(rect_a.x1, rect_b.x1) - max(rect_a.x0, rect_b.x0))
    width = max(rect_a.x1 - rect_a.x0, rect_b.x1 - rect_b.x0)
    if width <= 0:
        return 0.0
    return overlap / width


class PDFParser:
    """
    Multi-strategy PDF parser with hierarchical structure extraction.
    
    Builds: Document → Pages → Blocks → Spans
    
    Extraction strategies:
    1. PyMuPDF (primary) - Fast text extraction for digital PDFs
    2. pdfplumber - Table detection and extraction
    3. pypdf - Metadata fallback
    4. PaddleOCR - OCR for scanned pages (stub for now)
    5. Qwen-VL - Math OCR for equations (stub for now)
    """
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        # Open with different libraries
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.pdfplumber_doc = pdfplumber.open(str(self.pdf_path))
        self.pypdf_reader = PdfReader(str(self.pdf_path))
        
        # Track extraction methods used
        self.extraction_methods = set()
        self._layout_engine = self._init_layout_engine()
        self._layout_cache: Dict[int, List[LayoutDetection]] = {}
        self._latex_ocr = None
        self._latex_ocr_failed = False
    
    def close(self):
        """Close all PDF readers."""
        if hasattr(self, 'fitz_doc'):
            self.fitz_doc.close()
        if hasattr(self, 'pdfplumber_doc'):
            self.pdfplumber_doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _init_layout_engine(self):
        """Initialize optional layout detection engine (PaddleOCR PPStructure)."""
        try:
            from paddleocr import PPStructure  # type: ignore

            logger.debug("Initializing PaddleOCR PPStructure for layout analysis")
            return PPStructure(show_log=False, layout=True, ocr=False)
        except Exception as exc:  # pragma: no cover - best effort optional dependency
            logger.warning(f"Layout detection disabled (PaddleOCR unavailable): {exc}")
            return None

    def _ensure_latex_ocr(self):
        """Lazily initialize pix2tex LaTeX OCR if available."""
        if self._latex_ocr_failed:
            return None

        if self._latex_ocr is not None:
            return self._latex_ocr

        try:
            from pix2tex.cli import LatexOCR  # type: ignore

            logger.debug("Initializing pix2tex LatexOCR for equation extraction")
            self._latex_ocr = LatexOCR()
        except Exception as exc:  # pragma: no cover - best effort optional dependency
            logger.warning(f"LaTeX OCR disabled (pix2tex unavailable): {exc}")
            self._latex_ocr_failed = True
            self._latex_ocr = None

        return self._latex_ocr

    def _analyze_document_layout(self) -> Dict[int, List[LayoutDetection]]:
        """Run layout analysis for the full document (cached)."""
        if not self._layout_engine:
            return {}

        if self._layout_cache:
            return self._layout_cache

        for page_index in range(len(self.fitz_doc)):
            detections = self._analyze_layout_for_page(page_index)
            if detections:
                self._layout_cache[page_index + 1] = detections

        return self._layout_cache

    def _analyze_layout_for_page(self, page_index: int) -> List[LayoutDetection]:
        """Run layout detector on a single page and convert to document coordinates."""
        if not self._layout_engine:
            return []

        page = self.fitz_doc[page_index]
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        np_image = np.array(image)

        try:
            raw_results = self._layout_engine(np_image)
        except Exception as exc:  # pragma: no cover - best effort optional dependency
            logger.warning(f"Layout analysis failed on page {page_index + 1}: {exc}")
            return []

        scale_x = page.rect.width / pix.width if pix.width else 1.0
        scale_y = page.rect.height / pix.height if pix.height else 1.0

        detections: List[LayoutDetection] = []
        for entry in raw_results or []:
            label = entry.get("type") or entry.get("layout_type")
            bbox = entry.get("bbox") or entry.get("box")
            if not label or not bbox or len(bbox) != 4:
                continue

            rect = fitz.Rect(
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )

            confidence = float(entry.get("confidence", 0.5))
            detections.append(LayoutDetection(page=page_index + 1, label=label.lower(), bbox=rect, confidence=confidence))

        return detections

    def _collect_captions(self) -> Dict[int, List[CaptionCandidate]]:
        """Collect caption candidates for figures and tables across the document."""
        captions: Dict[int, List[CaptionCandidate]] = defaultdict(list)

        for page_index, page in enumerate(self.fitz_doc, start=1):
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception as exc:
                logger.warning(f"Failed to collect captions on page {page_index}: {exc}")
                continue

            for block in blocks:
                if block.get("type") != 0:
                    continue

                bbox = block.get("bbox") or []
                if len(bbox) != 4:
                    continue

                block_text_parts: List[str] = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if text:
                            block_text_parts.append(text)

                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue

                match = CAPTION_REGEX.search(block_text)
                if not match:
                    continue

                normalized = block_text.lower().lstrip()
                if not normalized.startswith("figure") and not normalized.startswith("fig") \
                   and not normalized.startswith("table") and not normalized.startswith("tbl"):
                    # Ignore inline references like "In Table 2 we..."
                    continue

                if len(block_text) > 400:
                    # Unusually long blocks are likely prose rather than captions
                    continue

                raw_type = match.group(1).lower()
                caption_type = "figure" if raw_type.startswith("fig") else "table"
                number = match.group(2)

                candidate = CaptionCandidate(
                    page=page_index,
                    text=block_text,
                    caption_type=caption_type,
                    number=number,
                    rect=fitz.Rect(bbox),
                )
                captions[page_index].append(candidate)

        return captions

    def _match_caption(
        self,
        captions_by_page: Dict[int, List[CaptionCandidate]],
        page: int,
        bbox: fitz.Rect,
        caption_type: str,
    ) -> Optional[CaptionCandidate]:
        """Find the closest caption of the requested type near a bounding box."""
        candidates = captions_by_page.get(page)
        if not candidates:
            return None

        best_index = None
        best_score = float("inf")

        for idx, candidate in enumerate(candidates):
            if candidate.caption_type != caption_type:
                continue

            vertical_distance = 0.0
            if candidate.rect.y0 >= bbox.y1:
                vertical_distance = candidate.rect.y0 - bbox.y1
            elif bbox.y0 >= candidate.rect.y1:
                vertical_distance = bbox.y0 - candidate.rect.y1

            if vertical_distance > CAPTION_MAX_DISTANCE:
                continue

            overlap = _horizontal_overlap(candidate.rect, bbox)
            if overlap < 0.2:
                continue

            score = vertical_distance - overlap * 20.0
            if score < best_score:
                best_score = score
                best_index = idx

        if best_index is None:
            return None

        return candidates.pop(best_index)

    def parse(self) -> UnifiedDocumentRepresentation:
        """
        Parse PDF and create hierarchical UDR.
        
        Returns:
            UnifiedDocumentRepresentation with pages → blocks → spans structure
        """
        logger.info(f"Parsing PDF: {self.pdf_path.name}")
        
        # Extract metadata
        metadata = self._extract_metadata()
        
        # Extract pages with hierarchical structure
        pages, ocr_pages = self._extract_pages_with_blocks()
        
        # Build legacy raw text list (backward compatibility)
        raw_page_texts = [page.text for page in pages]
        
        # Extract sections (heuristic-based from headings)
        sections = self._extract_sections_from_blocks(pages)

        # Analyze layout (vision-based) and captions for downstream detection
        layout_by_page = self._analyze_document_layout()
        captions_by_page = self._collect_captions()

        # Extract tables and figures using multi-signal fusion
        tables = self._extract_tables(layout_by_page, captions_by_page)
        figures = self._extract_figures(layout_by_page, captions_by_page)

        # Extract equations (still heuristic-based with optional OCR)
        equations = self._extract_equations()
        
        # Extract references (stub for now)
        references = self._extract_references()
        
        # Update metadata with page type distribution
        metadata.num_digital_pages = sum(1 for p in pages if p.page_type == PageType.DIGITAL)
        metadata.num_scanned_pages = sum(1 for p in pages if p.page_type == PageType.SCANNED)
        metadata.num_mixed_pages = sum(1 for p in pages if p.page_type == PageType.MIXED)
        
        # Build UDR
        udr = UnifiedDocumentRepresentation(
            metadata=metadata,
            pages=pages,
            raw_page_texts=raw_page_texts,
            sections=sections,
            tables=tables,
            figures=figures,
            equations=equations,
            references=references,
            extraction_methods_used=list(self.extraction_methods),
            ocr_pages=ocr_pages
        )
        
        logger.info(
            f"Parsed {metadata.num_pages} pages "
            f"({metadata.num_digital_pages} digital, {metadata.num_scanned_pages} scanned, {metadata.num_mixed_pages} mixed), "
            f"{len(sections)} sections, "
            f"{len(tables)} tables, "
            f"{sum(len(p.blocks) for p in pages)} blocks"
        )
        
        return udr
    
    def _detect_page_type(self, page: fitz.Page) -> PageType:
        """
        Detect if page is digital (text-based), scanned (image), or mixed.
        
        Heuristic:
        - If page has significant extractable text → DIGITAL
        - If page is mostly images with little text → SCANNED
        - Otherwise → MIXED
        """
        # Get text content
        text = page.get_text("text").strip()
        text_length = len(text)
        
        # Get images
        images = page.get_images()
        num_images = len(images)
        
        # Get page area
        page_area = page.rect.width * page.rect.height
        
        # Calculate image coverage (rough estimate)
        image_area = 0
        for img_ref in images:
            try:
                # Get image dimensions (rough estimate)
                xref = img_ref[0]
                base_image = self.fitz_doc.extract_image(xref)
                if base_image:
                    # Simple heuristic: assume images take significant space
                    image_area += page_area * IMAGE_AREA_ESTIMATE
            except:
                pass
        
        image_coverage = min(image_area / page_area, 1.0) if page_area > 0 else 0
        
        # Decision logic
        if text_length > DIGITAL_TEXT_THRESHOLD and image_coverage < DIGITAL_IMAGE_COVERAGE_THRESHOLD:
            # Significant text, few images → Digital
            return PageType.DIGITAL
        elif text_length < SCANNED_TEXT_THRESHOLD and (num_images > 0 or image_coverage > SCANNED_IMAGE_COVERAGE_THRESHOLD):
            # Little text, has images → Scanned
            return PageType.SCANNED
        else:
            # Mixed content
            return PageType.MIXED
    
    def _extract_pages_with_blocks(self) -> Tuple[List[Page], List[int]]:
        """
        Extract pages with hierarchical block structure.
        
        Returns:
            Tuple of (pages, ocr_pages_list)
        """
        logger.debug(f"Extracting {len(self.fitz_doc)} pages with blocks")
        
        pages = []
        ocr_pages = []
        
        for page_num, fitz_page in enumerate(self.fitz_doc, start=1):
            try:
                # Detect page type
                page_type = self._detect_page_type(fitz_page)
                
                # Extract blocks based on page type
                if page_type == PageType.SCANNED:
                    # TODO: Implement PaddleOCR extraction
                    # For now, fallback to PyMuPDF (will have limited text)
                    blocks, ocr_applied = self._extract_blocks_pymupdf(fitz_page, page_num)
                    logger.warning(f"Page {page_num} is scanned but OCR not yet implemented")
                else:
                    # Use PyMuPDF for digital/mixed pages
                    blocks, ocr_applied = self._extract_blocks_pymupdf(fitz_page, page_num)
                
                if ocr_applied:
                    ocr_pages.append(page_num)
                
                # Compute average OCR confidence if applicable
                ocr_confidence = None
                if ocr_applied and blocks:
                    confidences = [b.confidence for b in blocks if b.confidence is not None]
                    if confidences:
                        ocr_confidence = sum(confidences) / len(confidences)
                
                # Get page dimensions
                rect = fitz_page.rect
                
                # Build Page object
                page = Page(
                    page_num=page_num,
                    page_type=page_type,
                    width=rect.width,
                    height=rect.height,
                    text="\n\n".join([b.text for b in blocks if b.text]),
                    blocks=blocks,
                    extraction_method=ExtractionMethod.PADDLEOCR if ocr_applied else ExtractionMethod.PYMUPDF,
                    ocr_applied=ocr_applied,
                    ocr_confidence=ocr_confidence
                )
                
                pages.append(page)
                
                logger.debug(
                    f"Page {page_num}: {page_type.value}, "
                    f"{len(blocks)} blocks, "
                    f"{len(page.text)} chars"
                )
                
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}: {e}")
                # Create empty page on error
                pages.append(Page(
                    page_num=page_num,
                    page_type=PageType.DIGITAL,
                    width=612.0,
                    height=792.0,
                    text="",
                    blocks=[],
                    extraction_method=ExtractionMethod.PYMUPDF,
                    ocr_applied=False
                ))
        
        # Mark PyMuPDF as used
        self.extraction_methods.add(ExtractionMethod.PYMUPDF)
        
        return pages, ocr_pages
    
    def _extract_blocks_pymupdf(
        self, 
        page: fitz.Page, 
        page_num: int
    ) -> Tuple[List[TextBlock], bool]:
        """
        Extract text blocks with spans using PyMuPDF.
        
        Returns:
            Tuple of (blocks, ocr_applied)
        """
        blocks = []
        
        # Get text with detailed formatting using "dict" mode
        text_dict = page.get_text("dict")
        
        block_num = 0
        for block in text_dict.get("blocks", []):
            # Skip non-text blocks (images)
            if block.get("type") != 0:
                continue
            
            # Extract spans from lines
            spans = []
            block_text_parts = []
            
            for line in block.get("lines", []):
                for span_data in line.get("spans", []):
                    span_text = span_data.get("text", "").strip()
                    if not span_text:
                        continue
                    
                    # Extract font information
                    font = span_data.get("font", "")
                    size = span_data.get("size", 0)
                    
                    # Detect bold/italic from font name
                    is_bold = "bold" in font.lower()
                    is_italic = "italic" in font.lower() or "oblique" in font.lower()
                    
                    # Create bounding box
                    bbox_coords = span_data.get("bbox")
                    bbox = None
                    if bbox_coords and len(bbox_coords) == 4:
                        bbox = BoundingBox(
                            x0=bbox_coords[0],
                            y0=bbox_coords[1],
                            x1=bbox_coords[2],
                            y1=bbox_coords[3],
                            page=page_num
                        )
                    
                    span = TextSpan(
                        text=span_text,
                        bbox=bbox,
                        font_size=size,
                        font_name=font,
                        is_bold=is_bold,
                        is_italic=is_italic
                    )
                    
                    spans.append(span)
                    block_text_parts.append(span_text)
                
                # Add line break
                block_text_parts.append("\n")
            
            if not spans:
                continue
            
            # Combine text
            block_text = " ".join(block_text_parts).strip()
            
            # Clean up excessive whitespace
            block_text = re.sub(r'\s+', ' ', block_text)
            
            # Detect block type (heuristic)
            block_type = self._detect_block_type(spans, block_text)
            
            # Create block bounding box
            block_bbox_coords = block.get("bbox")
            block_bbox = None
            if block_bbox_coords and len(block_bbox_coords) == 4:
                block_bbox = BoundingBox(
                    x0=block_bbox_coords[0],
                    y0=block_bbox_coords[1],
                    x1=block_bbox_coords[2],
                    y1=block_bbox_coords[3],
                    page=page_num
                )
            
            # Create TextBlock
            text_block = TextBlock(
                block_id=f"p{page_num}_b{block_num}",
                block_type=block_type,
                text=block_text,
                spans=spans,
                bbox=block_bbox,
                reading_order=block_num,
                extraction_method=ExtractionMethod.PYMUPDF
            )
            
            blocks.append(text_block)
            block_num += 1
        
        return blocks, False  # OCR not applied (PyMuPDF)
    
    def _detect_block_type(self, spans: List[TextSpan], text: str) -> str:
        """
        Detect block type from spans and text content.
        
        Heuristics:
        - Large font + bold → heading
        - Starts with number/bullet → list
        - Short text + ends with colon → caption/label
        - Otherwise → paragraph
        """
        if not spans:
            return "paragraph"
        
        # Get average font size
        font_sizes = [s.font_size for s in spans if s.font_size]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        
        # Check if mostly bold
        bold_count = sum(1 for s in spans if s.is_bold)
        mostly_bold = bold_count / len(spans) > 0.5 if spans else False
        
        # Heading: Large font + bold
        if avg_font_size > 14 and mostly_bold:
            return "heading"
        
        # List: Starts with number or bullet
        if re.match(r'^[\d\-•·◦▪▫]+[\.\)]\s+', text):
            return "list"
        
        # Caption: Short + ends with colon
        if len(text) < 100 and text.strip().endswith(':'):
            return "caption"
        
        return "paragraph"
    
    def _extract_metadata(self) -> DocumentMetadata:
        """Extract document metadata from PDF."""
        logger.debug("Extracting metadata")
        
        # Try PyMuPDF first
        fitz_meta = self.fitz_doc.metadata or {}
        
        # Try pypdf for additional metadata
        pypdf_meta = self.pypdf_reader.metadata or {}
        
        # Extract title (priority: PDF metadata → first page heuristic)
        title = (
            fitz_meta.get("title") or 
            pypdf_meta.get("/Title") or 
            self._extract_title_from_first_page()
        )
        
        # Extract authors
        author_str = fitz_meta.get("author") or pypdf_meta.get("/Author") or ""
        authors = [a.strip() for a in author_str.split(",") if a.strip()] if author_str else []
        
        # Extract year from creation date or content
        year = self._extract_year(fitz_meta, pypdf_meta)
        
        metadata = DocumentMetadata(
            title=title,
            authors=authors,
            publication_year=year,
            num_pages=len(self.fitz_doc),
        )
        
        logger.debug(f"Extracted metadata: title='{title}', authors={authors}, year={year}")
        
        return metadata
    
    def _extract_title_from_first_page(self) -> Optional[str]:
        """Heuristic: Extract title from first page (usually largest/bold text at top)."""
        if not self.fitz_doc or len(self.fitz_doc) == 0:
            return None
        
        try:
            first_page = self.fitz_doc[0]
            
            # Get text blocks with font information
            blocks = first_page.get_text("dict")["blocks"]
            
            # Find largest text in top portion of page
            candidates = []
            page_height = first_page.rect.height
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    bbox = block.get("bbox", [])
                    if len(bbox) >= 4 and bbox[1] < page_height * 0.3:  # Top 30%
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                size = span.get("size", 0)
                                if text and size > 14 and len(text) > 10:  # Likely title
                                    candidates.append((size, text))
            
            if candidates:
                # Return largest text
                candidates.sort(reverse=True)
                return candidates[0][1]
        
        except Exception as e:
            logger.warning(f"Error extracting title: {e}")
        
        return None
    
    def _extract_abstract_from_first_page(self) -> Optional[str]:
        """Heuristic: Find abstract section on first page."""
        if not self.fitz_doc or len(self.fitz_doc) == 0:
            return None
        
        try:
            first_page = self.fitz_doc[0]
            text = first_page.get_text("text")
            
            # Look for "Abstract" section
            abstract_match = re.search(
                r'abstract[:\s]*\n?(.*?)(?:\n\n|\n[A-Z]|\Z)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                # Limit length
                if len(abstract) > 1000:
                    abstract = abstract[:1000] + "..."
                return abstract
        
        except Exception as e:
            logger.warning(f"Error extracting abstract: {e}")
        
        return None
    
    def _extract_year(self, fitz_meta: dict, pypdf_meta: dict) -> Optional[int]:
        """Extract publication year from metadata."""
        # Try creation date
        date_str = fitz_meta.get("creationDate") or pypdf_meta.get("/CreationDate") or ""
        
        # Extract 4-digit year
        year_match = re.search(r'(19|20)\d{2}', date_str)
        if year_match:
            return int(year_match.group())
        
        return None
    
    def _extract_sections_from_blocks(self, pages: List[Page]) -> List[Section]:
        """
        Extract sections from heading blocks.
        
        Identifies heading blocks and creates Section objects.
        """
        logger.debug("Extracting sections from heading blocks")
        
        sections = []
        section_counter = 0
        
        for page in pages:
            for block in page.blocks:
                if block.block_type == "heading":
                    # Create section from heading
                    section = Section(
                        section_id=f"sec_{section_counter}",
                        title=block.text,
                        level=1,  # Could be enhanced with level detection
                        text="",  # Will be populated during chunking
                        page_start=page.page_num,
                        page_end=page.page_num,
                        block_ids=[block.block_id]
                    )
                    sections.append(section)
                    section_counter += 1
        
        logger.debug(f"Extracted {len(sections)} sections from headings")
        
        return sections
    
    def _extract_tables(self) -> List[Table]:
        """Extract tables using pdfplumber."""
        logger.debug("Extracting tables")
        
        tables = []
        self.extraction_methods.add(ExtractionMethod.PDFPLUMBER)
        
        for page_num, page in enumerate(self.pdfplumber_doc.pages, start=1):
            try:
                page_tables = page.extract_tables()
                
                for table_idx, table_data in enumerate(page_tables):
                    if not table_data:
                        continue
                    
                    # Convert to text representation
                    text_rows = []
                    for row in table_data:
                        row_text = " | ".join([str(cell) if cell else "" for cell in row])
                        text_rows.append(row_text)
                    
                    text = "\n".join(text_rows)
                    
                    table = Table(
                        table_id=f"tbl_p{page_num}_{table_idx}",
                        page=page_num,
                        data=table_data,
                        text=text,
                        extraction_method=ExtractionMethod.PDFPLUMBER
                    )
                    
                    tables.append(table)
                    
                    logger.debug(f"Extracted table from page {page_num}: {len(table_data)} rows")
                    
            except Exception as e:
                logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        logger.debug(f"Extracted {len(tables)} tables total")
        
        return tables
    
    def _extract_figures(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[Figure]:
        """Extract figures by combining raster images, layout detections, and captions."""
        logger.debug("Extracting figures with fused signals")

        figure_candidates: List[FigureCandidate] = []
        self.extraction_methods.add(ExtractionMethod.PYMUPDF)

        def merge_candidate(candidate: FigureCandidate):
            for existing in figure_candidates:
                if existing.page != candidate.page:
                    continue
                if _rect_iou(existing.bbox, candidate.bbox) > 0.5:
                    if candidate.caption and not existing.caption:
                        existing.caption = candidate.caption
                    existing.confidence = max(existing.confidence, candidate.confidence)
                    if candidate.subtype and not existing.subtype:
                        existing.subtype = candidate.subtype
                    return
            figure_candidates.append(candidate)

        for page_index, page in enumerate(self.fitz_doc):
            page_num = page_index + 1
            page_rect = page.rect

            # Strategy 1: layout detections (charts, figures)
            for detection in layout_by_page.get(page_num, []):
                label = detection.label.lower()
                if "caption" in label:
                    continue
                if label not in {"figure", "image", "chart", "graph", "picture"}:
                    continue

                caption = self._match_caption(captions_by_page, page_num, detection.bbox, "figure")

                subtype = None
                if "chart" in label or "graph" in label:
                    subtype = "chart"

                candidate = FigureCandidate(
                    page=page_num,
                    bbox=detection.bbox,
                    source="layout",
                    confidence=detection.confidence,
                    subtype=subtype,
                    caption=caption,
                )
                merge_candidate(candidate)
                self.extraction_methods.add(ExtractionMethod.PADDLEOCR)

            # Strategy 2: raster images on the page
            try:
                image_list = page.get_images(full=True)
            except Exception as exc:
                logger.warning(f"Failed to enumerate images on page {page_num}: {exc}")
                image_list = []

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue

                    img_rect = rects[0]
                    base_image = self.fitz_doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image.get("image")
                    if not image_bytes or len(image_bytes) < 5000:
                        continue

                    img_area = img_rect.get_area()
                    if img_area < 25000:
                        caption_probe = self._match_caption(captions_by_page, page_num, img_rect, "figure")
                        if not caption_probe:
                            continue
                        caption = caption_probe
                    else:
                        caption = self._match_caption(captions_by_page, page_num, img_rect, "figure")

                    candidate = FigureCandidate(
                        page=page_num,
                        bbox=img_rect,
                        source="image",
                        confidence=0.75,
                        caption=caption,
                    )
                    merge_candidate(candidate)
                except Exception as exc:
                    logger.warning(f"Error processing image {img_index} on page {page_num}: {exc}")
                    continue

            # Strategy 3: leftover captions without matched visuals
            remaining_captions = [
                caption
                for caption in captions_by_page.get(page_num, [])
                if caption.caption_type == "figure"
            ]

            if remaining_captions:
                updated_captions = []
                for caption in captions_by_page.get(page_num, []):
                    if caption in remaining_captions:
                        continue
                    updated_captions.append(caption)
                captions_by_page[page_num] = updated_captions

            for caption in remaining_captions:
                expanded_rect = fitz.Rect(
                    caption.rect.x0,
                    max(0.0, caption.rect.y0 - 80.0),
                    caption.rect.x1,
                    min(page_rect.y1, caption.rect.y1 + 20.0),
                )

                candidate = FigureCandidate(
                    page=page_num,
                    bbox=expanded_rect,
                    source="caption",
                    confidence=0.45,
                    caption=caption,
                )
                merge_candidate(candidate)
                self.extraction_methods.add(ExtractionMethod.HEURISTIC)

        figures: List[Figure] = []
        for idx, candidate in enumerate(figure_candidates):
            bbox_model = BoundingBox(
                x0=candidate.bbox.x0,
                y0=candidate.bbox.y0,
                x1=candidate.bbox.x1,
                y1=candidate.bbox.y1,
                page=candidate.page,
            )

            if candidate.source == "image":
                extraction_method = ExtractionMethod.PYMUPDF
            elif candidate.source == "layout":
                extraction_method = ExtractionMethod.PADDLEOCR
            else:
                extraction_method = ExtractionMethod.HEURISTIC

            figure = Figure(
                figure_id=f"fig_{idx}",
                caption=candidate.caption.text if candidate.caption else None,
                page=candidate.page,
                bbox=bbox_model,
                s3_key=None,
                subtype=candidate.subtype or None,
                confidence=round(min(candidate.confidence + 0.05, 1.0), 2),
                caption_id=(candidate.caption.number if candidate.caption else None),
                artifacts={},
                extraction_method=extraction_method,
            )
            figures.append(figure)

        logger.debug(f"Extracted {len(figures)} figures total")
        return figures
    
    def _extract_tables(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[Table]:
        """Extract tables by fusing PDF primitives, layout detection, and captions."""
        logger.debug("Extracting tables with fused signals")

        table_candidates: List[TableCandidate] = []
        self.extraction_methods.add(ExtractionMethod.PDFPLUMBER)

        def merge_candidate(candidate: TableCandidate):
            for existing in table_candidates:
                if existing.page != candidate.page:
                    continue
                if _rect_iou(existing.bbox, candidate.bbox) > 0.55:
                    if not existing.data and candidate.data:
                        existing.data = candidate.data
                        existing.extraction_method = candidate.extraction_method
                    if candidate.caption and not existing.caption:
                        existing.caption = candidate.caption
                    existing.confidence = max(existing.confidence, candidate.confidence)
                    if candidate.header_rows is not None:
                        existing.header_rows = candidate.header_rows
                    if candidate.header_cols is not None:
                        existing.header_cols = candidate.header_cols
                    return
            table_candidates.append(candidate)

        def normalize_table_data(table_data: Optional[List[List[str]]]) -> List[List[str]]:
            """Convert table cells to strings and replace missing values."""
            cleaned: List[List[str]] = []
            if not table_data:
                return cleaned

            for row in table_data:
                if row is None:
                    continue
                cleaned.append(["" if cell is None else str(cell) for cell in row])

            return cleaned

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        }

        for page_num, plumber_page in enumerate(self.pdfplumber_doc.pages, start=1):
            # Strategy 1: Tables detected via pdfplumber line analysis
            try:
                detected_tables = plumber_page.find_tables(table_settings)
            except Exception as exc:
                logger.warning(f"pdfplumber failed on page {page_num}: {exc}")
                detected_tables = []

            for table_obj in detected_tables or []:
                try:
                    data = table_obj.extract()
                except Exception:
                    data = None

                data = normalize_table_data(data)

                bbox = fitz.Rect(*table_obj.bbox)
                caption = self._match_caption(captions_by_page, page_num, bbox, "table")

                candidate = TableCandidate(
                    page=page_num,
                    bbox=bbox,
                    data=data,
                    extraction_method=ExtractionMethod.PDFPLUMBER,
                    confidence=0.8,
                    caption=caption,
                )
                merge_candidate(candidate)

            # Strategy 2: Layout detection to catch tables without explicit lines
            for detection in layout_by_page.get(page_num, []):
                if detection.label not in {"table", "table caption"}:
                    continue

                self.extraction_methods.add(ExtractionMethod.PADDLEOCR)

                caption = self._match_caption(captions_by_page, page_num, detection.bbox, "table")

                # Attempt to extract data from the detected region if not already covered
                data = None
                try:
                    cropped = plumber_page.crop((detection.bbox.x0, detection.bbox.y0, detection.bbox.x1, detection.bbox.y1))
                    data = cropped.extract_table(table_settings=table_settings)
                except Exception:
                    data = None

                data = normalize_table_data(data)

                candidate = TableCandidate(
                    page=page_num,
                    bbox=detection.bbox,
                    data=data,
                    extraction_method=ExtractionMethod.PADDLEOCR,
                    confidence=detection.confidence,
                    caption=caption,
                )
                merge_candidate(candidate)

        tables: List[Table] = []
        for idx, candidate in enumerate(table_candidates):
            text_lines: List[str] = []
            normalized_data = normalize_table_data(candidate.data)
            if normalized_data:
                for row in normalized_data:
                    text_lines.append(" | ".join(row))
            else:
                # Fallback to textual extraction from PDF for bounding box region
                try:
                    page = self.fitz_doc[candidate.page - 1]
                    clip_text = page.get_text("text", clip=candidate.bbox)
                    text_lines = [line.strip() for line in clip_text.splitlines() if line.strip()]
                except Exception:
                    text_lines = []

            row_count = len(normalized_data)
            col_count = max((len(row) for row in normalized_data), default=0)
            has_structured_shape = row_count >= 2 and col_count >= 2
            has_multiline_text = len(text_lines) >= 2

            if not has_structured_shape and not has_multiline_text:
                logger.debug(
                    "Skipping table candidate on page %s (rows=%s, cols=%s, lines=%s)",
                    candidate.page,
                    row_count,
                    col_count,
                    len(text_lines),
                )
                continue

            table_text = "\n".join(text_lines)

            bbox_model = BoundingBox(
                x0=candidate.bbox.x0,
                y0=candidate.bbox.y0,
                x1=candidate.bbox.x1,
                y1=candidate.bbox.y1,
                page=candidate.page,
            )

            table = Table(
                table_id=f"tbl_{idx}",
                caption=candidate.caption.text if candidate.caption else None,
                page=candidate.page,
                bbox=bbox_model,
                data=normalized_data,
                text=table_text,
                subtype="data" if has_structured_shape else None,
                confidence=round(min(candidate.confidence + 0.1, 1.0), 2),
                caption_id=(candidate.caption.number if candidate.caption else None),
                header_rows=candidate.header_rows,
                header_cols=candidate.header_cols,
                artifacts={},
                extraction_method=candidate.extraction_method,
            )
            tables.append(table)

        logger.debug(f"Extracted {len(tables)} tables total")
        return tables

    def _extract_equations(self) -> List[Equation]:
        """Extract display equations using text layout heuristics."""
        logger.debug("Extracting equations with layout heuristics")

        equations: List[Equation] = []
        latex_ocr = self._ensure_latex_ocr()
        has_latex_ocr = latex_ocr is not None

        self.extraction_methods.add(ExtractionMethod.PYMUPDF)
        if has_latex_ocr:
            self.extraction_methods.add(ExtractionMethod.QWEN_VL)

        math_symbols = set("=±×÷∑∫∏√∞∂πθλμσΩαβγδ")

        for page_index, page in enumerate(self.fitz_doc):
            page_number = page_index + 1
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception as exc:
                logger.warning(f"Failed to extract text blocks for equations on page {page_number}: {exc}")
                continue

            equation_idx = 0

            for block_idx, block in enumerate(blocks):
                if block.get("type") != 0:
                    continue

                bbox_list = block.get("bbox", [])
                if len(bbox_list) < 4:
                    continue

                block_rect = fitz.Rect(bbox_list)

                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                block_text = block_text.strip()

                if not block_text:
                    continue

                lower_text = block_text.lower()
                common_words = ["the", "and", "for", "are", "that", "this", "with", "from", "where", "which"]
                if sum(1 for word in common_words if word in lower_text) >= 3:
                    continue

                has_math = any(char in math_symbols for char in block_text)
                has_latex_pattern = bool(re.search(r"[∑∫∏√]|[α-ω]|=", block_text))
                is_display = self._is_display_equation(block, blocks, block_idx)
                is_small_centered = is_display and len(block_text) < 200

                if not (is_small_centered and (has_math or has_latex_pattern)):
                    continue

                equation_id = f"eq_p{page_number}_{equation_idx}"
                equation_idx += 1

                latex_str = None
                latex_source = None
                confidence = None

                if has_latex_ocr:
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat, clip=block_rect)
                        image_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(image_bytes))
                        try:
                            latex_str = latex_ocr(img)
                        finally:
                            img.close()
                        latex_source = ExtractionMethod.QWEN_VL
                        confidence = 0.8
                        logger.debug(f"Extracted equation {equation_id}: {latex_str[:50]}...")
                    except Exception as exc:
                        logger.warning(f"Error converting equation {equation_id} to LaTeX: {exc}")
                        latex_str = None

                bbox = BoundingBox(
                    x0=block_rect.x0,
                    y0=block_rect.y0,
                    x1=block_rect.x1,
                    y1=block_rect.y1,
                    page=page_number,
                )

                equation = Equation(
                    equation_id=equation_id,
                    latex=latex_str,
                    text=block_text,
                    page=page_number,
                    bbox=bbox,
                    is_inline=not is_display,
                    image_s3_key=None,
                    extraction_method=ExtractionMethod.PYMUPDF,
                    latex_source=latex_source,
                    confidence=confidence,
                )
                equations.append(equation)

        logger.debug(f"Extracted {len(equations)} equations total")
        return equations
    
    def _is_display_equation(self, block: dict, all_blocks: List[dict], block_idx: int) -> bool:
        """
        Determine if a block is a display (centered) equation.
        
        Heuristics:
        - Block is relatively small
        - Block has significant left/right margins (centered)
        - Block is isolated (has space above/below)
        """
        bbox_list = block.get("bbox", [])
        if len(bbox_list) < 4:
            return False
        
        block_rect = fitz.Rect(bbox_list)
        block_width = block_rect.x1 - block_rect.x0
        
        # Get page width (approximate from first block or assume standard)
        page_width = 612.0  # Standard US Letter width in points
        for b in all_blocks:
            if b.get("type") == 0:
                bb = b.get("bbox", [])
                if len(bb) >= 4:
                    page_width = max(page_width, bb[2])
                    break
        
        # Check if centered (has significant margins)
        left_margin = block_rect.x0
        right_margin = page_width - block_rect.x1
        
        # If both margins are significant and roughly equal, likely centered
        is_centered = (
            left_margin > 100 and 
            right_margin > 100 and 
            abs(left_margin - right_margin) < 50
        )
        
        # Check if relatively small
        is_small = block_width < page_width * 0.6
        
        # Check for isolation (space above/below)
        has_space = self._check_vertical_space(block_rect, all_blocks, block_idx)
        
        return is_centered and is_small and has_space
    
    def _check_vertical_space(self, block_rect: fitz.Rect, all_blocks: List[dict], block_idx: int) -> bool:
        """Check if block has vertical space above and below."""
        min_space = 10  # pixels
        
        # Check previous block
        if block_idx > 0:
            prev_block = all_blocks[block_idx - 1]
            if prev_block.get("type") == 0:
                prev_bbox = prev_block.get("bbox", [])
                if len(prev_bbox) >= 4:
                    space_above = block_rect.y0 - prev_bbox[3]
                    if space_above < min_space:
                        return False
        
        # Check next block
        if block_idx < len(all_blocks) - 1:
            next_block = all_blocks[block_idx + 1]
            if next_block.get("type") == 0:
                next_bbox = next_block.get("bbox", [])
                if len(next_bbox) >= 4:
                    space_below = next_bbox[1] - block_rect.y1
                    if space_below < min_space:
                        return False
        
        return True
    
    def _extract_references(self) -> List[Reference]:
        """Extract references (stub for now)."""
        logger.debug("Extracting references (stub)")
        
        # TODO: Implement reference extraction
        # - Find References/Bibliography section
        # - Parse individual citations
        # - Extract author, title, year, venue, DOI
        
        references = []
        
        return references


def parse_pdf(pdf_path: Path) -> UnifiedDocumentRepresentation:
    """
    Convenience function to parse a PDF file.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        UnifiedDocumentRepresentation
    """
    with PDFParser(pdf_path) as parser:
        return parser.parse()
