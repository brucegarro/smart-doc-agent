"""PDF parsing with multi-strategy extraction.

Implements revised component design:
- Layer 1: Text Extraction (PyMuPDF for digital PDFs)
- Layer 2: OCR fallback (PaddleOCR for scanned/mixed) - stub for now
- Layer 3: Math OCR (Qwen-VL for equations → LaTeX) - stub for now
- Layer 4: Structure Parser (pages → blocks → spans → relations)
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
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
        
        # Extract tables (using pdfplumber)
        tables = self._extract_tables()
        
        # Extract figures (stub for now)
        figures = self._extract_figures()
        
        # Extract equations (stub for now)
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
    
    def _extract_figures(self) -> List[Figure]:
        """Extract figures (stub for now)."""
        logger.debug("Extracting figures (stub)")
        
        # TODO: Implement figure extraction
        # - Detect image blocks
        # - Extract images
        # - Upload to S3
        # - Detect captions
        
        figures = []
        
        return figures
    
    def _extract_equations(self) -> List[Equation]:
        """Extract equations (stub for now)."""
        logger.debug("Extracting equations (stub)")
        
        # TODO: Implement equation extraction
        # - Detect math regions (LaTeX patterns, special symbols)
        # - Extract equation images
        # - Call Qwen-VL for LaTeX conversion
        # - Store images in S3
        
        equations = []
        
        return equations
    
    def _extract_references(self) -> List[Reference]:
        """Extract references (stub for now)."""
        logger.debug("Extracting references (stub)")
        
        # TODO: Implement reference extraction
        # - Find References/Bibliography section
        # - Parse individual citations
        # - Extract author, title, year, venue, DOI
        
        references = []
        
        return references
        
        tables = []
        
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
                        page=page_num,
                        data=table_data,
                        text=text
                    )
                    
                    tables.append(table)
                    
                    logger.debug(f"Extracted table {table_idx + 1} from page {page_num}: {len(table_data)} rows")
                    
            except Exception as e:
                logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        logger.debug(f"Extracted {len(tables)} tables total")
        
        return tables


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
