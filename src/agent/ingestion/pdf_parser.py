"""PDF parsing with multi-strategy extraction.

Implements revised component design:
- Layer 1: Text Extraction (PyMuPDF for digital PDFs)
- Layer 2: OCR fallback (PaddleOCR for scanned/mixed) - stub for now
- Layer 3: Math OCR (Qwen-VL for equations → LaTeX) - stub for now
- Layer 4: Structure Parser (pages → blocks → spans → relations)
"""

import base64
import io
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pdfplumber
from PIL import Image
from pypdf import PdfReader

from agent.config import settings
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

try:  # Optional multimodal model client
    from ollama import Client as OllamaClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OllamaClient = None  # type: ignore

# Page type detection thresholds
DIGITAL_TEXT_THRESHOLD = 200  # Minimum text length for digital page
DIGITAL_IMAGE_COVERAGE_THRESHOLD = 0.3  # Maximum image coverage for digital page
SCANNED_TEXT_THRESHOLD = 50  # Maximum text length for scanned page
SCANNED_IMAGE_COVERAGE_THRESHOLD = 0.5  # Minimum image coverage for scanned page
IMAGE_AREA_ESTIMATE = 0.1  # Rough estimate of image area as fraction of page

REFERENCE_HEADING_PATTERN = re.compile(r"\b(references?|bibliography|works cited)\b", re.IGNORECASE)
REFERENCE_ENTRY_START_PATTERN = re.compile(r"^\s*(?:\[\d+\]|\(?\d+\)?[\.)])\s+")

CAPTION_REGEX = re.compile(r'(?i)\b(fig(?:ure)?|tbl|table)\s*([0-9]+[A-Za-z]?)')
CAPTION_MAX_DISTANCE = 120  # Points distance to associate captions with regions

# Geometric thresholds for reliable table detection
MIN_TABLE_WIDTH = 120.0   # points (~1.6in)
MIN_TABLE_HEIGHT = 80.0   # points (~1.1in)
MIN_TABLE_ASPECT_RATIO = 0.2  # prevent ultra-narrow tables (w / h)
MAX_TABLE_ASPECT_RATIO = 5.0  # prevent ultra-flat tables

# Vision-language (Qwen-VL) analysis settings
VLM_MAX_TABLES = 4
VLM_MAX_FIGURES = 4
VLM_RENDER_SCALE = 2.0
VLM_RENDER_PADDING = 8.0


class _ParseStepTracker:
    """Utility to measure parse step durations while keeping cyclomatic complexity low."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}
        self._step_start: float = time.perf_counter()

    def wrap(self, label: str, key: str, func, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Execute func, recording elapsed time under the provided label/key."""
        self._step_start = time.perf_counter()
        result = func(*args, **kwargs)
        self._record(label, key)
        return result

    def skip(self, label: str, key: str) -> None:
        """Mark a step as skipped and reset the timer baseline."""
        self.timings[key] = 0.0
        logger.info("%s timings (s) | %s=0.00 (skipped)", label, key)
        self._step_start = time.perf_counter()

    def _record(self, label: str, key: str) -> None:
        now = time.perf_counter()
        elapsed = now - self._step_start
        self.timings[key] = elapsed
        logger.info("%s timings (s) | %s=%.2f", label, key, elapsed)
        self._step_start = now

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


@dataclass
class _ReferenceAccumulator:
    """Mutable state while scanning reference sections."""

    references: List[Reference] = field(default_factory=list)
    collecting: bool = False
    current_lines: List[str] = field(default_factory=list)
    current_page: Optional[int] = None
    counter: int = 1

    def start_section(self) -> None:
        self.finalize_current()
        self.collecting = True

    def end_section(self) -> None:
        self.finalize_current()
        self.collecting = False

    def add_block(self, page_num: int, block_text: str) -> None:
        normalized = re.sub(r"\s+", " ", block_text).strip()
        if not normalized:
            return
        if self.current_lines and REFERENCE_ENTRY_START_PATTERN.match(normalized):
            self.finalize_current()
        if not self.current_lines:
            self.current_page = page_num
        self.current_lines.append(normalized)

    def finalize_current(self) -> None:
        if not self.current_lines:
            self.current_page = None
            return

        text = re.sub(r"\s+", " ", " ".join(self.current_lines)).strip()
        self.current_lines = []
        if not text:
            self.current_page = None
            return

        reference = Reference(
            reference_id=f"ref_{self.counter}",
            text=text,
            page=self.current_page,
        )
        self.references.append(reference)
        self.counter += 1
        self.current_page = None

@dataclass(frozen=True)
class ParseOptions:
    """Feature toggles for PDF parsing."""

    enable_layout_analysis: bool = True
    enable_table_detection: bool = True
    enable_figure_detection: bool = True
    enable_vlm_enrichment: bool = True
    enable_equation_detection: bool = True
    enable_reference_detection: bool = True

    @classmethod
    def fast_ingest(cls) -> "ParseOptions":
        """Return options optimized for ingestion speed."""

        return cls(
            enable_layout_analysis=False,
            enable_table_detection=False,
            enable_figure_detection=False,
            enable_vlm_enrichment=False,
            enable_equation_detection=False,
            enable_reference_detection=True,
        )


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
    
    def __init__(self, pdf_path: Path, options: Optional[ParseOptions] = None):
        self.pdf_path = Path(pdf_path)
        self.options = options or ParseOptions()
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        # Open with different libraries
        self.fitz_doc = fitz.open(str(self.pdf_path))
        self.pdfplumber_doc = pdfplumber.open(str(self.pdf_path))
        self.pypdf_reader = PdfReader(str(self.pdf_path))
        
        # Track extraction methods used
        self.extraction_methods = set()
        if (
            self.options.enable_layout_analysis
            or self.options.enable_table_detection
            or self.options.enable_figure_detection
        ):
            self._layout_engine = self._init_layout_engine()
        else:
            self._layout_engine = None
        self._layout_cache: Dict[int, List[LayoutDetection]] = {}
        self._latex_ocr = None
        self._latex_ocr_failed = False
        self._vlm_client = None
        self._vlm_unavailable = False
    
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

    def _ensure_vlm_client(self):
        """Lazily initialize Ollama client for Qwen-VL analysis."""
        if self._vlm_unavailable:
            return None

        if self._vlm_client is not None:
            return self._vlm_client

        if OllamaClient is None:
            logger.warning("Qwen-VL analysis disabled (ollama package unavailable)")
            self._vlm_unavailable = True
            return None

        try:
            self._vlm_client = OllamaClient(host=settings.ollama_base)
            logger.debug("Initialized Qwen-VL client via Ollama")
        except Exception as exc:  # pragma: no cover - optional best-effort dependency
            logger.warning(f"Qwen-VL analysis disabled (client error): {exc}")
            self._vlm_client = None
            self._vlm_unavailable = True

        return self._vlm_client

    def _render_bbox_to_png(self, page_num: int, bbox: BoundingBox) -> Optional[bytes]:
        """Render a bounding box region to PNG bytes for vision analysis."""
        try:
            page = self.fitz_doc[page_num - 1]
        except Exception as exc:
            logger.warning(f"Failed to access page {page_num} for vision crop: {exc}")
            return None

        page_rect = page.rect
        rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

        padded = fitz.Rect(
            max(page_rect.x0, rect.x0 - VLM_RENDER_PADDING),
            max(page_rect.y0, rect.y0 - VLM_RENDER_PADDING),
            min(page_rect.x1, rect.x1 + VLM_RENDER_PADDING),
            min(page_rect.y1, rect.y1 + VLM_RENDER_PADDING),
        )

        if padded.width <= 0 or padded.height <= 0:
            return None

        try:
            matrix = fitz.Matrix(VLM_RENDER_SCALE, VLM_RENDER_SCALE)
            pix = page.get_pixmap(matrix=matrix, clip=padded, alpha=False)
            return pix.tobytes("png")
        except Exception as exc:
            logger.warning(f"Failed to render crop on page {page_num}: {exc}")
            return None

    def _analyze_image_with_vlm(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Send a cropped region to Qwen-VL for analysis."""
        if not image_bytes:
            return None

        client = self._ensure_vlm_client()
        if client is None:
            return None

        try:
            encoded_image = base64.b64encode(image_bytes).decode("ascii")
        except Exception as exc:
            logger.warning(f"Failed to encode image for Qwen-VL: {exc}")
            return None

        try:
            response = client.chat(
                model=settings.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [encoded_image],
                    }
                ],
            )
        except Exception as exc:  # pragma: no cover - network dependency
            logger.warning(f"Qwen-VL analysis failed: {exc}")
            return None

        message = response.get("message") if isinstance(response, dict) else None
        if not message:
            return None

        content = message.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            return stripped or None

        return None

    def _build_table_prompt(self, table: Table) -> str:
        """Create prompt for table-focused vision analysis."""
        prompt_lines = [
            "You are examining a cropped table from a scientific paper.",
            "Summarize the table in 2-3 sentences, mentioning column/row semantics and any notable values.",
            "If the table is unreadable, say so explicitly.",
        ]

        if table.caption:
            prompt_lines.append(f"Paper caption: {table.caption}")

        return "\n".join(prompt_lines)

    def _build_figure_prompt(self, figure: Figure) -> str:
        """Create prompt for figure-focused vision analysis."""
        prompt_lines = [
            "You are examining a cropped figure from a scientific paper.",
            "Describe what the figure shows in up to 3 sentences, noting axes, trends, or key elements.",
            "If the figure is unclear, state that explicitly.",
        ]

        if figure.caption:
            prompt_lines.append(f"Paper caption: {figure.caption}")

        return "\n".join(prompt_lines)

    def _enrich_visual_elements_with_vlm(self, tables: List[Table], figures: List[Figure]) -> None:
        """Attach Qwen-VL analysis outputs to tables and figures."""
        if self._vlm_unavailable or (not tables and not figures):
            return

        if self._ensure_vlm_client() is None:
            return

        tables_enriched = self._enrich_tables_with_vlm(tables)
        figures_enriched = self._enrich_figures_with_vlm(figures)

        if tables_enriched or figures_enriched:
            self.extraction_methods.add(ExtractionMethod.QWEN_VL)

    def _enrich_tables_with_vlm(self, tables: List[Table]) -> bool:
        processed_any = False
        for idx, table in enumerate(tables):
            if idx >= VLM_MAX_TABLES:
                break
            if not table.bbox:
                continue

            image_bytes = self._render_bbox_to_png(table.page, table.bbox)
            if not image_bytes:
                continue

            prompt = self._build_table_prompt(table)
            summary = self._analyze_image_with_vlm(image_bytes, prompt)
            if summary:
                table.artifacts["qwen_vl_summary"] = summary
                table.artifacts["qwen_vl_prompt"] = prompt
                table.artifacts["qwen_vl_model"] = settings.vlm_model
                processed_any = True
        return processed_any

    def _enrich_figures_with_vlm(self, figures: List[Figure]) -> bool:
        processed_any = False
        for idx, figure in enumerate(figures):
            if idx >= VLM_MAX_FIGURES:
                break
            if not figure.bbox:
                continue

            image_bytes = self._render_bbox_to_png(figure.page, figure.bbox)
            if not image_bytes:
                continue

            prompt = self._build_figure_prompt(figure)
            description = self._analyze_image_with_vlm(image_bytes, prompt)
            if description:
                figure.artifacts["qwen_vl_description"] = description
                figure.artifacts["qwen_vl_prompt"] = prompt
                figure.artifacts["qwen_vl_model"] = settings.vlm_model
                processed_any = True
        return processed_any

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
            page_candidates = self._caption_candidates_for_page(page_index, page)
            if page_candidates:
                captions[page_index].extend(page_candidates)
        return captions

    def _caption_candidates_for_page(self, page_index: int, page: fitz.Page) -> List[CaptionCandidate]:
        blocks = self._safe_page_blocks(page_index, page)
        candidates: List[CaptionCandidate] = []
        for block in blocks:
            candidate = self._caption_candidate_from_block(page_index, block)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _safe_page_blocks(self, page_index: int, page: fitz.Page) -> List[Dict[str, Any]]:
        try:
            block_data = page.get_text("dict").get("blocks", [])
        except Exception as exc:
            logger.warning("Failed to collect captions on page %s: %s", page_index, exc)
            return []
        return block_data

    def _caption_candidate_from_block(
        self,
        page_index: int,
        block: Dict[str, Any],
    ) -> Optional[CaptionCandidate]:
        if block.get("type") != 0:
            return None

        bbox = block.get("bbox") or []
        if len(bbox) != 4:
            return None

        block_text = self._flatten_block_text(block)
        if not block_text or len(block_text) > 400:
            return None

        match = CAPTION_REGEX.search(block_text)
        if not match:
            return None

        if not self._is_caption_prefix(block_text):
            return None

        raw_type = match.group(1).lower()
        caption_type = "figure" if raw_type.startswith("fig") else "table"
        number = match.group(2)

        return CaptionCandidate(
            page=page_index,
            text=block_text,
            caption_type=caption_type,
            number=number,
            rect=fitz.Rect(bbox),
        )

    def _flatten_block_text(self, block: Dict[str, Any]) -> str:
        parts: List[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    def _is_caption_prefix(self, text: str) -> bool:
        normalized = text.lower().lstrip()
        return any(
            normalized.startswith(prefix)
            for prefix in ("figure", "fig", "table", "tbl")
        )

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
        """Parse the PDF and build a unified representation."""
        logger.info("Parsing PDF: %s", self.pdf_path.name)

        tracker = _ParseStepTracker()
        overall_start = time.perf_counter()

        metadata = tracker.wrap("Metadata", "metadata", self._extract_metadata)
        pages, ocr_pages = tracker.wrap("Pages", "pages", self._extract_pages_with_blocks)
        raw_page_texts = [page.text for page in pages]
        sections = tracker.wrap("Sections", "sections", self._extract_sections_from_blocks, pages)

        layout_by_page, captions_by_page = self._prepare_layout_and_captions(tracker)
        tables = self._extract_tables_with_timing(layout_by_page, captions_by_page, tracker)
        figures = self._extract_figures_with_timing(layout_by_page, captions_by_page, tracker)
        self._maybe_enrich_visuals(tables, figures, tracker)
        equations = self._extract_equations_with_timing(tracker)
        references = self._extract_references_with_timing(pages, tracker)

        self._update_metadata_page_types(metadata, pages)
        udr = self._assemble_udr(
            metadata=metadata,
            pages=pages,
            raw_page_texts=raw_page_texts,
            sections=sections,
            tables=tables,
            figures=figures,
            equations=equations,
            references=references,
            ocr_pages=ocr_pages,
        )

        total_duration = time.perf_counter() - overall_start
        tracker.timings["total"] = total_duration
        logger.info("Total timings (s) | total=%.2f", total_duration)
        self._log_parse_summary(metadata, sections, tables, pages)

        return udr

    def _prepare_layout_and_captions(
        self,
        tracker: _ParseStepTracker,
    ) -> Tuple[Dict[int, List[LayoutDetection]], Dict[int, List[CaptionCandidate]]]:
        should_analyze_layout = (
            self.options.enable_layout_analysis
            or self.options.enable_table_detection
            or self.options.enable_figure_detection
        )
        if not should_analyze_layout:
            tracker.skip("Layout", "layout")
            tracker.skip("Captions", "captions")
            return {}, {}

        layout_by_page: Dict[int, List[LayoutDetection]] = {}
        if self.options.enable_layout_analysis:
            layout_by_page = tracker.wrap("Layout", "layout", self._analyze_document_layout)
        else:
            tracker.skip("Layout", "layout")

        captions_by_page: Dict[int, List[CaptionCandidate]] = {}
        if self.options.enable_table_detection or self.options.enable_figure_detection:
            captions_by_page = tracker.wrap("Captions", "captions", self._collect_captions)
        else:
            tracker.skip("Captions", "captions")

        return layout_by_page, captions_by_page

    def _extract_tables_with_timing(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
        tracker: _ParseStepTracker,
    ) -> List[Table]:
        if not self.options.enable_table_detection:
            tracker.skip("Tables", "tables")
            return []
        return tracker.wrap("Tables", "tables", self._extract_tables, layout_by_page, captions_by_page)

    def _extract_figures_with_timing(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
        tracker: _ParseStepTracker,
    ) -> List[Figure]:
        if not self.options.enable_figure_detection:
            tracker.skip("Figures", "figures")
            return []
        return tracker.wrap("Figures", "figures", self._extract_figures, layout_by_page, captions_by_page)

    def _maybe_enrich_visuals(
        self,
        tables: List[Table],
        figures: List[Figure],
        tracker: _ParseStepTracker,
    ) -> None:
        if not (self.options.enable_vlm_enrichment and (tables or figures)):
            tracker.skip("VLM enrichment", "vlm_enrichment")
            return
        tracker.wrap("VLM enrichment", "vlm_enrichment", self._enrich_visual_elements_with_vlm, tables, figures)

    def _extract_equations_with_timing(self, tracker: _ParseStepTracker) -> List[Equation]:
        if not self.options.enable_equation_detection:
            tracker.skip("Equations", "equations")
            return []
        return tracker.wrap("Equations", "equations", self._extract_equations)

    def _extract_references_with_timing(
        self,
        pages: List[Page],
        tracker: _ParseStepTracker,
    ) -> List[Reference]:
        if not self.options.enable_reference_detection:
            tracker.skip("References", "references")
            return []
        return tracker.wrap("References", "references", self._extract_references, pages)

    def _update_metadata_page_types(self, metadata: DocumentMetadata, pages: List[Page]) -> None:
        metadata.num_digital_pages = sum(1 for page in pages if page.page_type == PageType.DIGITAL)
        metadata.num_scanned_pages = sum(1 for page in pages if page.page_type == PageType.SCANNED)
        metadata.num_mixed_pages = sum(1 for page in pages if page.page_type == PageType.MIXED)

    def _assemble_udr(
        self,
        *,
        metadata: DocumentMetadata,
        pages: List[Page],
        raw_page_texts: List[str],
        sections: List[Section],
        tables: List[Table],
        figures: List[Figure],
        equations: List[Equation],
        references: List[Reference],
        ocr_pages: List[int],
    ) -> UnifiedDocumentRepresentation:
        return UnifiedDocumentRepresentation(
            metadata=metadata,
            pages=pages,
            raw_page_texts=raw_page_texts,
            sections=sections,
            tables=tables,
            figures=figures,
            equations=equations,
            references=references,
            extraction_methods_used=list(self.extraction_methods),
            ocr_pages=ocr_pages,
        )

    def _log_parse_summary(
        self,
        metadata: DocumentMetadata,
        sections: List[Section],
        tables: List[Table],
        pages: List[Page],
    ) -> None:
        block_total = sum(len(page.blocks) for page in pages)
        logger.info(
            "Parsed %s pages (%s digital, %s scanned, %s mixed), %s sections, %s tables, %s blocks",
            metadata.num_pages,
            metadata.num_digital_pages,
            metadata.num_scanned_pages,
            metadata.num_mixed_pages,
            len(sections),
            len(tables),
            block_total,
        )
    
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
        logger.debug("Extracting %s pages with blocks", len(self.fitz_doc))

        pages: List[Page] = []
        ocr_pages: List[int] = []

        for page_num, fitz_page in enumerate(self.fitz_doc, start=1):
            page, ocr_applied = self._extract_single_page(page_num, fitz_page)
            pages.append(page)
            if ocr_applied:
                ocr_pages.append(page_num)

        self.extraction_methods.add(ExtractionMethod.PYMUPDF)
        return pages, ocr_pages

    def _extract_single_page(self, page_num: int, fitz_page: fitz.Page) -> Tuple[Page, bool]:
        try:
            page_type = self._detect_page_type(fitz_page)
            blocks, ocr_applied = self._extract_blocks_for_page(fitz_page, page_type, page_num)
            ocr_confidence = self._average_block_confidence(blocks) if ocr_applied else None
            page = self._build_page_model(page_num, page_type, fitz_page, blocks, ocr_applied, ocr_confidence)
            self._log_page_extract_debug(page_num, page_type, blocks, page.text)
            return page, ocr_applied
        except Exception as exc:
            logger.warning("Error extracting page %s: %s", page_num, exc)
            return self._empty_page(page_num), False

    def _extract_blocks_for_page(
        self,
        fitz_page: fitz.Page,
        page_type: PageType,
        page_num: int,
    ) -> Tuple[List[TextBlock], bool]:
        if page_type == PageType.SCANNED:
            blocks, ocr_applied = self._extract_blocks_pymupdf(fitz_page, page_num)
            logger.warning("Page %s is scanned but OCR not yet implemented", page_num)
            return blocks, ocr_applied
        return self._extract_blocks_pymupdf(fitz_page, page_num)

    def _average_block_confidence(self, blocks: List[TextBlock]) -> Optional[float]:
        confidences = [block.confidence for block in blocks if block.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def _build_page_model(
        self,
        page_num: int,
        page_type: PageType,
        fitz_page: fitz.Page,
        blocks: List[TextBlock],
        ocr_applied: bool,
        ocr_confidence: Optional[float],
    ) -> Page:
        rect = fitz_page.rect
        page_text = "\n\n".join(block.text for block in blocks if block.text)
        extraction_method = ExtractionMethod.PADDLEOCR if ocr_applied else ExtractionMethod.PYMUPDF
        return Page(
            page_num=page_num,
            page_type=page_type,
            width=rect.width,
            height=rect.height,
            text=page_text,
            blocks=blocks,
            extraction_method=extraction_method,
            ocr_applied=ocr_applied,
            ocr_confidence=ocr_confidence,
        )

    def _log_page_extract_debug(
        self,
        page_num: int,
        page_type: PageType,
        blocks: List[TextBlock],
        page_text: str,
    ) -> None:
        logger.debug(
            "Page %s: %s, %s blocks, %s chars",
            page_num,
            page_type.value,
            len(blocks),
            len(page_text),
        )

    def _empty_page(self, page_num: int) -> Page:
        return Page(
            page_num=page_num,
            page_type=PageType.DIGITAL,
            width=612.0,
            height=792.0,
            text="",
            blocks=[],
            extraction_method=ExtractionMethod.PYMUPDF,
            ocr_applied=False,
        )
    
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

        if self._is_heading_block(spans):
            return "heading"

        if self._is_list_block(text):
            return "list"

        if self._is_caption_block(text):
            return "caption"

        return "paragraph"

    def _is_heading_block(self, spans: List[TextSpan]) -> bool:
        avg_font_size = self._average_font_size(spans)
        return avg_font_size > 14 and self._mostly_bold(spans)

    def _average_font_size(self, spans: List[TextSpan]) -> float:
        font_sizes = [span.font_size for span in spans if span.font_size]
        if not font_sizes:
            return 12.0
        return sum(font_sizes) / len(font_sizes)

    def _mostly_bold(self, spans: List[TextSpan]) -> bool:
        if not spans:
            return False
        bold_count = sum(1 for span in spans if span.is_bold)
        return (bold_count / len(spans)) > 0.5

    def _is_list_block(self, text: str) -> bool:
        return bool(re.match(r'^[\d\-•·◦▪▫]+[\.\)]\s+', text))

    def _is_caption_block(self, text: str) -> bool:
        trimmed = text.strip()
        return len(trimmed) < 100 and trimmed.endswith(':')
    
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
        if not self.fitz_doc:
            return None

        try:
            first_page = self.fitz_doc[0]
            blocks = self._first_page_text_blocks(first_page)
            candidates = self._title_candidates_from_blocks(first_page, blocks)
            return self._select_best_title_candidate(candidates)
        except Exception as exc:
            logger.warning("Error extracting title: %s", exc)
            return None

    def _first_page_text_blocks(self, first_page: fitz.Page) -> List[Dict[str, Any]]:
        block_data = first_page.get_text("dict").get("blocks", [])
        return [block for block in block_data if block.get("type") == 0]

    def _title_candidates_from_blocks(
        self,
        first_page: fitz.Page,
        blocks: List[Dict[str, Any]],
    ) -> List[Tuple[float, str]]:
        candidates: List[Tuple[float, str]] = []
        page_height = first_page.rect.height
        cutoff = page_height * 0.3

        for block in blocks:
            bbox = block.get("bbox", [])
            if len(bbox) < 4 or bbox[1] >= cutoff:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = float(span.get("size", 0))
                    if text and size > 14 and len(text) > 10:
                        candidates.append((size, text))

        return candidates

    def _select_best_title_candidate(self, candidates: List[Tuple[float, str]]) -> Optional[str]:
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    
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
    
    def _extract_figures(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[Figure]:
        """Extract figures by combining raster images, layout detections, and captions."""
        logger.debug("Extracting figures with fused signals")

        figure_candidates = self._collect_figure_candidates(layout_by_page, captions_by_page)
        figures = self._build_figures_from_candidates(figure_candidates)
        logger.debug("Extracted %d figures total", len(figures))
        return figures

    def _collect_figure_candidates(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[FigureCandidate]:
        figure_candidates: List[FigureCandidate] = []
        self.extraction_methods.add(ExtractionMethod.PYMUPDF)

        for page_index, page in enumerate(self.fitz_doc):
            page_candidates = self._figure_candidates_for_page(
                page_index=page_index,
                page=page,
                layout_by_page=layout_by_page,
                captions_by_page=captions_by_page,
            )
            for candidate in page_candidates:
                self._merge_figure_candidate(figure_candidates, candidate)

        return figure_candidates

    def _figure_candidates_for_page(
        self,
        *,
        page_index: int,
        page: fitz.Page,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[FigureCandidate]:
        page_num = page_index + 1
        candidates: List[FigureCandidate] = []
        candidates.extend(
            self._figure_candidates_from_layout(page_num, layout_by_page, captions_by_page)
        )
        candidates.extend(
            self._figure_candidates_from_images(page, page_num, captions_by_page)
        )
        candidates.extend(
            self._figure_candidates_from_captions(page, page_num, captions_by_page)
        )
        return candidates
    
    def _merge_figure_candidate(
        self,
        figure_candidates: List[FigureCandidate],
        candidate: FigureCandidate,
    ) -> None:
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

    def _figure_candidates_from_layout(
        self,
        page_num: int,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[FigureCandidate]:
        candidates: List[FigureCandidate] = []
        for detection in layout_by_page.get(page_num, []):
            label = detection.label.lower()
            if "caption" in label:
                continue
            if label not in {"figure", "image", "chart", "graph", "picture"}:
                continue

            caption = self._match_caption(captions_by_page, page_num, detection.bbox, "figure")
            subtype = "chart" if label in {"chart", "graph"} else None

            candidate = FigureCandidate(
                page=page_num,
                bbox=detection.bbox,
                source="layout",
                confidence=detection.confidence,
                subtype=subtype,
                caption=caption,
            )
            candidates.append(candidate)
            self.extraction_methods.add(ExtractionMethod.PADDLEOCR)
        return candidates

    def _figure_candidates_from_images(
        self,
        page: fitz.Page,
        page_num: int,
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[FigureCandidate]:
        candidates: List[FigureCandidate] = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.warning("Failed to enumerate images on page %s: %s", page_num, exc)
            return candidates

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
                caption = self._match_caption(captions_by_page, page_num, img_rect, "figure")
                if img_area < 25000 and not caption:
                    continue

                candidate = FigureCandidate(
                    page=page_num,
                    bbox=img_rect,
                    source="image",
                    confidence=0.75,
                    caption=caption,
                )
                candidates.append(candidate)
            except Exception as exc:
                logger.warning("Error processing image %s on page %s: %s", img_index, page_num, exc)
        return candidates

    def _figure_candidates_from_captions(
        self,
        page: fitz.Page,
        page_num: int,
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[FigureCandidate]:
        figure_captions = [
            caption
            for caption in captions_by_page.get(page_num, [])
            if caption.caption_type == "figure"
        ]
        if not figure_captions:
            return []

        captions_by_page[page_num] = [
            caption
            for caption in captions_by_page.get(page_num, [])
            if caption not in figure_captions
        ]

        page_rect = page.rect
        candidates: List[FigureCandidate] = []
        for caption in figure_captions:
            expanded_rect = fitz.Rect(
                caption.rect.x0,
                max(0.0, caption.rect.y0 - 80.0),
                caption.rect.x1,
                min(page_rect.y1, caption.rect.y1 + 20.0),
            )
            candidates.append(
                FigureCandidate(
                    page=page_num,
                    bbox=expanded_rect,
                    source="caption",
                    confidence=0.45,
                    caption=caption,
                )
            )
            self.extraction_methods.add(ExtractionMethod.HEURISTIC)
        return candidates

    def _build_figures_from_candidates(
        self,
        figure_candidates: List[FigureCandidate],
    ) -> List[Figure]:
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

            figures.append(
                Figure(
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
            )
        return figures

    def _extract_tables(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[Table]:
        """Extract tables by fusing PDF primitives, layout detection, and captions."""
        logger.debug("Extracting tables with fused signals")
        table_candidates = self._collect_table_candidates(layout_by_page, captions_by_page)
        tables = self._build_tables_from_candidates(table_candidates)
        logger.debug("Extracted %d tables total", len(tables))
        return tables

    def _collect_table_candidates(
        self,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
    ) -> List[TableCandidate]:
        table_candidates: List[TableCandidate] = []
        self.extraction_methods.add(ExtractionMethod.PDFPLUMBER)
        table_settings = self._table_detection_settings()

        for page_num, plumber_page in enumerate(self.pdfplumber_doc.pages, start=1):
            page_candidates = self._table_candidates_for_page(
                page_num=page_num,
                plumber_page=plumber_page,
                layout_by_page=layout_by_page,
                captions_by_page=captions_by_page,
                table_settings=table_settings,
            )
            for candidate in page_candidates:
                self._merge_table_candidate(table_candidates, candidate)

        return table_candidates

    def _table_candidates_for_page(
        self,
        *,
        page_num: int,
        plumber_page: pdfplumber.page.Page,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
        table_settings: Dict[str, object],
    ) -> List[TableCandidate]:
        candidates: List[TableCandidate] = []
        candidates.extend(
            self._table_candidates_from_pdfplumber(
                page_num,
                plumber_page,
                captions_by_page,
                table_settings,
            )
        )
        candidates.extend(
            self._table_candidates_from_layout(
                page_num,
                plumber_page,
                layout_by_page,
                captions_by_page,
                table_settings,
            )
        )
        return candidates

    def _table_detection_settings(self) -> Dict[str, object]:
        return {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        }

    def _table_candidates_from_pdfplumber(
        self,
        page_num: int,
        plumber_page: pdfplumber.page.Page,
        captions_by_page: Dict[int, List[CaptionCandidate]],
        table_settings: Dict[str, object],
    ) -> List[TableCandidate]:
        candidates: List[TableCandidate] = []
        try:
            detected_tables = plumber_page.find_tables(table_settings)
        except Exception as exc:
            logger.warning("pdfplumber failed on page %s: %s", page_num, exc)
            detected_tables = []

        for table_obj in detected_tables or []:
            try:
                data = table_obj.extract()
            except Exception:
                data = None

            normalized = self._normalize_table_data(data)
            bbox = fitz.Rect(*table_obj.bbox)
            caption = self._match_caption(captions_by_page, page_num, bbox, "table")

            candidates.append(
                TableCandidate(
                    page=page_num,
                    bbox=bbox,
                    data=normalized,
                    extraction_method=ExtractionMethod.PDFPLUMBER,
                    confidence=0.8,
                    caption=caption,
                )
            )
        return candidates

    def _table_candidates_from_layout(
        self,
        page_num: int,
        plumber_page: pdfplumber.page.Page,
        layout_by_page: Dict[int, List[LayoutDetection]],
        captions_by_page: Dict[int, List[CaptionCandidate]],
        table_settings: Dict[str, object],
    ) -> List[TableCandidate]:
        candidates: List[TableCandidate] = []
        for detection in layout_by_page.get(page_num, []):
            if detection.label not in {"table", "table caption"}:
                continue

            self.extraction_methods.add(ExtractionMethod.PADDLEOCR)
            caption = self._match_caption(captions_by_page, page_num, detection.bbox, "table")

            try:
                cropped = plumber_page.crop((detection.bbox.x0, detection.bbox.y0, detection.bbox.x1, detection.bbox.y1))
                data = cropped.extract_table(table_settings=table_settings)
            except Exception:
                data = None

            normalized = self._normalize_table_data(data)

            candidates.append(
                TableCandidate(
                    page=page_num,
                    bbox=detection.bbox,
                    data=normalized,
                    extraction_method=ExtractionMethod.PADDLEOCR,
                    confidence=detection.confidence,
                    caption=caption,
                )
            )
        return candidates

    def _merge_table_candidate(
        self,
        table_candidates: List[TableCandidate],
        candidate: TableCandidate,
    ) -> None:
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

    def _normalize_table_data(
        self,
        table_data: Optional[List[List[str]]],
    ) -> List[List[str]]:
        cleaned: List[List[str]] = []
        if not table_data:
            return cleaned

        for row in table_data:
            if row is None:
                continue
            cleaned.append(["" if cell is None else str(cell) for cell in row])
        return cleaned

    def _build_tables_from_candidates(
        self,
        table_candidates: List[TableCandidate],
    ) -> List[Table]:
        tables: List[Table] = []
        for idx, candidate in enumerate(table_candidates):
            if not self._is_valid_table_candidate(candidate):
                continue

            normalized_data = self._normalize_table_data(candidate.data)
            text_lines = self._table_text_lines(candidate, normalized_data)

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
            tables.append(
                self._create_table_from_candidate(
                    idx,
                    candidate,
                    normalized_data,
                    table_text,
                    has_structured_shape,
                )
            )
        return tables

    def _is_valid_table_candidate(self, candidate: TableCandidate) -> bool:
        rect_width = candidate.bbox.width
        rect_height = candidate.bbox.height

        if rect_width < MIN_TABLE_WIDTH or rect_height < MIN_TABLE_HEIGHT:
            logger.debug(
                "Skipping table candidate on page %s (width=%.1f, height=%.1f)",
                candidate.page,
                rect_width,
                rect_height,
            )
            return False

        if rect_height == 0:
            return False

        aspect_ratio = rect_width / rect_height
        if aspect_ratio < MIN_TABLE_ASPECT_RATIO or aspect_ratio > MAX_TABLE_ASPECT_RATIO:
            logger.debug(
                "Skipping table candidate on page %s due to aspect ratio %.2f",
                candidate.page,
                aspect_ratio,
            )
            return False
        return True

    def _table_text_lines(
        self,
        candidate: TableCandidate,
        normalized_data: List[List[str]],
    ) -> List[str]:
        if normalized_data:
            return [" | ".join(row) for row in normalized_data]

        try:
            page = self.fitz_doc[candidate.page - 1]
            clip_text = page.get_text("text", clip=candidate.bbox)
            return [line.strip() for line in clip_text.splitlines() if line.strip()]
        except Exception:
            return []

    def _create_table_from_candidate(
        self,
        index: int,
        candidate: TableCandidate,
        normalized_data: List[List[str]],
        table_text: str,
        has_structured_shape: bool,
    ) -> Table:
        bbox_model = BoundingBox(
            x0=candidate.bbox.x0,
            y0=candidate.bbox.y0,
            x1=candidate.bbox.x1,
            y1=candidate.bbox.y1,
            page=candidate.page,
        )

        return Table(
            table_id=f"tbl_{index}",
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

    def _extract_equations(self) -> List[Equation]:
        """Extract display equations using text layout heuristics."""
        logger.debug("Extracting equations with layout heuristics")

        latex_ocr = self._ensure_latex_ocr()
        has_latex_ocr = latex_ocr is not None
        self._register_equation_methods(has_latex_ocr)

        math_symbols = self._math_symbol_set()
        equations = self._collect_equations(math_symbols, latex_ocr, has_latex_ocr)

        logger.debug("Extracted %d equations total", len(equations))
        return equations

    def _register_equation_methods(self, has_latex_ocr: bool) -> None:
        self.extraction_methods.add(ExtractionMethod.PYMUPDF)
        if has_latex_ocr:
            self.extraction_methods.add(ExtractionMethod.QWEN_VL)

    def _math_symbol_set(self) -> set:
        return set("=±×÷∑∫∏√∞∂πθλμσΩαβγδ")

    def _collect_equations(
        self,
        math_symbols: set,
        latex_ocr,
        has_latex_ocr: bool,
    ) -> List[Equation]:
        equations: List[Equation] = []
        for page_index, page in enumerate(self.fitz_doc):
            equations.extend(
                self._equations_from_page(
                    page_index,
                    page,
                    latex_ocr,
                    has_latex_ocr,
                    math_symbols,
                )
            )
        return equations
    
    def _equations_from_page(
        self,
        page_index: int,
        page: fitz.Page,
        latex_ocr,
        has_latex_ocr: bool,
        math_symbols: set,
    ) -> List[Equation]:
        page_number = page_index + 1
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception as exc:
            logger.warning(
                "Failed to extract text blocks for equations on page %s: %s",
                page_number,
                exc,
            )
            return []

        equations: List[Equation] = []
        equation_idx = 0

        for block_idx, block in enumerate(blocks):
            if not self._equation_block_is_candidate(block, blocks, block_idx, math_symbols):
                continue

            block_rect = fitz.Rect(block.get("bbox"))
            block_text = self._block_text(block)
            is_display = self._is_display_equation(block, blocks, block_idx)

            equation_id = f"eq_p{page_number}_{equation_idx}"
            equation_idx += 1

            latex_str, latex_source, confidence = self._run_latex_ocr(
                page,
                block_rect,
                equation_id,
                latex_ocr,
                has_latex_ocr,
            )

            equations.append(
                Equation(
                    equation_id=equation_id,
                    latex=latex_str,
                    text=block_text,
                    page=page_number,
                    bbox=BoundingBox(
                        x0=block_rect.x0,
                        y0=block_rect.y0,
                        x1=block_rect.x1,
                        y1=block_rect.y1,
                        page=page_number,
                    ),
                    is_inline=not is_display,
                    image_s3_key=None,
                    extraction_method=ExtractionMethod.PYMUPDF,
                    latex_source=latex_source,
                    confidence=confidence,
                )
            )
        return equations

    def _block_text(self, block: dict) -> str:
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
        return block_text.strip()

    def _equation_block_is_candidate(
        self,
        block: dict,
        all_blocks: List[dict],
        block_idx: int,
        math_symbols: set,
    ) -> bool:
        if block.get("type") != 0:
            return False

        bbox_list = block.get("bbox", [])
        if len(bbox_list) < 4:
            return False

        block_text = self._block_text(block)
        if not block_text:
            return False

        lower_text = block_text.lower()
        common_words = ["the", "and", "for", "are", "that", "this", "with", "from", "where", "which"]
        if sum(1 for word in common_words if word in lower_text) >= 3:
            return False

        has_math = any(char in math_symbols for char in block_text)
        has_latex_pattern = bool(re.search(r"[∑∫∏√]|[α-ω]|=", block_text))
        is_display = self._is_display_equation(block, all_blocks, block_idx)
        is_small_centered = is_display and len(block_text) < 200

        return is_small_centered and (has_math or has_latex_pattern)

    def _run_latex_ocr(
        self,
        page: fitz.Page,
        block_rect: fitz.Rect,
        equation_id: str,
        latex_ocr,
        has_latex_ocr: bool,
    ) -> Tuple[Optional[str], Optional[ExtractionMethod], Optional[float]]:
        if not has_latex_ocr:
            return None, None, None

        try:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=block_rect)
            image_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(image_bytes))
            try:
                latex_str = latex_ocr(img)
            finally:
                img.close()
            logger.debug("Extracted equation %s: %s...", equation_id, latex_str[:50])
            return latex_str, ExtractionMethod.QWEN_VL, 0.8
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Error converting equation %s to LaTeX: %s", equation_id, exc)
            return None, None, None
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
    
    def _extract_references(self, pages: List[Page]) -> List[Reference]:
        """Extract reference entries from detected reference section."""
        logger.debug("Extracting references")

        accumulator = _ReferenceAccumulator()

        for page in pages:
            section_finished = self._process_reference_page(page, accumulator)
            if section_finished:
                break

        accumulator.finalize_current()

        if accumulator.references:
            self.extraction_methods.add(ExtractionMethod.HEURISTIC)

        logger.debug("Extracted %d reference(s)", len(accumulator.references))
        return accumulator.references

    def _process_reference_page(self, page: Page, accumulator: _ReferenceAccumulator) -> bool:
        for block in page.blocks:
            block_text = (block.text or "").strip()
            if not block_text:
                continue

            if block.block_type == "heading":
                if self._is_reference_heading(block_text):
                    logger.debug("Detected references section heading: '%s'", block_text)
                    accumulator.start_section()
                    continue
                if accumulator.collecting:
                    accumulator.end_section()
                    return True
                continue

            if not accumulator.collecting:
                continue

            if block.block_type not in {"paragraph", "list"}:
                continue

            accumulator.add_block(page.page_num, block_text)
        return False

    def _is_reference_heading(self, text: str) -> bool:
        return bool(REFERENCE_HEADING_PATTERN.search(text))


def parse_pdf(pdf_path: Path, options: Optional[ParseOptions] = None) -> UnifiedDocumentRepresentation:
    """
    Convenience function to parse a PDF file.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        UnifiedDocumentRepresentation
    """
    with PDFParser(pdf_path, options=options) as parser:
        return parser.parse()
