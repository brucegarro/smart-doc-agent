"""Document processing orchestrator for PDF ingestion pipeline."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

from psycopg.types.json import Json

from agent.config import settings
from agent.embedding import ChunkPayload, build_chunks, embedding_client
from agent.db import get_db_connection
from agent.ingestion.pdf_parser import ParseOptions, parse_pdf
from agent.ingestion.udr import UnifiedDocumentRepresentation
from agent.storage import s3_client

logger = logging.getLogger(__name__)


@dataclass
class IngestionContext:
    pdf_path: Path
    pdf_hash: str
    doc_id: str
    options: ParseOptions
    source_name: Optional[str]


class DocumentProcessor:
    """Orchestrates PDF ingestion: parse → store → database."""
    
    def __init__(self, parse_options: Optional[ParseOptions] = None):
        self.s3_client = s3_client
        if parse_options is not None:
            self.parse_options = parse_options
        elif getattr(settings, "ingestion_fast_mode", True):
            self.parse_options = ParseOptions.fast_ingest()
        else:
            self.parse_options = ParseOptions()
    
    def process_pdf(
        self,
        pdf_path: Path,
        source_name: Optional[str] = None,
        parse_options: Optional[ParseOptions] = None,
    ) -> str:
        """
        Process a PDF file through the full ingestion pipeline.
        
        Args:
            pdf_path: Path to PDF file
            source_name: Optional source identifier (e.g., "arxiv", "pubmed")
        
        Returns:
            Document UUID
        
        Raises:
            ValueError: If PDF already exists in database
            Exception: If processing fails
        """
        context = self._build_context(pdf_path, source_name, parse_options)
        self._log_context(context)
        return self._ingest_document(context)

    def _build_context(
        self,
        pdf_path: Path,
        source_name: Optional[str],
        parse_options: Optional[ParseOptions],
    ) -> IngestionContext:
        normalized_path = self._normalize_pdf_path(pdf_path)
        options = self._resolve_parse_options(parse_options)
        pdf_hash = self._calculate_hash(normalized_path)
        self._guard_against_duplicates(pdf_hash, normalized_path)

        return IngestionContext(
            pdf_path=normalized_path,
            pdf_hash=pdf_hash,
            doc_id=str(uuid4()),
            options=options,
            source_name=source_name,
        )

    def _log_context(self, context: IngestionContext) -> None:
        logger.info("Processing PDF: %s", context.pdf_path.name)
        logger.debug("Parse options: %s", context.options)
        logger.debug("PDF hash: %s", context.pdf_hash)

    def _normalize_pdf_path(self, pdf_path: Path) -> Path:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return path

    def _resolve_parse_options(self, parse_options: Optional[ParseOptions]) -> ParseOptions:
        return parse_options or self.parse_options

    def _guard_against_duplicates(self, pdf_hash: str, pdf_path: Path) -> None:
        if self._document_exists(pdf_hash):
            raise ValueError(f"Document already exists: {pdf_path.name} (hash: {pdf_hash})")

    def _ingest_document(self, context: IngestionContext) -> str:
        s3_key: Optional[str] = None
        total_start = time.perf_counter()

        try:
            udr, parse_metrics = self._parse_document(context.pdf_path, context.options)
            s3_key, upload_elapsed = self._upload_pdf_to_storage(context.pdf_path, context.doc_id)
            db_elapsed = self._store_document_metadata(
                doc_id=context.doc_id,
                pdf_hash=context.pdf_hash,
                pdf_path=context.pdf_path,
                s3_key=s3_key,
                udr=udr,
                source_name=context.source_name,
            )
            final_status, chunk_stats, chunk_elapsed = self._run_chunk_pipeline(context.doc_id, udr)
            if final_status:
                self._update_document_status(context.doc_id, final_status)

            total_elapsed = time.perf_counter() - total_start
            self._log_processing_success(
                pdf_path=context.pdf_path,
                total_elapsed=total_elapsed,
                parse_metrics=parse_metrics,
                upload_elapsed=upload_elapsed,
                db_elapsed=db_elapsed,
                chunk_elapsed=chunk_elapsed,
                chunk_stats=chunk_stats,
            )
            logger.debug("Processed document id: %s", context.doc_id)
            return context.doc_id
        except Exception as exc:
            self._handle_processing_failure(context.pdf_path, s3_key, exc)
            raise

    def _handle_processing_failure(
        self,
        pdf_path: Path,
        s3_key: Optional[str],
        exc: Exception,
    ) -> None:
        logger.error("Failed to process %s: %s", pdf_path.name, exc, exc_info=True)
        if not s3_key:
            return
        try:
            if self.s3_client.object_exists(s3_key):
                self.s3_client.delete_object(s3_key)
                logger.debug("Cleaned up S3 object: %s", s3_key)
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to clean up S3 object: %s", cleanup_error)
    
    def _parse_document(
        self,
        pdf_path: Path,
        options: ParseOptions,
    ) -> Tuple[UnifiedDocumentRepresentation, Dict[str, float]]:
        logger.info("Step 1/4: Parsing PDF...")
        parse_start = time.perf_counter()
        udr = parse_pdf(pdf_path, options=options)
        parse_elapsed = time.perf_counter() - parse_start

        num_pages = getattr(udr.metadata, "num_pages", None)
        if num_pages:
            pages_per_second = num_pages / parse_elapsed if parse_elapsed else 0.0
            logger.info(
                "Parsed %s page(s) in %.2fs (%.2f pages/s)",
                num_pages,
                parse_elapsed,
                pages_per_second,
            )
        else:
            logger.info("Parsed PDF in %.2fs", parse_elapsed)

        return udr, {"elapsed": parse_elapsed, "pages": num_pages or 0}

    def _upload_pdf_to_storage(self, pdf_path: Path, doc_id: str) -> Tuple[str, float]:
        logger.info("Step 2/4: Uploading to S3...")
        s3_key = f"pdfs/{doc_id}/{pdf_path.name}"
        upload_start = time.perf_counter()
        self.s3_client.upload_file(file_path=pdf_path, object_name=s3_key)
        upload_elapsed = time.perf_counter() - upload_start
        logger.debug("Uploaded to S3: %s", s3_key)
        logger.info("Uploaded PDF to object storage in %.2fs", upload_elapsed)
        return s3_key, upload_elapsed

    def _store_document_metadata(
        self,
        *,
        doc_id: str,
        pdf_hash: str,
        pdf_path: Path,
        s3_key: str,
        udr: UnifiedDocumentRepresentation,
        source_name: Optional[str],
    ) -> float:
        logger.info("Step 3/4: Storing in database...")
        db_start = time.perf_counter()
        self._insert_document(
            doc_id=doc_id,
            pdf_hash=pdf_hash,
            filename=pdf_path.name,
            s3_key=s3_key,
            udr=udr,
            source_name=source_name,
        )
        db_elapsed = time.perf_counter() - db_start
        logger.info("Inserted document metadata in %.2fs", db_elapsed)
        return db_elapsed

    def _run_chunk_pipeline(
        self,
        doc_id: str,
        udr: UnifiedDocumentRepresentation,
    ) -> Tuple[Optional[str], Optional[Dict[str, float]], float]:
        logger.info("Step 4/4: Generating retrieval chunks and embeddings...")
        chunks_start = time.perf_counter()
        final_status, chunk_stats = self._create_chunks_and_embeddings(doc_id, udr)
        chunk_elapsed = time.perf_counter() - chunks_start

        if chunk_stats:
            logger.info(
                "Chunk pipeline: %s chunk(s) built in %.2fs; embeddings %.2fs; DB insert %.2fs",
                chunk_stats.get("count", 0),
                chunk_stats.get("build_elapsed", 0.0),
                chunk_stats.get("embed_elapsed", 0.0),
                chunk_stats.get("insert_elapsed", 0.0),
            )
        else:
            logger.info("Chunk pipeline completed in %.2fs", chunk_elapsed)

        return final_status, chunk_stats, chunk_elapsed

    def _log_processing_success(
        self,
        *,
        pdf_path: Path,
        total_elapsed: float,
        parse_metrics: Dict[str, float],
        upload_elapsed: float,
        db_elapsed: float,
        chunk_elapsed: float,
        chunk_stats: Optional[Dict[str, float]],
    ) -> None:
        pages = parse_metrics.get("pages") or (chunk_stats or {}).get("pages")
        if pages:
            per_page = total_elapsed / pages if pages else 0.0
            logger.info(
                "✓ Successfully processed %s in %.2fs (%.2fs/page)",
                pdf_path.name,
                total_elapsed,
                per_page,
            )
        else:
            logger.info("✓ Successfully processed %s in %.2fs", pdf_path.name, total_elapsed)

        logger.debug(
            "Timing breakdown - parse: %.2fs, upload: %.2fs, db: %.2fs, chunks: %.2fs",
            parse_metrics.get("elapsed", 0.0),
            upload_elapsed,
            db_elapsed,
            chunk_elapsed,
        )

    def _calculate_hash(self, pdf_path: Path) -> str:
        """Calculate SHA256 hash of PDF file."""
        sha256 = hashlib.sha256()
        
        with open(pdf_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _document_exists(self, pdf_hash: str) -> bool:
        """Check if document with given hash already exists."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE pdf_hash = %s LIMIT 1",
                    (pdf_hash,)
                )
                return cur.fetchone() is not None
    
    def _insert_document(
        self,
        doc_id: str,
        pdf_hash: str,
        filename: str,
        s3_key: str,
        udr: UnifiedDocumentRepresentation,
        source_name: Optional[str]
    ) -> None:
        """Insert document record into database."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Prepare UDR JSON
                udr_json = json.loads(udr.model_dump_json())
                
                # Insert document
                cur.execute(
                    """
                    INSERT INTO documents (
                        id,
                        filename,
                        pdf_hash,
                        s3_key,
                        title,
                        authors,
                        publication_year,
                        num_pages,
                        udr_data,
                        processing_status,
                        processed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        doc_id,
                        filename,
                        pdf_hash,
                        s3_key,
                        udr.metadata.title,
                        udr.metadata.authors,
                        udr.metadata.publication_year,
                        udr.metadata.num_pages,
                        json.dumps(udr_json),
                        "ingested",
                        datetime.now(timezone.utc)
                    )
                )
                
                conn.commit()
                
                logger.debug(f"Inserted document {doc_id} into database")
    
    def _create_chunks_and_embeddings(
        self,
        doc_id: str,
        udr: UnifiedDocumentRepresentation,
    ) -> tuple[Optional[str], Optional[dict[str, float | int]]]:
        """Build retrieval chunks, generate embeddings, and persist them with timing stats."""
        chunk_stats: dict[str, float | int] = {}
        try:
            build_start = time.perf_counter()
            chunks = build_chunks(udr)
            chunk_stats["build_elapsed"] = time.perf_counter() - build_start
            chunk_stats["count"] = len(chunks)
            chunk_stats["pages"] = getattr(udr.metadata, "num_pages", 0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to build chunks for %s: %s", doc_id, exc, exc_info=True)
            return None, None

        if not chunks:
            logger.warning("No retrieval chunks generated for document %s", doc_id)
            return "ingested", chunk_stats

        embed_start = time.perf_counter()
        embeddings_ready = self._generate_embeddings(chunks)
        chunk_stats["embed_elapsed"] = time.perf_counter() - embed_start

        insert_start = time.perf_counter()
        self._insert_chunks(doc_id, chunks)
        chunk_stats["insert_elapsed"] = time.perf_counter() - insert_start

        return ("indexed" if embeddings_ready else "chunked"), chunk_stats

    def _generate_embeddings(self, chunks: list[ChunkPayload]) -> bool:
        """Populate in-memory chunk payloads with embeddings."""
        if not chunks:
            return False

        contents = [chunk.content for chunk in chunks]
        vectors = embedding_client.embed(contents)
        if vectors is None:
            logger.warning("Embedding client unavailable; storing chunks without vectors")
            return False

        if len(vectors) != len(chunks):
            logger.warning(
                "Embedding client returned %s vectors for %s chunks",
                len(vectors),
                len(chunks),
            )
            return False

        assigned = 0
        for chunk, vector in zip(chunks, vectors):
            if vector:
                chunk.embedding = vector
                assigned += 1

        if assigned == 0:
            logger.warning("No embeddings generated (all chunks empty?)")
            return False

        logger.debug("Generated embeddings for %s chunk(s)", assigned)
        return True

    def _insert_chunks(self, doc_id: str, chunks: list[ChunkPayload]) -> None:
        """Persist chunk payloads into the chunks table."""
        if not chunks:
            return

        records = []
        for chunk in chunks:
            vector = chunk.embedding if chunk.embedding else None
            records.append(
                (
                    doc_id,
                    chunk.content,
                    chunk.content_type,
                    chunk.chunk_index,
                    chunk.page_number,
                    chunk.section_title,
                    chunk.token_count,
                    chunk.char_count,
                    vector,
                    Json(chunk.metadata),
                )
            )

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        document_id,
                        content,
                        content_type,
                        chunk_index,
                        page_number,
                        section_title,
                        token_count,
                        char_count,
                        embedding,
                        metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    records,
                )
                conn.commit()

        logger.info("Persisted %s chunk(s) for document %s", len(records), doc_id)

    def _update_document_status(self, doc_id: str, status: str) -> None:
        """Update the processing status for a document."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if status == "indexed":
                    cur.execute(
                        """
                        UPDATE documents
                        SET processing_status = %s,
                            processed_at = COALESCE(processed_at, NOW())
                        WHERE id = %s
                        """,
                        (status, doc_id),
                    )
                else:
                    cur.execute(
                        "UPDATE documents SET processing_status = %s WHERE id = %s",
                        (status, doc_id),
                    )

                conn.commit()

    def get_document_status(self, doc_id: str) -> Optional[dict]:
        """
        Get processing status of a document.
        
        Returns:
            Dict with status info, or None if not found
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        filename,
                        title,
                        status,
                        ingested_at,
                        indexed_at
                    FROM documents
                    WHERE id = %s
                    """,
                    (doc_id,)
                )
                
                row = cur.fetchone()
                
                if not row:
                    return None
                
                return {
                    "id": row[0],
                    "filename": row[1],
                    "title": row[2],
                    "status": row[3],
                    "ingested_at": row[4].isoformat() if row[4] else None,
                    "indexed_at": row[5].isoformat() if row[5] else None,
                }
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """
        List all documents in database.
        
        Args:
            limit: Max number of documents to return
            offset: Pagination offset
        
        Returns:
            List of document info dicts
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        filename,
                        title,
                        authors,
                        publication_year,
                        num_pages,
                        source,
                        status,
                        ingested_at
                    FROM documents
                    ORDER BY ingested_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )
                
                rows = cur.fetchall()
                
                documents = []
                for row in rows:
                    documents.append({
                        "id": row[0],
                        "filename": row[1],
                        "title": row[2],
                        "authors": row[3],
                        "publication_year": row[4],
                        "num_pages": row[5],
                        "source": row[6],
                        "status": row[7],
                        "ingested_at": row[8].isoformat() if row[8] else None,
                    })
                
                return documents


# Global processor instance
processor = DocumentProcessor()
