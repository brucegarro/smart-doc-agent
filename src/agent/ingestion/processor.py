"""Document processing orchestrator for PDF ingestion pipeline."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from psycopg.types.json import Json

from agent.config import settings
from agent.embedding import ChunkPayload, build_chunks, embedding_client
from agent.db import get_db_connection
from agent.ingestion.pdf_parser import parse_pdf
from agent.ingestion.udr import UnifiedDocumentRepresentation
from agent.storage import s3_client

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates PDF ingestion: parse → store → database."""
    
    def __init__(self):
        self.s3_client = s3_client
    
    def process_pdf(self, pdf_path: Path, source_name: Optional[str] = None) -> str:
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
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Calculate PDF hash for deduplication
        pdf_hash = self._calculate_hash(pdf_path)
        logger.debug(f"PDF hash: {pdf_hash}")
        
        # Check if already processed
        if self._document_exists(pdf_hash):
            raise ValueError(f"Document already exists: {pdf_path.name} (hash: {pdf_hash})")
        
        # Generate document UUID
        doc_id = str(uuid4())
        logger.debug(f"Generated doc_id: {doc_id}")
        
        s3_key: Optional[str] = None

        total_start = time.perf_counter()
        try:
            # 1. Parse PDF
            logger.info("Step 1/3: Parsing PDF...")
            parse_start = time.perf_counter()
            udr = parse_pdf(pdf_path)
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
            
            # 2. Upload to S3
            logger.info("Step 2/3: Uploading to S3...")
            s3_key = f"pdfs/{doc_id}/{pdf_path.name}"
            upload_start = time.perf_counter()
            self.s3_client.upload_file(
                file_path=pdf_path,
                object_name=s3_key
            )
            logger.debug(f"Uploaded to S3: {s3_key}")
            upload_elapsed = time.perf_counter() - upload_start
            logger.info("Uploaded PDF to object storage in %.2fs", upload_elapsed)
            
            # 3. Store in database
            logger.info("Step 3/3: Storing in database...")
            db_start = time.perf_counter()
            self._insert_document(
                doc_id=doc_id,
                pdf_hash=pdf_hash,
                filename=pdf_path.name,
                s3_key=s3_key,
                udr=udr,
                source_name=source_name
            )
            db_elapsed = time.perf_counter() - db_start
            logger.info("Inserted document metadata in %.2fs", db_elapsed)
            
            # 4. Generate retrieval chunks + embeddings
            logger.info("Step 4/4: Generating retrieval chunks and embeddings...")
            chunks_start = time.perf_counter()
            final_status, chunk_stats = self._create_chunks_and_embeddings(doc_id, udr)
            chunks_elapsed = time.perf_counter() - chunks_start
            if chunk_stats:
                logger.info(
                    "Chunk pipeline: %s chunk(s) built in %.2fs; embeddings %.2fs; DB insert %.2fs",
                    chunk_stats.get("count", 0),
                    chunk_stats.get("build_elapsed", 0.0),
                    chunk_stats.get("embed_elapsed", 0.0),
                    chunk_stats.get("insert_elapsed", 0.0),
                )
            else:
                logger.info("Chunk pipeline completed in %.2fs", chunks_elapsed)
            if final_status:
                self._update_document_status(doc_id, final_status)

            total_elapsed = time.perf_counter() - total_start
            pages = num_pages
            if not pages and chunk_stats:
                pages = chunk_stats.get("pages")
            if pages:
                per_page = total_elapsed / pages
                logger.info(
                    "✓ Successfully processed %s in %.2fs (%.2fs/page)",
                    pdf_path.name,
                    total_elapsed,
                    per_page,
                )
            else:
                logger.info(
                    "✓ Successfully processed %s in %.2fs",
                    pdf_path.name,
                    total_elapsed,
                )
            logger.debug("Processed document id: %s", doc_id)
            
            return doc_id
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
            
            # Clean up S3 if upload succeeded but DB insert failed
            try:
                if s3_key and self.s3_client.object_exists(s3_key):
                    self.s3_client.delete_object(s3_key)
                    logger.debug(f"Cleaned up S3 object: {s3_key}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up S3 object: {cleanup_error}")
            
            raise
    
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
