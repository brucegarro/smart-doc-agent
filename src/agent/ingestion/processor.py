"""Document processing orchestrator for PDF ingestion pipeline."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from agent.config import settings
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
        
        try:
            # 1. Parse PDF
            logger.info("Step 1/3: Parsing PDF...")
            udr = parse_pdf(pdf_path)
            
            # 2. Upload to S3
            logger.info("Step 2/3: Uploading to S3...")
            s3_key = f"pdfs/{doc_id}/{pdf_path.name}"
            self.s3_client.upload_file(
                file_path=pdf_path,
                object_name=s3_key
            )
            logger.debug(f"Uploaded to S3: {s3_key}")
            
            # 3. Store in database
            logger.info("Step 3/3: Storing in database...")
            self._insert_document(
                doc_id=doc_id,
                pdf_hash=pdf_hash,
                filename=pdf_path.name,
                s3_key=s3_key,
                udr=udr,
                source_name=source_name
            )
            
            logger.info(f"✓ Successfully processed: {pdf_path.name} → {doc_id}")
            
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
                        source,
                        udr_data,
                        status,
                        ingested_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                        source_name,
                        json.dumps(udr_json),
                        "ingested",  # Status: ingested (not yet indexed)
                        datetime.now(timezone.utc)
                    )
                )
                
                conn.commit()
                
                logger.debug(f"Inserted document {doc_id} into database")
    
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
