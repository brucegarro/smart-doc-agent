"""S3/MinIO storage client for document and artifact storage."""

import io
import logging
from pathlib import Path
from typing import BinaryIO, Optional, Union

from minio import Minio
from minio.error import S3Error

from agent.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """MinIO/S3 client wrapper for document storage."""
    
    def __init__(self):
        # Parse endpoint to remove http:// prefix if present
        endpoint = settings.s3_endpoint.replace("http://", "").replace("https://", "")
        secure = settings.s3_endpoint.startswith("https://")
        
        self.client = Minio(
            endpoint=endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=secure,
            region=settings.s3_region
        )
        
        self.bucket = settings.s3_bucket
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                logger.info(f"Creating bucket: {self.bucket}")
                self.client.make_bucket(self.bucket, location=settings.s3_region)
            else:
                logger.debug(f"Bucket exists: {self.bucket}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            file_path: Local file path
            object_name: S3 object key (path in bucket)
            content_type: MIME type (auto-detected if None)
        
        Returns:
            S3 object key
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect content type if not provided
        if content_type is None:
            ext = file_path.suffix.lower()
            content_types = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".json": "application/json",
                ".txt": "text/plain",
            }
            content_type = content_types.get(ext, "application/octet-stream")
        
        try:
            logger.info(f"Uploading {file_path} to s3://{self.bucket}/{object_name}")
            
            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=str(file_path),
                content_type=content_type
            )
            
            logger.info(f"Successfully uploaded to s3://{self.bucket}/{object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload bytes data to S3.
        
        Args:
            data: Bytes to upload
            object_name: S3 object key
            content_type: MIME type
        
        Returns:
            S3 object key
        """
        try:
            logger.info(f"Uploading {len(data)} bytes to s3://{self.bucket}/{object_name}")
            
            data_stream = io.BytesIO(data)
            
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )
            
            logger.info(f"Successfully uploaded to s3://{self.bucket}/{object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Error uploading bytes: {e}")
            raise
    
    def download_file(self, object_name: str, file_path: Union[str, Path]) -> Path:
        """
        Download a file from S3.
        
        Args:
            object_name: S3 object key
            file_path: Local destination path
        
        Returns:
            Path to downloaded file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading s3://{self.bucket}/{object_name} to {file_path}")
            
            self.client.fget_object(
                bucket_name=self.bucket,
                object_name=object_name,
                file_path=str(file_path)
            )
            
            logger.info(f"Successfully downloaded to {file_path}")
            return file_path
            
        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    def download_bytes(self, object_name: str) -> bytes:
        """
        Download file content as bytes.
        
        Args:
            object_name: S3 object key
        
        Returns:
            File content as bytes
        """
        try:
            logger.info(f"Downloading s3://{self.bucket}/{object_name} as bytes")
            
            response = self.client.get_object(
                bucket_name=self.bucket,
                object_name=object_name
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"Successfully downloaded {len(data)} bytes")
            return data
            
        except S3Error as e:
            logger.error(f"Error downloading bytes: {e}")
            raise
    
    def object_exists(self, object_name: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False
    
    def delete_object(self, object_name: str):
        """Delete an object from S3."""
        try:
            logger.info(f"Deleting s3://{self.bucket}/{object_name}")
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"Successfully deleted {object_name}")
        except S3Error as e:
            logger.error(f"Error deleting object: {e}")
            raise


# Global S3 client instance
s3_client = S3Client()
