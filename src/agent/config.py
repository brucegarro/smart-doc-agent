"""Configuration management using pydantic-settings."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Database
    db_host: str = "db"
    db_port: int = 5432
    db_user: str = "doc"
    db_password: str = "doc"
    db_name: str = "docdb"
    
    # Object Storage (MinIO/S3)
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_bucket: str = "doc-bucket"
    s3_region: str = "us-east-1"
    
    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_queue_ingest: str = "q:ingest"
    redis_queue_index: str = "q:index"
    
    # Models - Ollama
    ollama_base: str = "http://ollama:11434"
    text_llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    vlm_model: str = "qwen2-vl:7b-instruct-q4_K_M"
    
    # Embedder (local)
    embedder_provider: str = "local"
    embedder_model: str = "BAAI/bge-small-en-v1.5" # Consider switching to SciBERT-based sentence transformers for scientific texts
    embedder_device: Literal["cpu", "mps", "cuda"] = "mps"
    embedder_normalize: bool = True
    
    # OCR
    ocr_engine: str = "paddleocr"
    ocr_lang: str = "en"
    
    # Application
    log_level: str = "INFO"
    
    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def redis_url(self) -> str:
        """Build Redis connection string."""
        return f"redis://{self.redis_host}:{self.redis_port}"


# Global settings instance
settings = Settings()
