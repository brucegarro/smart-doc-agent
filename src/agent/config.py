"""Configuration management compatible with Pydantic v1, v2, or a simple fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal

from pydantic import Field

try:  # Prefer the dedicated package used by Pydantic v2.
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore[misc]
    except Exception:
        BaseSettings = None  # type: ignore[assignment]
        SettingsConfigDict = None  # type: ignore[assignment]
        _SETTINGS_BACKEND = "fallback"
    else:
        SettingsConfigDict = dict  # type: ignore[assignment]
        _SETTINGS_BACKEND = "pydantic_v1"
else:
    _SETTINGS_BACKEND = "pydantic_v2"


if BaseSettings is not None:

    class Settings(BaseSettings):
        """Application settings loaded from environment variables (Pydantic-powered)."""

        if _SETTINGS_BACKEND == "pydantic_v2":
            model_config = SettingsConfigDict(
                env_file=".env",
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )
        else:
            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"
                case_sensitive = False
                extra = "ignore"

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
        embedder_model: str = "BAAI/bge-small-en-v1.5"
        embedder_device: Literal["cpu", "mps", "cuda"] = "mps"
        embedder_normalize: bool = True

        # OCR
        ocr_engine: str = "paddleocr"
        ocr_lang: str = "en"

        # Application
        log_level: str = "INFO"
        ingestion_fast_mode: bool = True
        ingest_queue_dir: Path = Field(default=Path("/data/ingest_queue"))

        @property
        def database_url(self) -> str:
            """Build PostgreSQL connection string."""
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

        @property
        def redis_url(self) -> str:
            """Build Redis connection string."""
            return f"redis://{self.redis_host}:{self.redis_port}"


else:

    def _load_env_file(path: str = ".env") -> Dict[str, str]:
        """Lightweight .env parser for fallback settings."""
        env_path = Path(path)
        if not env_path.is_file():
            return {}

        values: Dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            values[key.strip()] = value
        return values

    def _coerce(value: str, default):
        """Coerce environment strings into typed values."""
        if isinstance(default, bool):
            return value.lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int):
            return int(value)
        if isinstance(default, Path):
            return Path(value)
        return value

    class Settings:
        """Minimal settings loader used when pydantic-settings is unavailable."""

        _DEFAULTS: Dict[str, object] = {
            # Database
            "db_host": "db",
            "db_port": 5432,
            "db_user": "doc",
            "db_password": "doc",
            "db_name": "docdb",
            # Object Storage (MinIO/S3)
            "s3_endpoint": "http://minio:9000",
            "s3_access_key": "minio",
            "s3_secret_key": "minio123",
            "s3_bucket": "doc-bucket",
            "s3_region": "us-east-1",
            # Redis
            "redis_host": "redis",
            "redis_port": 6379,
            "redis_queue_ingest": "q:ingest",
            "redis_queue_index": "q:index",
            # Models - Ollama
            "ollama_base": "http://ollama:11434",
            "text_llm_model": "qwen2.5:7b-instruct-q4_K_M",
            "vlm_model": "qwen2-vl:7b-instruct-q4_K_M",
            # Embedder
            "embedder_provider": "local",
            "embedder_model": "BAAI/bge-small-en-v1.5",
            "embedder_device": "mps",
            "embedder_normalize": True,
            # OCR
            "ocr_engine": "paddleocr",
            "ocr_lang": "en",
            # Application
            "log_level": "INFO",
            "ingestion_fast_mode": True,
            "ingest_queue_dir": Path("/data/ingest_queue"),
        }

        def __init__(self) -> None:
            env_values = _load_env_file()

            for key, default in self._DEFAULTS.items():
                env_key = key.upper()
                raw_value = os.getenv(env_key, env_values.get(env_key, env_values.get(key)))
                if raw_value is None:
                    value = default
                else:
                    try:
                        value = _coerce(raw_value, default)
                    except Exception:
                        value = default
                setattr(self, key, value)

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
