"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',),
    )

    # Application
    app_name: str = "Smart CV Screening & Matching"
    app_version: str = "1.0.0"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/cvscreening"
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM Provider (groq or openai)
    llm_provider: str = "groq"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    model_cache_dir: str = "./model_cache"

    # OCR
    ocr_lang: str = "vi"
    ocr_use_gpu: bool = False

    # Upload settings
    upload_dir: str = "./uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # Search settings
    rrf_k: int = 60  # RRF constant
    search_top_k: int = 20

    # Chat settings
    chat_memory_ttl_seconds: int = 3600  # Session expiry (1 hour)
    chat_history_max_messages: int = 10  # Sliding window size
    chat_model: str = "llama-3.3-70b-versatile"  # Model for chat responses
    chat_max_candidates: int = 5  # Max candidates to include in context

    @property
    def async_database_url(self) -> str:
        """Convert sync database URL to async."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
