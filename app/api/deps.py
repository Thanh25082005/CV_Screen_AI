"""API dependencies for route handlers."""

from typing import Generator, AsyncGenerator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, get_sync_db
from app.services.search.hybrid import HybridSearchEngine, get_search_engine
from app.services.embedding.embedder import EmbeddingService, get_embedding_service
from app.services.parsing.llm_parser import LLMParser, get_parser
from app.config import get_settings, Settings


def get_settings_dep() -> Settings:
    """Dependency to get application settings."""
    return get_settings()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session."""
    async for session in get_db():
        yield session


def get_sync_db_session() -> Generator[Session, None, None]:
    """Dependency to get sync database session."""
    for session in get_sync_db():
        yield session


def get_search_engine_dep() -> HybridSearchEngine:
    """Dependency to get search engine."""
    return get_search_engine()


def get_embedding_service_dep() -> EmbeddingService:
    """Dependency to get embedding service."""
    return get_embedding_service()


def get_llm_parser_dep() -> LLMParser:
    """Dependency to get LLM parser."""
    return get_parser()


def require_openai_key(settings: Settings = Depends(get_settings_dep)) -> Settings:
    """Dependency that requires OpenAI API key to be configured."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. LLM features unavailable.",
        )
    return settings
