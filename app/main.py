"""
Smart CV Screening & Matching System - FastAPI Application.

Main entry point for the API server.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import cv_router, search_router, candidates_router, chat_router

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    # Load BM25 index from database at startup
    try:
        from app.services.search.hybrid import get_search_engine
        from app.core.database import SessionLocal
        
        search_engine = get_search_engine()
        db = SessionLocal()
        try:
            num_indexed = search_engine.refresh_bm25_index_sync(db)
            logger.info(f"BM25 index initialized with {num_indexed} documents")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to initialize BM25 index: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Smart CV Screening & Matching System

An AI-powered system for intelligent CV parsing, processing, and candidate matching.

### Features
- **Layout-Aware OCR**: Handle scanned PDFs and two-column layouts
- **Vietnamese NLP**: Word segmentation for better understanding
- **LLM Parsing**: GPT-4o powered data extraction
- **Hybrid Search**: BM25 + Vector search with RRF fusion
- **Job Matching**: AI-powered candidate-job matching

### Quick Start
1. Upload CVs via `/api/v1/cv/upload`
2. Search candidates via `/api/v1/search`
3. Match to job descriptions via `/api/v1/search/match`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Smart CV Screening & Matching System",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


# Include routers
app.include_router(cv_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(candidates_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
