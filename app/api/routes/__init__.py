"""API routes initialization."""

from app.api.routes.cv import router as cv_router
from app.api.routes.search import router as search_router
from app.api.routes.candidates import router as candidates_router
from app.api.routes.chat import router as chat_router

__all__ = ["cv_router", "search_router", "candidates_router", "chat_router"]
