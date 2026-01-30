"""Celery workers module initialization."""

from app.workers.tasks import process_cv_task, generate_embeddings_task

__all__ = ["process_cv_task", "generate_embeddings_task"]
