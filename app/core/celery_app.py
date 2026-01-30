"""Celery application configuration."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "cv_screening",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # For heavy tasks
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,
    result_expires=3600 * 24,  # Results expire after 24 hours
)

# Task routes
celery_app.conf.task_routes = {
    "app.workers.tasks.process_cv": {"queue": "cv_processing"},
    "app.workers.tasks.generate_embeddings": {"queue": "embeddings"},
}
