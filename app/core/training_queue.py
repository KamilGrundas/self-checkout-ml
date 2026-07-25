"""Persistent Redis/RQ queue for classifier training."""

from __future__ import annotations

from typing import Any

from redis import Redis
from rq import Queue

from app.core.config import settings

QUEUE_NAME = "classifier-training"
JOB_RETENTION_SECONDS = 7 * 24 * 60 * 60
JOB_TIMEOUT = "24h"


def get_redis() -> Redis:
    if not settings.TRAINING_QUEUE_URL:
        raise RuntimeError("Training queue is not configured")
    return Redis.from_url(
        settings.TRAINING_QUEUE_URL,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
    )


def get_training_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def initial_job_meta(total_epochs: int) -> dict[str, Any]:
    return {
        "stage": "queued",
        "message": "Training queued",
        "progress": 0.0,
        "current_epoch": None,
        "total_epochs": total_epochs,
        "metrics": None,
        "error": None,
    }
