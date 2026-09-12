from __future__ import annotations

import hashlib
import json
from typing import Any

from redis import Redis
from rq import Queue

from app.core.config import settings

QUEUE_NAME = "scale-autolabel"
JOB_RETENTION_SECONDS = 7 * 24 * 60 * 60
JOB_TIMEOUT = "30d"
IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


def get_redis() -> Redis:
    if not settings.TRAINING_QUEUE_URL:
        raise RuntimeError("Redis queue is not configured")
    return Redis.from_url(
        settings.TRAINING_QUEUE_URL,
        socket_connect_timeout=3,
        socket_timeout=5,
        health_check_interval=30,
    )


def get_autolabel_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def request_digest(object_names: list[str]) -> str:
    canonical = json.dumps(object_names, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def initial_job_meta(
    object_names: list[str],
    *,
    include_items: bool = True,
    provider_cancellation: str = "not_supported",
) -> dict[str, Any]:
    return {
        "status": "queued",
        "total": len(object_names),
        "completed": 0,
        "matched": 0,
        "unmatched": 0,
        "failed": 0,
        "cancelled": 0,
        "provider_cancellation": provider_cancellation,
        # Bulk batches intentionally omit items from the public response. Keep
        # just completed object names internally so the image list can mark
        # the remaining selection as processing.
        "completed_object_names": [],
        "items": [
            {
                "object_name": object_name,
                "status": "queued",
                "product_id": None,
                "product_name": None,
                "error": None,
            }
            for object_name in object_names
        ]
        if include_items
        else [],
    }
