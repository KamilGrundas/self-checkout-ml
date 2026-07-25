from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.api.deps import SuperuserDep
from app.core.training_queue import (
    JOB_RETENTION_SECONDS,
    JOB_TIMEOUT,
    get_training_queue,
    initial_job_meta,
)
from app.core.training_worker import run_training_job

router = APIRouter(prefix="/train", tags=["train"], dependencies=[SuperuserDep])

TrainingJobStatus = Literal["queued", "running", "completed", "failed"]


class TrainingJob(BaseModel):
    job_id: str
    status: TrainingJobStatus
    stage: str
    message: str
    progress: float = Field(ge=0, le=100)
    current_epoch: int | None = None
    total_epochs: int
    metrics: dict[str, float] | None = None
    result: dict | None = None
    error: str | None = None


class TrainRequest(BaseModel):
    yolo_datasets: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="S3-compatible object storage prefixes of YOLO datasets (crops)",
    )
    csv_datasets: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="S3-compatible object storage prefixes of CSV datasets (whole images)",
    )
    image_size: int = Field(
        default=160, ge=32, le=512, description="Input image size (square)"
    )
    epochs: int = Field(default=12, ge=1, le=500, description="Max training epochs")
    batch_size: int = Field(
        default=16, ge=1, le=1024, description="Training batch size"
    )
    validation_ratio: float = Field(
        default=0.2,
        ge=0,
        le=0.5,
        description="Fraction of data for validation",
    )


def _serialize_job(job: Job) -> TrainingJob:
    rq_status = job.get_status(refresh=True)
    if rq_status in {"finished"}:
        public_status: TrainingJobStatus = "completed"
    elif rq_status in {"failed", "stopped", "canceled"}:
        public_status = "failed"
    elif rq_status in {"started", "busy"}:
        public_status = "running"
    else:
        public_status = "queued"

    meta = job.meta or {}
    result = job.return_value() if public_status == "completed" else None
    error = meta.get("error")
    if public_status == "failed" and not error:
        error = "Training worker stopped before completing the job"

    return TrainingJob(
        job_id=job.id,
        status=public_status,
        stage=str(meta.get("stage") or public_status),
        message=str(meta.get("message") or f"Training {public_status}"),
        progress=float(meta.get("progress") or 0),
        current_epoch=meta.get("current_epoch"),
        total_epochs=int(meta.get("total_epochs") or 1),
        metrics=meta.get("metrics"),
        result=result,
        error=str(error) if error else None,
    )


@router.post(
    "/classifier",
    response_model=TrainingJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def classifier(body: TrainRequest) -> TrainingJob:
    """Persist and queue classifier training for a dedicated worker."""
    if not body.yolo_datasets and not body.csv_datasets:
        raise HTTPException(status_code=422, detail="At least one dataset is required")

    job_id = str(uuid4())
    try:
        job = get_training_queue().enqueue(
            run_training_job,
            body.model_dump(mode="json"),
            job_id=job_id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=JOB_RETENTION_SECONDS,
            failure_ttl=JOB_RETENTION_SECONDS,
            meta=initial_job_meta(body.epochs),
            description="Train product classifier",
        )
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Training queue is unavailable"
        ) from exc
    return _serialize_job(job)


@router.get("/classifier/{job_id}", response_model=TrainingJob)
async def classifier_status(job_id: str) -> TrainingJob:
    """Return durable progress and result data for a classifier training job."""
    try:
        job = Job.fetch(job_id, connection=get_training_queue().connection)
        return _serialize_job(job)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Training queue is unavailable"
        ) from exc
