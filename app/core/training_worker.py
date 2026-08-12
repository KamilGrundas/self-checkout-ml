"""RQ worker entry point for classifier training."""

from __future__ import annotations

from typing import Any

from rq import get_current_job

from app.core.training import train_classifier


def _update_job(**updates: object) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta.update(updates)
    job.save_meta()


def run_training_job(body: dict[str, Any]) -> dict[str, Any]:
    """Train and register a model while persisting progress in the RQ job."""
    try:

        def update_progress(update: dict[str, object]) -> None:
            _update_job(**update)

        result = train_classifier(
            body["yolo_datasets"],
            body["csv_datasets"] or None,
            image_size=body["image_size"],
            epochs=body["epochs"],
            batch_size=body["batch_size"],
            validation_ratio=body["validation_ratio"],
            progress_callback=update_progress,
        )
        final_metrics = {
            name: float(result[name])
            for name in ("val_accuracy", "val_loss")
            if name in result
        }
        completion: dict[str, object] = {
            "stage": "completed",
            "message": "Training completed",
            "progress": 100.0,
            "current_epoch": body["epochs"],
            "error": None,
        }
        if final_metrics:
            completion["metrics"] = final_metrics
        _update_job(
            **completion,
        )
        return result
    except Exception as exc:
        _update_job(
            stage="failed",
            message="Training failed",
            error=str(exc),
        )
        raise
