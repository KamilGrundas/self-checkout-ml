import asyncio
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes import train
from app.core import training, training_worker


class FakeJob:
    def __init__(
        self,
        *,
        status: str = "queued",
        meta: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.id = "test-job"
        self.meta = meta or {}
        self._status = status
        self._result = result
        self.saved_meta = 0

    def get_status(self, *, refresh: bool = False) -> str:
        return self._status

    def return_value(self) -> dict[str, Any] | None:
        return self._result

    def save_meta(self) -> None:
        self.saved_meta += 1


class FakeQueue:
    def __init__(self, job: FakeJob) -> None:
        self.job = job
        self.enqueued: dict[str, Any] | None = None

    def enqueue(self, function, body, **kwargs):
        self.enqueued = {"function": function, "body": body, **kwargs}
        self.job.id = kwargs["job_id"]
        self.job.meta = kwargs["meta"]
        return self.job


def valid_body(**updates: Any) -> dict[str, Any]:
    body = {
        "yolo_datasets": ["dataset"],
        "csv_datasets": [],
        "image_size": 160,
        "epochs": 3,
        "batch_size": 16,
        "validation_ratio": 0.2,
    }
    body.update(updates)
    return body


def test_training_request_enforces_resource_bounds() -> None:
    for update in (
        {"image_size": 16},
        {"image_size": 1024},
        {"epochs": 0},
        {"epochs": 501},
        {"batch_size": 0},
        {"validation_ratio": 0.75},
    ):
        with pytest.raises(ValidationError):
            train.TrainRequest(**valid_body(**update))


def test_classifier_persists_job_in_queue(monkeypatch) -> None:
    job = FakeJob()
    queue = FakeQueue(job)
    monkeypatch.setattr(train, "get_training_queue", lambda: queue)

    response = asyncio.run(train.classifier(train.TrainRequest(**valid_body(epochs=7))))

    assert response.status == "queued"
    assert response.total_epochs == 7
    assert queue.enqueued is not None
    assert queue.enqueued["function"] is training_worker.run_training_job
    assert queue.enqueued["job_timeout"] == "24h"
    assert queue.enqueued["result_ttl"] == 7 * 24 * 60 * 60


def test_classifier_rejects_empty_dataset_selection() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(train.classifier(train.TrainRequest()))
    assert error.value.status_code == 422


def test_training_worker_persists_progress_and_result(monkeypatch) -> None:
    job = FakeJob(meta={"total_epochs": 3})
    monkeypatch.setattr(training_worker, "get_current_job", lambda: job)

    def fake_train_classifier(*args, progress_callback, **kwargs):
        progress_callback(
            {
                "stage": "training",
                "message": "Completed epoch 2 of 3",
                "progress": 65,
                "current_epoch": 2,
                "total_epochs": 3,
                "metrics": {"accuracy": 0.75},
            }
        )
        return {"model_id": "model-1"}

    monkeypatch.setattr(training_worker, "train_classifier", fake_train_classifier)

    result = training_worker.run_training_job(valid_body())

    assert result == {"model_id": "model-1"}
    assert job.meta["stage"] == "completed"
    assert job.meta["current_epoch"] == 3
    assert job.meta["metrics"] == {"accuracy": 0.75}
    assert job.saved_meta == 2


def test_training_worker_persists_failure(monkeypatch) -> None:
    job = FakeJob()
    monkeypatch.setattr(training_worker, "get_current_job", lambda: job)

    def fail_training(*args, **kwargs):
        raise RuntimeError("Object storage unavailable")

    monkeypatch.setattr(training_worker, "train_classifier", fail_training)

    with pytest.raises(RuntimeError, match="Object storage unavailable"):
        training_worker.run_training_job(valid_body())

    assert job.meta["stage"] == "failed"
    assert job.meta["error"] == "Object storage unavailable"


def test_completed_queue_job_is_serialized() -> None:
    job = FakeJob(
        status="finished",
        meta={
            "stage": "completed",
            "message": "Training completed",
            "progress": 100,
            "total_epochs": 3,
        },
        result={"model_id": "model-1"},
    )

    response = train._serialize_job(job)  # type: ignore[arg-type]

    assert response.status == "completed"
    assert response.result == {"model_id": "model-1"}


def test_classifier_trains_on_image_features_and_reports_each_epoch(tmp_path) -> None:
    x_train = np.zeros((4, 32, 32, 3), dtype=np.float32)
    x_train[1] = 0.1
    x_train[2] = 0.9
    x_train[3] = 1.0
    y_train = np.array([0, 0, 1, 1], dtype=np.int32)
    x_val = np.stack(
        [
            np.full((32, 32, 3), 0.05, dtype=np.float32),
            np.full((32, 32, 3), 0.95, dtype=np.float32),
        ]
    )
    y_val = np.array([0, 1], dtype=np.int32)
    reported_epochs: list[int] = []

    model, metrics = training._fit_classifier(
        x_train,
        y_train,
        x_val,
        y_val,
        num_classes=2,
        epochs=3,
        batch_size=2,
        epoch_callback=lambda epoch, _: reported_epochs.append(epoch),
    )

    assert reported_epochs == [1, 2, 3]
    assert model.predict_proba(x_val.reshape(2, -1)).shape == (2, 2)
    assert 0 <= metrics["val_accuracy"] <= 1

    import joblib

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    assert restored.predict_proba(x_val.reshape(2, -1)).shape == (2, 2)
