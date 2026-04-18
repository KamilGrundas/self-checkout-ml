from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from app.core.config import settings

if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion


def configure_local_caches() -> None:
    local_cache_dir = Path(".cache")
    os.environ.setdefault("XDG_CACHE_HOME", str(local_cache_dir.resolve()))
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((local_cache_dir / "matplotlib").resolve()),
    )
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")


class KerasRegistryModelStore:
    REFRESH_COOLDOWN_SECONDS = 60

    def __init__(
        self,
        *,
        registered_model_name: str,
        cache_prefix: str,
    ) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        configure_local_caches()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self._client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        self._registered_model_name = registered_model_name
        self._cache_prefix = cache_prefix
        self._cached_cache_key: str | None = None
        self._cached_run_id: str | None = None
        self._cached_model: Any | None = None
        self._cached_labels: list[str] | None = None
        self._refresh_lock = threading.Lock()
        self._last_manual_refresh_at = 0.0
        self._cache_dir = Path(settings.MODEL_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _disk_model_path(self) -> Path:
        return self._cache_dir / f"{self._cache_prefix}.keras"

    def _disk_metadata_path(self) -> Path:
        return self._cache_dir / f"{self._cache_prefix}_metadata.json"

    def _write_disk_cache(
        self,
        *,
        model: Any,
        labels: list[str],
        run_id: str,
        registered_model_name: str,
        registered_model_version: str,
    ) -> None:
        model.save(self._disk_model_path())
        self._disk_metadata_path().write_text(
            json.dumps(
                {
                    "labels": labels,
                    "run_id": run_id,
                    "registered_model_name": registered_model_name,
                    "registered_model_version": registered_model_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_from_disk_cache(self) -> tuple[Any, list[str], str] | None:
        import tensorflow as tf

        model_path = self._disk_model_path()
        metadata_path = self._disk_metadata_path()
        if not model_path.exists() or not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = metadata.get("labels")
        run_id = metadata.get("run_id")
        registered_model_name = metadata.get("registered_model_name")
        registered_model_version = metadata.get("registered_model_version")
        if (
            not isinstance(labels, list)
            or not labels
            or not isinstance(run_id, str)
            or not isinstance(registered_model_name, str)
            or not isinstance(registered_model_version, str)
        ):
            return None

        model = tf.keras.models.load_model(model_path)
        self._cached_run_id = run_id
        self._cached_cache_key = f"{registered_model_name}:{registered_model_version}"
        self._cached_model = model
        self._cached_labels = labels
        return model, labels, run_id

    def _latest_registered_model(self) -> ModelVersion:
        from mlflow.exceptions import MlflowException

        try:
            versions = list(
                self._client.search_model_versions(
                    f"name='{self._registered_model_name}'"
                )
            )
        except MlflowException as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "MLflow is unavailable or misconfigured. "
                    f"Tracking URI: {settings.MLFLOW_TRACKING_URI}."
                ),
            ) from error

        if not versions:
            raise HTTPException(
                status_code=503,
                detail=f"No registered model found in MLflow for {self._registered_model_name}.",
            )

        return max(versions, key=lambda version: int(version.version))

    def _load_from_registry(
        self, latest_version: ModelVersion
    ) -> tuple[Any, list[str]]:
        import mlflow.keras
        from mlflow.exceptions import MlflowException
        from mlflow.models import get_model_info

        model_uri = f"models:/{latest_version.name}/{latest_version.version}"
        try:
            model_info = get_model_info(model_uri)
            metadata = model_info.metadata or {}
            labels = metadata.get("labels")
            if not isinstance(labels, list) or not labels:
                raise ValueError("Missing labels in MLflow model metadata")
            model = mlflow.keras.load_model(model_uri)
        except (MlflowException, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model from MLflow for {self._registered_model_name}.",
            ) from error
        return model, labels

    def _ensure_loaded(self) -> tuple[Any, list[str], str]:
        if (
            self._cached_model is not None
            and self._cached_labels
            and self._cached_run_id
            and self._cached_cache_key
        ):
            return self._cached_model, self._cached_labels, self._cached_run_id

        disk_cached = self._load_from_disk_cache()
        if disk_cached is not None:
            return disk_cached

        latest_version = self._latest_registered_model()
        model, labels = self._load_from_registry(latest_version)

        self._write_disk_cache(
            model=model,
            labels=labels,
            run_id=latest_version.run_id,
            registered_model_name=latest_version.name,
            registered_model_version=latest_version.version,
        )
        self._cached_cache_key = f"{latest_version.name}:{latest_version.version}"
        self._cached_run_id = latest_version.run_id
        self._cached_model = model
        self._cached_labels = labels
        return model, labels, latest_version.run_id

    def refresh_latest_model(self) -> dict[str, str | int]:
        with self._refresh_lock:
            now = time.monotonic()
            seconds_since_refresh = now - self._last_manual_refresh_at
            if seconds_since_refresh < self.REFRESH_COOLDOWN_SECONDS:
                retry_after = int(self.REFRESH_COOLDOWN_SECONDS - seconds_since_refresh)
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Model refresh is rate limited. "
                        f"Retry after {retry_after} seconds."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

            latest_version = self._latest_registered_model()
            model, labels = self._load_from_registry(latest_version)
            cache_key = f"{latest_version.name}:{latest_version.version}"

            self._write_disk_cache(
                model=model,
                labels=labels,
                run_id=latest_version.run_id,
                registered_model_name=latest_version.name,
                registered_model_version=latest_version.version,
            )
            self._cached_cache_key = cache_key
            self._cached_run_id = latest_version.run_id
            self._cached_model = model
            self._cached_labels = labels
            self._last_manual_refresh_at = time.monotonic()

            return {
                "model_name": latest_version.name,
                "model_version": int(latest_version.version),
                "run_id": latest_version.run_id,
                "cache_key": cache_key,
            }

    def predict(self, image_bytes: bytes) -> tuple[dict[str, float], str]:
        import cv2
        import numpy as np

        model, labels, run_id = self._ensure_loaded()

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, image_height, image_width, _ = model.input_shape
        resized = cv2.resize(
            rgb,
            (image_width, image_height),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)

        probabilities = model.predict(batch, verbose=0)[0]
        scores = dict(
            sorted(
                (
                    (label, float(probabilities[index]))
                    for index, label in enumerate(labels)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return scores, run_id


classifier_model_store = KerasRegistryModelStore(
    registered_model_name=settings.MLFLOW_REGISTERED_MODEL_NAME,
    cache_prefix="classifier_model",
)

shelf_model_store = KerasRegistryModelStore(
    registered_model_name=settings.MLFLOW_SHELF_MODEL_NAME,
    cache_prefix="shelf_model",
)
