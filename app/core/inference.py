from __future__ import annotations

import io
import json
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from botocore.exceptions import ClientError
from fastapi import HTTPException

from app.core.config import settings
from app.core.object_storage import get_object_storage

MODEL_PREFIX = "models"


class ObjectStorageModelStore:
    """Small model registry backed only by the configured S3-compatible store."""

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._cached_version: int | None = None
        self._cached_model: Any | None = None
        self._cached_metadata: dict[str, Any] | None = None

    @property
    def _versions_prefix(self) -> str:
        return f"{MODEL_PREFIX}/{self.model_name}/versions"

    @property
    def _active_object(self) -> str:
        return f"{MODEL_PREFIX}/{self.model_name}/active.json"

    def _metadata_object(self, version: int) -> str:
        return f"{self._versions_prefix}/{version}/metadata.json"

    def _model_object(self, version: int) -> str:
        return f"{self._versions_prefix}/{version}/model.joblib"

    def _read_json(self, object_name: str) -> dict[str, Any] | None:
        try:
            body = get_object_storage().get_bytes(
                settings.S3_TRAINING_BUCKET, object_name
            )
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                return None
            raise
        value = json.loads(body)
        return value if isinstance(value, dict) else None

    def _write_json(self, object_name: str, value: dict[str, Any]) -> None:
        get_object_storage().put_bytes(
            settings.S3_TRAINING_BUCKET,
            object_name,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
            content_type="application/json",
        )

    def list_versions(self) -> list[dict[str, Any]]:
        storage = get_object_storage()
        if not storage.bucket_exists(settings.S3_TRAINING_BUCKET):
            return []
        active = self._read_json(self._active_object) or {}
        active_version = active.get("version")
        versions: list[dict[str, Any]] = []
        for item in storage.list_objects(
            settings.S3_TRAINING_BUCKET, prefix=f"{self._versions_prefix}/"
        ):
            if not item.object_name.endswith("/metadata.json"):
                continue
            metadata = self._read_json(item.object_name)
            if not metadata:
                continue
            metadata["is_active"] = metadata.get("version") == active_version
            versions.append(metadata)
        return sorted(versions, key=lambda value: int(value["version"]), reverse=True)

    def register(
        self,
        *,
        model: Any,
        labels: list[str],
        image_size: int,
        metrics: dict[str, float],
        parameters: dict[str, Any],
        activate: bool = True,
    ) -> dict[str, Any]:
        import joblib

        with self._lock:
            versions = self.list_versions()
            version = max((int(item["version"]) for item in versions), default=0) + 1
            model_id = str(uuid4())
            created_at = datetime.now(UTC).isoformat()
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            get_object_storage().put_bytes(
                settings.S3_TRAINING_BUCKET,
                self._model_object(version),
                buffer.getvalue(),
                content_type="application/octet-stream",
            )
            metadata: dict[str, Any] = {
                "name": self.model_name,
                "version": version,
                "model_id": model_id,
                "status": "ready",
                "description": None,
                "created_at": created_at,
                "labels": labels,
                "image_size": image_size,
                "metrics": metrics,
                "parameters": parameters,
                "artifact_object": self._model_object(version),
            }
            self._write_json(self._metadata_object(version), metadata)
            if activate:
                self._activate(version, metadata)
            return {**metadata, "is_active": activate}

    def _activate(self, version: int, metadata: dict[str, Any]) -> None:
        self._write_json(
            self._active_object,
            {
                "version": version,
                "model_id": metadata["model_id"],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        self._cached_version = None
        self._cached_model = None
        self._cached_metadata = None

    def set_version(self, version: int) -> dict[str, Any]:
        with self._lock:
            metadata = self._read_json(self._metadata_object(version))
            if metadata is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version} not found for model {self.model_name}",
                )
            self._activate(version, metadata)
            return {
                "model_name": self.model_name,
                "model_version": version,
                "model_id": metadata["model_id"],
                "cache_key": f"{self.model_name}:{version}",
            }

    def _load_active(self) -> tuple[Any, dict[str, Any]]:
        import joblib

        active = self._read_json(self._active_object)
        if not active or not isinstance(active.get("version"), int):
            raise HTTPException(
                status_code=503,
                detail=f"No active model is configured for {self.model_name}",
            )
        version = int(active["version"])
        if self._cached_version == version and self._cached_model is not None:
            assert self._cached_metadata is not None
            return self._cached_model, self._cached_metadata
        metadata = self._read_json(self._metadata_object(version))
        if not metadata:
            raise HTTPException(
                status_code=503, detail="Active model metadata is missing"
            )
        try:
            payload = get_object_storage().get_bytes(
                settings.S3_TRAINING_BUCKET, self._model_object(version)
            )
            model = joblib.load(io.BytesIO(payload))
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Active model artifact could not be loaded"
            ) from exc
        self._cached_version = version
        self._cached_model = model
        self._cached_metadata = metadata
        return model, metadata

    def predict(self, image_bytes: bytes) -> tuple[dict[str, float], str]:
        import cv2
        import numpy as np

        model, metadata = self._load_active()
        labels = metadata.get("labels")
        image_size = metadata.get("image_size")
        if (
            not isinstance(labels, list)
            or not labels
            or not isinstance(image_size, int)
        ):
            raise HTTPException(
                status_code=503, detail="Active model metadata is invalid"
            )
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        normalized = resized.astype(np.float32) / 255.0
        probabilities = model.predict_proba(normalized.reshape(1, -1))[0]
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
        return scores, str(metadata["model_id"])


classifier_model_store = ObjectStorageModelStore(model_name="classifier")
shelf_model_store = ObjectStorageModelStore(model_name="detector")
