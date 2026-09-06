from __future__ import annotations

import json
from dataclasses import dataclass

from botocore.exceptions import ClientError

from app.core import inference, labeled_images
from app.core.object_storage import S3Object, S3ObjectMetadata


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.copies: list[tuple[str, str, str, str]] = []

    def bucket_exists(self, bucket: str) -> bool:
        return True

    def put_bytes(self, bucket: str, object_name: str, data: bytes, **kwargs) -> None:
        self.objects[(bucket, object_name)] = data

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        try:
            return self.objects[(bucket, object_name)]
        except KeyError as exc:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject"
            ) from exc

    def list_objects(self, bucket: str, prefix: str = ""):
        for stored_bucket, object_name in sorted(self.objects):
            if stored_bucket == bucket and object_name.startswith(prefix):
                yield S3Object(
                    object_name=object_name,
                    size=len(self.objects[(bucket, object_name)]),
                )

    def head_object(self, bucket: str, object_name: str) -> S3ObjectMetadata:
        return S3ObjectMetadata("image/jpeg", {}, size=10, etag="etag")

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        for object_name in object_names:
            self.objects.pop((bucket, object_name), None)

    def copy_object(
        self,
        *,
        source_bucket: str,
        source_object: str,
        target_bucket: str,
        target_object: str,
        **kwargs,
    ) -> None:
        self.copies.append((source_bucket, source_object, target_bucket, target_object))


@dataclass
class SerializableModel:
    value: str = "model"


def test_native_model_registry_stores_metrics_and_active_version(monkeypatch) -> None:
    storage = MemoryStorage()
    monkeypatch.setattr(inference, "get_object_storage", lambda: storage)
    monkeypatch.setattr(inference.settings, "S3_TRAINING_BUCKET", "training")
    store = inference.ObjectStorageModelStore(model_name="classifier")

    registered = store.register(
        model=SerializableModel(),
        labels=["Apple", "Banana"],
        image_size=160,
        metrics={"accuracy": 0.9, "val_accuracy": 0.8},
        parameters={"epochs": 3},
    )

    assert registered["version"] == 1
    assert registered["is_active"] is True
    versions = store.list_versions()
    assert versions[0]["metrics"]["val_accuracy"] == 0.8
    assert versions[0]["is_active"] is True
    active = json.loads(storage.objects[("training", "models/classifier/active.json")])
    assert active["version"] == 1


def test_native_model_registry_deletes_model_and_activates_latest_remaining(
    monkeypatch,
) -> None:
    storage = MemoryStorage()
    monkeypatch.setattr(inference, "get_object_storage", lambda: storage)
    monkeypatch.setattr(inference.settings, "S3_TRAINING_BUCKET", "training")
    store = inference.ObjectStorageModelStore(model_name="classifier")

    first = store.register(
        model=SerializableModel("first"),
        labels=["Apple"],
        image_size=32,
        metrics={},
        parameters={},
    )
    second = store.register(
        model=SerializableModel("second"),
        labels=["Apple"],
        image_size=32,
        metrics={},
        parameters={},
    )

    result = store.delete_version(second["version"])

    assert result["model_id"] == second["model_id"]
    versions = store.list_versions()
    assert [item["model_id"] for item in versions] == [first["model_id"]]
    assert versions[0]["is_active"] is True


def test_export_selected_images_creates_self_contained_csv_dataset(monkeypatch) -> None:
    storage = MemoryStorage()
    monkeypatch.setattr(labeled_images, "get_object_storage", lambda: storage)
    monkeypatch.setattr(labeled_images.settings, "S3_TRAINING_BUCKET", "training")

    result = labeled_images.export_csv_dataset(
        release_name="Fruit Review",
        images=[
            ("source", "labeled/apple.jpg", "Apple"),
            ("source", "labeled/banana.jpg", "Banana"),
        ],
    )

    assert result["sample_count"] == 2
    prefix = result["release_prefix"]
    csv_body = storage.objects[("training", f"{prefix}/dataset.csv")].decode()
    assert "Apple" in csv_body
    assert "Banana" in csv_body
    assert len(storage.copies) == 2
