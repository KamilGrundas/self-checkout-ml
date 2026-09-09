from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jwt
import pytest
from fastapi import HTTPException

from app.api.routes import autolabel_scale
from app.core import autolabel
from app.core.autolabel import (
    AutolabelConfiguration,
    AutolabelSidecar,
    CatalogCandidate,
)
from app.core.object_storage import S3Object, S3ObjectMetadata


@pytest.fixture(autouse=True)
def authenticated_backend(monkeypatch):
    # Route tests use a stubbed authenticated backend; dependency auth has separate tests.
    monkeypatch.setattr(
        "app.api.deps.backend_identity",
        lambda _: {"id": "user-1", "is_superuser": True},
    )


def token() -> str:
    autolabel_scale.settings.SECRET_KEY = "unit-test-secret-at-least-32-bytes"
    return jwt.encode(
        {"sub": "user-1", "is_superuser": True},
        autolabel_scale.settings.SECRET_KEY,
        algorithm="HS256",
    )


class PageStorage:
    def __init__(self) -> None:
        self.objects = [
            S3Object("sessions/s1/captures/0000-empty.jpg", 10, etag="e0"),
            S3Object("sessions/s1/captures/0001-product.jpg", 20, etag="e1"),
            S3Object("raw/scale/upload.png", 30, etag="e2"),
            S3Object("_autolabel/scale/v1/result.json", 40, etag="e3"),
            S3Object("raw/scale/readme.txt", 50, etag="e4"),
        ]

    def list_objects_page(self, bucket: str, **kwargs):
        if kwargs.get("continuation_token") == "bad-token":
            from botocore.exceptions import ClientError

            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 400}},
                "ListObjectsV2",
            )
        if kwargs.get("continuation_token") == "next-token":
            return [], None
        return self.objects, "next-token"

    def list_objects(self, bucket: str):
        yield from self.objects

    def head_object(self, bucket: str, object_name: str) -> S3ObjectMetadata:
        content_type = (
            "image/jpeg"
            if object_name.endswith(".jpg")
            else "image/png"
            if object_name.endswith(".png")
            else "text/plain"
            if object_name.endswith(".txt")
            else "application/json"
        )
        item = next(item for item in self.objects if item.object_name == object_name)
        metadata = (
            {"product-id": "manual-id", "product-name": "Jab%C5%82ko"}
            if "0001-product" in object_name
            else {}
        )
        return S3ObjectMetadata(content_type, metadata, item.size, item.etag)

    def presigned_get_url(self, bucket: str, object_name: str) -> str:
        return f"http://storage.test/{bucket}/{object_name}"

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        return b"image-bytes"

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        self.objects = [
            item for item in self.objects if item.object_name not in object_names
        ]


def test_paginated_list_contains_only_scale_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PageStorage()
    monkeypatch.setattr(autolabel_scale, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel_scale, "read_sidecar", lambda *args: None)
    monkeypatch.setattr(autolabel_scale, "_active_items", lambda token: {})
    monkeypatch.setattr(autolabel_scale.settings, "S3_SCALE_BUCKET", "scale")

    page = autolabel_scale.list_scale_images(
        token(), page_size=100, label_product_id=None
    )

    assert [item.object_name for item in page.data] == [
        "sessions/s1/captures/0000-empty.jpg",
        "sessions/s1/captures/0001-product.jpg",
        "raw/scale/upload.png",
    ]
    assert page.data[0].is_empty is True
    assert page.data[0].is_imported is False
    assert page.data[1].capture_index == 1
    assert page.data[1].existing_product_name == "Jabłko"
    assert page.data[2].is_imported is True
    assert page.data[0].image_url == "autolabel/scale/images/content"
    assert page.next_cursor is None


def test_scale_images_are_counted_and_filtered_by_autolabel_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PageStorage()

    def sidecar(object_name: str, fingerprint: str):
        if "0001-product" not in object_name:
            return None
        return AutolabelSidecar(
            bucket="scale",
            object_name=object_name,
            source_size=20,
            source_fingerprint=fingerprint,
            product_id="apple-id",
            product_name="Apple",
            state="matched",
            timestamp=datetime.now(UTC),
            batch_id="batch-1",
            endpoint_url="test",
            max_tokens=1,
        )

    monkeypatch.setattr(autolabel_scale, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel_scale, "read_sidecar", sidecar)
    monkeypatch.setattr(autolabel_scale, "_active_items", lambda token: {})
    monkeypatch.setattr(autolabel_scale.settings, "S3_SCALE_BUCKET", "scale")

    counts = autolabel_scale.scale_label_counts()
    page = autolabel_scale.list_scale_images(
        token(), page_size=100, label_product_id="apple-id"
    )

    assert counts.total == 3
    assert [(item.product_name, item.count) for item in counts.labels] == [("Apple", 1)]
    assert [item.object_name for item in page.data] == [
        "sessions/s1/captures/0001-product.jpg"
    ]


def test_scale_image_content_is_returned_through_authenticated_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PageStorage()
    monkeypatch.setattr(autolabel_scale, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel_scale.settings, "S3_SCALE_BUCKET", "scale")

    response = autolabel_scale.get_scale_image_content(
        token(),
        "sessions/s1/captures/0001-product.jpg",
    )

    assert response.body == b"image-bytes"
    assert response.media_type == "image/jpeg"


def test_delete_scale_images_removes_sources_and_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PageStorage()
    monkeypatch.setattr(autolabel_scale, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel_scale.settings, "S3_SCALE_BUCKET", "scale")

    result = autolabel_scale.delete_scale_images(
        autolabel_scale.ImageDeleteRequest(
            object_names=["sessions/s1/captures/0001-product.jpg"]
        ),
        token(),
    )

    assert result == {"deleted": 1}
    assert "sessions/s1/captures/0001-product.jpg" not in {
        item.object_name for item in storage.objects
    }


def test_invalid_cursor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autolabel_scale.settings, "S3_SCALE_BUCKET", "scale")
    with pytest.raises(HTTPException) as error:
        autolabel_scale.list_scale_images(
            token(), cursor="!!", page_size=24, label_product_id=None
        )
    assert error.value.status_code == 422


class FakeJob:
    def __init__(self, job_id: str, meta: dict[str, Any]) -> None:
        self.id = job_id
        self.meta = meta

    def get_status(self, *, refresh: bool = False) -> str:
        return "queued"


class FakeQueue:
    def __init__(self) -> None:
        self.connection = object()
        self.jobs: dict[str, FakeJob] = {}
        self.enqueue_count = 0

    def enqueue(self, function, body, **kwargs):
        self.enqueue_count += 1
        job = FakeJob(kwargs["job_id"], kwargs["meta"])
        self.jobs[job.id] = job
        return job


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def set(self, key: str, value: str, *, ex: int, nx: bool = False):
        if nx and key in self.values:
            return False
        self.values[key] = str(value).encode()
        return True

    def get(self, key: str):
        return self.values.get(key)

    def delete(self, key: str):
        self.values.pop(key, None)


def test_batch_enqueue_deduplicates_objects_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue()
    redis = FakeRedis()
    monkeypatch.setattr(autolabel_scale, "get_autolabel_queue", lambda: queue)
    monkeypatch.setattr(autolabel_scale, "get_redis", lambda: redis)
    monkeypatch.setattr(
        autolabel_scale,
        "load_configuration",
        lambda token: AutolabelConfiguration(
            endpoint_url="http://vlm.test/inference",
            max_tokens=512,
            connect_timeout_seconds=5,
            read_timeout_seconds=120,
            configured=True,
        ),
    )
    monkeypatch.setattr(
        autolabel_scale,
        "load_catalog",
        lambda token: [
            CatalogCandidate(
                key="P0001",
                product_id="product-id",
                name="Jabłko",
                category="Owoce",
            )
        ],
    )
    monkeypatch.setattr(
        autolabel_scale,
        "require_source_image",
        lambda name: S3ObjectMetadata("image/jpeg", {}, 10, "etag"),
    )
    monkeypatch.setattr(
        autolabel_scale,
        "read_sidecar",
        lambda *args: type("Sidecar", (), {"state": "matched"})(),
    )
    monkeypatch.setattr(
        autolabel_scale.Job,
        "fetch",
        lambda job_id, connection: queue.jobs[job_id],
    )
    request = autolabel_scale.AutolabelBatchRequest(
        object_names=["one.jpg", "one.jpg", "two.jpg"]
    )

    first = autolabel_scale.create_batch(request, token(), "request-key-123")
    second = autolabel_scale.create_batch(request, token(), "request-key-123")

    assert first.batch_id == second.batch_id
    assert first.total == 2
    assert queue.enqueue_count == 1

    class BulkStorage:
        def list_objects(self, bucket: str):
            yield S3Object("sessions/s1/captures/0000-empty.jpg", 10, etag="empty")
            for index in range(50_000):
                yield S3Object(f"raw/scale/image-{index}.jpg", 10, etag=str(index))

    monkeypatch.setattr(autolabel_scale, "get_object_storage", lambda: BulkStorage())
    bulk = autolabel_scale.create_batch(
        autolabel_scale.AutolabelBatchRequest(selection="all_non_empty"),
        token(),
        "request-key-bulk",
    )

    assert bulk.total == 50_000
    assert bulk.items == []
    assert queue.enqueue_count == 2


def test_batch_rejects_invalid_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        autolabel_scale,
        "load_configuration",
        lambda token: AutolabelConfiguration(
            endpoint_url="http://vlm.test/inference",
            max_tokens=512,
            connect_timeout_seconds=5,
            read_timeout_seconds=120,
            configured=True,
        ),
    )
    monkeypatch.setattr(
        autolabel_scale,
        "load_catalog",
        lambda token: [
            CatalogCandidate(
                key="P0001",
                product_id="product-id",
                name="Jabłko",
                category="Owoce",
            )
        ],
    )

    def invalid(name: str):
        raise ValueError("not an image")

    monkeypatch.setattr(autolabel_scale, "require_source_image", invalid)
    with pytest.raises(HTTPException) as invalid_error:
        autolabel_scale.create_batch(
            autolabel_scale.AutolabelBatchRequest(object_names=["bad.txt"]),
            token(),
            "request-key-123",
        )
    assert invalid_error.value.status_code == 422
