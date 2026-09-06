from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.api.routes import datasets
from app.core.labeled_images import label_metadata
from app.core.object_storage import S3Object, S3ObjectMetadata


def image_upload(name: str, body: bytes = b"image") -> UploadFile:
    return UploadFile(
        file=BytesIO(body),
        filename=name,
        headers=Headers({"content-type": "image/jpeg"}),
    )


def test_scale_import_rejects_more_than_one_hundred_images() -> None:
    files = [image_upload(f"image-{index}.jpg") for index in range(101)]

    with pytest.raises(HTTPException) as error:
        asyncio.run(datasets.upload_scale_images(files))

    assert error.value.status_code == 413


def test_raw_import_rejects_images_larger_than_twenty_megabytes() -> None:
    file = image_upload(
        "large.jpg",
        b"x" * (datasets.MAX_UPLOAD_IMAGE_BYTES + 1),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            datasets._upload_raw_image(
                bucket_name="scale",
                prefix="raw/scale",
                file=file,
            )
        )

    assert error.value.status_code == 413


class LabeledStorage:
    def __init__(self) -> None:
        self.objects = [
            S3Object("labeled/apple-1.jpg", 10, "image/jpeg", "a1"),
            S3Object("labeled/banana.jpg", 10, "image/jpeg", "b1"),
            S3Object("labeled/apple-2.jpg", 10, "image/jpeg", "a2"),
        ]

    def list_objects_page(self, bucket: str, **kwargs):
        start = int(kwargs.get("continuation_token") or 0)
        end = min(start + kwargs["max_keys"], len(self.objects))
        return self.objects[start:end], str(end) if end < len(self.objects) else None

    def list_objects(self, bucket: str, prefix: str = ""):
        yield from (
            item for item in self.objects if item.object_name.startswith(prefix)
        )

    def head_object(self, bucket: str, object_name: str) -> S3ObjectMetadata:
        product_id = "banana-id" if "banana" in object_name else "apple-id"
        product_name = "Banana" if product_id == "banana-id" else "Apple"
        item = next(item for item in self.objects if item.object_name == object_name)
        return S3ObjectMetadata(
            "image/jpeg",
            label_metadata(product_id, product_name),
            item.size,
            item.etag,
        )

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        self.objects = [
            item for item in self.objects if item.object_name not in object_names
        ]


def test_labeled_images_are_counted_and_filtered_by_final_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LabeledStorage()
    monkeypatch.setattr(datasets, "get_object_storage", lambda: storage)
    monkeypatch.setattr(datasets.settings, "S3_EXTERNAL_BUCKET", "external")

    counts = asyncio.run(datasets.labeled_image_counts())
    page = asyncio.run(
        datasets.list_labeled_images(
            cursor=None, page_size=100, label_product_id="apple-id"
        )
    )

    assert counts.total == 3
    assert [(item.product_name, item.count) for item in counts.labels] == [
        ("Apple", 2),
        ("Banana", 1),
    ]
    assert [item.object_name for item in page.data] == [
        "labeled/apple-1.jpg",
        "labeled/apple-2.jpg",
    ]


def test_delete_labeled_images_removes_selected_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LabeledStorage()
    monkeypatch.setattr(datasets, "get_object_storage", lambda: storage)
    monkeypatch.setattr(datasets.settings, "S3_EXTERNAL_BUCKET", "external")

    result = asyncio.run(
        datasets.delete_labeled_images(
            datasets.ImageDeleteRequest(
                object_names=["labeled/apple-1.jpg", "labeled/banana.jpg"]
            )
        )
    )

    assert result == {"deleted": 2}
    assert [item.object_name for item in storage.objects] == ["labeled/apple-2.jpg"]
