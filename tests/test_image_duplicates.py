from __future__ import annotations

import cv2
import numpy as np

from app.core import image_duplicates
from app.core.object_storage import S3Object


class DuplicateStorage:
    def __init__(self, images: dict[str, bytes]) -> None:
        self.images = images
        self.deleted: list[str] = []

    def list_objects(self, bucket: str, prefix: str = ""):
        for name, body in self.images.items():
            if name.startswith(prefix):
                yield S3Object(name, len(body), content_type="image/jpeg")

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        return self.images[object_name]

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        self.deleted.extend(object_names)


def encoded_image(value: int, quality: int = 95) -> bytes:
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    image[16:48, 16:48] = 255 - value
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return encoded.tobytes()


def test_duplicate_scan_keeps_first_image_and_reports_near_identical_copy(
    monkeypatch,
) -> None:
    original = encoded_image(20)
    storage = DuplicateStorage(
        {
            "labeled/first.jpg": original,
            "labeled/duplicate.jpg": original,
            "labeled/near-duplicate.jpg": encoded_image(20, quality=90),
            "labeled/different.jpg": encoded_image(120),
        }
    )
    monkeypatch.setattr(image_duplicates, "get_object_storage", lambda: storage)
    monkeypatch.setattr(image_duplicates.settings, "S3_EXTERNAL_BUCKET", "external")
    monkeypatch.setattr(image_duplicates, "get_current_job", lambda: None)

    result = image_duplicates.find_duplicate_images()

    assert result["duplicate_count"] == 2
    assert result["duplicate_object_names"] == [
        "labeled/duplicate.jpg",
        "labeled/near-duplicate.jpg",
    ]
    assert result["similarity_threshold"] == 0.99


def test_delete_duplicates_rejects_non_labeled_objects(monkeypatch) -> None:
    storage = DuplicateStorage({})
    monkeypatch.setattr(image_duplicates, "get_object_storage", lambda: storage)
    monkeypatch.setattr(image_duplicates.settings, "S3_EXTERNAL_BUCKET", "external")

    result = image_duplicates.delete_duplicate_images(["labeled/duplicate.jpg"])

    assert result == {"deleted": 1}
    assert storage.deleted == ["labeled/duplicate.jpg"]
