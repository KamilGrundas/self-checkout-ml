from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
from rq import get_current_job
from skimage.metrics import structural_similarity

from app.core.config import settings
from app.core.object_storage import get_object_storage

SIMILARITY_THRESHOLD = 0.99


def _update_progress(processed: int, total: int) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta.update(
        status="processing",
        processed=processed,
        total=total,
        progress=(processed / total * 100 if total else 100),
    )
    job.save_meta()


def _decode_image(body: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Stored image cannot be decoded")
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)


def _difference_hash(image: np.ndarray) -> int:
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def find_duplicate_images() -> dict:
    if not settings.S3_EXTERNAL_BUCKET:
        raise RuntimeError("Labeled image bucket is not configured")
    storage = get_object_storage()
    objects = list(storage.list_objects(settings.S3_EXTERNAL_BUCKET, prefix="labeled/"))
    representatives: list[tuple[str, str, int]] = []
    exact: dict[str, int] = {}
    hash_bands: dict[tuple[int, int], set[int]] = defaultdict(set)
    duplicates: list[str] = []

    @lru_cache(maxsize=256)
    def representative_image(index: int) -> np.ndarray:
        object_name = representatives[index][0]
        return _decode_image(
            storage.get_bytes(settings.S3_EXTERNAL_BUCKET, object_name)
        )

    for processed, item in enumerate(objects, start=1):
        body = storage.get_bytes(settings.S3_EXTERNAL_BUCKET, item.object_name)
        digest = hashlib.sha256(body).hexdigest()
        exact_match = exact.get(digest)
        if exact_match is not None:
            duplicates.append(item.object_name)
            if processed % 25 == 0 or processed == len(objects):
                _update_progress(processed, len(objects))
            continue

        image = _decode_image(body)
        image_hash = _difference_hash(image)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(hash_bands[(band, (image_hash >> (band * 16)) & 0xFFFF)])

        duplicate_of = next(
            (
                index
                for index in candidates
                if structural_similarity(
                    image, representative_image(index), data_range=255
                )
                >= SIMILARITY_THRESHOLD
            ),
            None,
        )
        if duplicate_of is not None:
            duplicates.append(item.object_name)
        else:
            index = len(representatives)
            representatives.append((item.object_name, digest, image_hash))
            exact[digest] = index
            for band in range(4):
                hash_bands[(band, (image_hash >> (band * 16)) & 0xFFFF)].add(index)
        if processed % 25 == 0 or processed == len(objects):
            _update_progress(processed, len(objects))

    result = {
        "duplicate_count": len(duplicates),
        "duplicate_object_names": duplicates,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    job = get_current_job()
    if job is not None:
        job.meta.update(status="completed", progress=100, **result)
        job.save_meta()
    return result


def delete_duplicate_images(object_names: list[str]) -> dict[str, int]:
    if not settings.S3_EXTERNAL_BUCKET:
        raise RuntimeError("Labeled image bucket is not configured")
    valid = list(
        dict.fromkeys(name for name in object_names if name.startswith("labeled/"))
    )
    if len(valid) != len(set(object_names)):
        raise ValueError("Invalid duplicate image selection")
    get_object_storage().delete_objects(settings.S3_EXTERNAL_BUCKET, valid)
    return {"deleted": len(valid)}
