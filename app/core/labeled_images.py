from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import quote, unquote
from uuid import uuid4

from app.core.config import settings
from app.core.object_storage import S3ObjectMetadata, get_object_storage

LABELED_PREFIX = "labeled"
DATASETS_PREFIX = "datasets/releases"
MAX_SELECTED_IMAGES = 1000


def read_label(metadata: S3ObjectMetadata) -> tuple[str | None, str | None]:
    product_id = unquote(metadata.metadata.get("product-id") or "") or None
    product_name = unquote(metadata.metadata.get("product-name") or "") or None
    return product_id, product_name


def labeled_object_name(filename: str | None, content_type: str) -> str:
    suffix = PurePosixPath(filename or "").suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".bin"
    return f"{LABELED_PREFIX}/{datetime.now(UTC):%Y/%m/%d}/{uuid4().hex}{suffix}"


def label_metadata(product_id: str, product_name: str) -> dict[str, str]:
    return {
        "product-id": quote(product_id, safe=""),
        "product-name": quote(product_name, safe=""),
        "labeled-at": datetime.now(UTC).isoformat(),
    }


def export_csv_dataset(
    *,
    release_name: str,
    images: Iterable[tuple[str, str, str]],
) -> dict:
    """Export (bucket, object name, label) tuples as a self-contained CSV dataset."""
    storage = get_object_storage()
    release_id = uuid4().hex[:8]
    normalized = "-".join(release_name.strip().lower().split())
    safe_name = "".join(ch for ch in normalized if ch.isalnum() or ch in "-_")
    if not safe_name:
        safe_name = datetime.now(UTC).strftime("dataset-%Y%m%d-%H%M%S")
    release = f"{safe_name}-{release_id}"
    release_prefix = f"{DATASETS_PREFIX}/labeled-images/{release}"
    rows: list[tuple[str, str]] = []
    for index, (bucket, object_name, label) in enumerate(images, start=1):
        metadata = storage.head_object(bucket, object_name)
        suffix = PurePosixPath(object_name).suffix.lower() or ".bin"
        filename = f"{index:06d}-{hashlib.sha256(object_name.encode()).hexdigest()[:12]}{suffix}"
        storage.copy_object(
            source_bucket=bucket,
            source_object=object_name,
            target_bucket=settings.S3_TRAINING_BUCKET,
            target_object=f"{release_prefix}/images/{filename}",
            content_type=metadata.content_type or "application/octet-stream",
            metadata={},
        )
        rows.append((filename, label))
    if not rows:
        raise ValueError("At least one labeled image is required")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)
    created_at = datetime.now(UTC).isoformat()
    storage.put_bytes(
        settings.S3_TRAINING_BUCKET,
        f"{release_prefix}/dataset.csv",
        output.getvalue().encode(),
        content_type="text/csv",
    )
    manifest = {
        "project_title": "Labeled images",
        "release_name": release,
        "export_type": "CSV",
        "sample_count": len(rows),
        "created_at": created_at,
    }
    storage.put_bytes(
        settings.S3_TRAINING_BUCKET,
        f"{release_prefix}/manifest.json",
        json.dumps(manifest, separators=(",", ":")).encode(),
        content_type="application/json",
    )
    return {**manifest, "release_prefix": release_prefix}
