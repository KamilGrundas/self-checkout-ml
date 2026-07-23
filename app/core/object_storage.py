import json
import mimetypes
import uuid
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from minio import Minio

from app.core.config import settings


PUBLIC_BUCKETS = {
    settings.ML_MINIO_SHELF_BUCKET_NAME,
    settings.ML_MINIO_SCALE_BUCKET_NAME,
    settings.ML_MINIO_EXTERNAL_BUCKET_NAME,
}


@lru_cache
def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
    )


def set_public_bucket_policy(bucket_name: str) -> None:
    client = get_minio_client()
    client.set_bucket_policy(
        bucket_name,
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                    }
                ],
            }
        ),
    )


def ensure_bucket_exists(bucket_name: str | None = None) -> None:
    target_bucket = bucket_name or settings.ML_MINIO_SHELF_BUCKET_NAME
    client = get_minio_client()
    if not client.bucket_exists(target_bucket):
        client.make_bucket(target_bucket)
    if target_bucket in PUBLIC_BUCKETS:
        set_public_bucket_policy(target_bucket)


def ensure_default_buckets_exist() -> None:
    for bucket_name in (
        settings.ML_MINIO_SHELF_BUCKET_NAME,
        settings.ML_MINIO_SCALE_BUCKET_NAME,
        settings.ML_MINIO_EXTERNAL_BUCKET_NAME,
        settings.ML_MINIO_TRAINING_BUCKET_NAME,
        settings.ML_MINIO_LABELSTUDIO_EXPORT_BUCKET_NAME,
    ):
        ensure_bucket_exists(bucket_name)


def build_snapshot_filename(
    capture_index: int,
    filename: str | None,
    content_type: str,
) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".bin"
    kind = "empty" if capture_index == 0 else "product"
    return f"{capture_index:04d}-{kind}{suffix}"


def build_object_name(session_id: str, filename: str) -> str:
    return f"sessions/{session_id}/captures/{filename}"


def public_object_url(object_name: str, bucket_name: str | None = None) -> str:
    base_url = settings.MINIO_PUBLIC_URL.rstrip("/")
    target_bucket = bucket_name or settings.ML_MINIO_SHELF_BUCKET_NAME
    return f"{base_url}/{target_bucket}/{quote(object_name, safe='/')}"


def store_session_snapshot(
    *,
    session_id: str,
    capture_index: int,
    product_id: str | None,
    product_name: str | None,
    filename: str | None,
    content_type: str,
    data: bytes,
    bucket_name: str | None = None,
) -> tuple[str, str]:
    target_bucket = bucket_name or settings.ML_MINIO_SHELF_BUCKET_NAME
    ensure_bucket_exists(target_bucket)
    snapshot_filename = build_snapshot_filename(
        capture_index=capture_index,
        filename=filename,
        content_type=content_type,
    )
    object_name = build_object_name(session_id, snapshot_filename)
    client = get_minio_client()
    client.put_object(
        bucket_name=target_bucket,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type=content_type,
        metadata={
            "capture-index": str(capture_index),
            "product-id": quote(product_id or "", safe=""),
            "product-name": quote(product_name or "", safe=""),
        },
    )
    return snapshot_filename, object_name


def build_dataset_object_name(
    prefix: str, filename: str | None, content_type: str
) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".bin"
    unique_name = uuid.uuid4().hex
    normalized_prefix = prefix.strip("/")
    return f"{normalized_prefix}/{unique_name}{suffix}"


def store_dataset_image(
    *,
    bucket_name: str,
    prefix: str,
    filename: str | None,
    content_type: str,
    data: bytes,
    metadata: dict[str, str] | None = None,
) -> str:
    ensure_bucket_exists(bucket_name)
    object_name = build_dataset_object_name(
        prefix=prefix,
        filename=filename,
        content_type=content_type,
    )
    client = get_minio_client()
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type=content_type,
        metadata=metadata or {},
    )
    return object_name
