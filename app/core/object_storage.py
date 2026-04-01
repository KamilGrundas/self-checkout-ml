import json
import mimetypes
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from minio import Minio

from app.core.config import settings


@lru_cache
def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
    )


def ensure_bucket_exists() -> None:
    client = get_minio_client()
    if not client.bucket_exists(settings.ML_MINIO_BUCKET_NAME):
        client.make_bucket(settings.ML_MINIO_BUCKET_NAME)
    client.set_bucket_policy(
        settings.ML_MINIO_BUCKET_NAME,
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{settings.ML_MINIO_BUCKET_NAME}/*"],
                    }
                ],
            }
        ),
    )


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


def public_object_url(object_name: str) -> str:
    base_url = settings.MINIO_PUBLIC_URL.rstrip("/")
    return f"{base_url}/{settings.ML_MINIO_BUCKET_NAME}/{quote(object_name, safe='/')}"


def store_session_snapshot(
    *,
    session_id: str,
    capture_index: int,
    product_id: str | None,
    product_name: str | None,
    filename: str | None,
    content_type: str,
    data: bytes,
) -> tuple[str, str]:
    ensure_bucket_exists()
    snapshot_filename = build_snapshot_filename(
        capture_index=capture_index,
        filename=filename,
        content_type=content_type,
    )
    object_name = build_object_name(session_id, snapshot_filename)
    client = get_minio_client()
    client.put_object(
        bucket_name=settings.ML_MINIO_BUCKET_NAME,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type=content_type,
        metadata={
            "capture-index": str(capture_index),
            "product-id": product_id or "",
            "product-name": product_name or "",
        },
    )
    return snapshot_filename, object_name
