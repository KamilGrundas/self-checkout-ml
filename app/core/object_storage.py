import mimetypes
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings


@dataclass(frozen=True)
class S3Object:
    object_name: str
    size: int
    content_type: str | None = None


@dataclass(frozen=True)
class S3ObjectMetadata:
    content_type: str | None
    metadata: dict[str, str]


class S3ObjectStorage:
    def __init__(self, client: BaseClient, create_buckets: bool) -> None:
        self.client = client
        self.create_buckets = create_buckets

    def bucket_exists(self, bucket: str) -> bool:
        try:
            self.client.head_bucket(Bucket=bucket)
            return True
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status in {400, 404}:
                return False
            raise

    def ensure_bucket_exists(self, bucket: str) -> None:
        if self.bucket_exists(bucket):
            return
        if not self.create_buckets:
            raise RuntimeError(
                f"S3 bucket {bucket!r} is unavailable and S3_CREATE_BUCKETS=false"
            )
        kwargs: dict[str, object] = {"Bucket": bucket}
        if settings.S3_REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": settings.S3_REGION
            }
        self.client.create_bucket(**kwargs)

    def list_objects(self, bucket: str, prefix: str = "") -> Iterator[S3Object]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield S3Object(
                    object_name=item["Key"],
                    size=item.get("Size", 0),
                )

    def put_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.client.put_object(
            Bucket=bucket,
            Key=object_name,
            Body=data,
            ContentType=content_type,
            Metadata=metadata or {},
        )

    def upload_file(
        self,
        bucket: str,
        object_name: str,
        source: Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.client.upload_file(
            str(source),
            bucket,
            object_name,
            ExtraArgs={"ContentType": content_type},
        )

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        response = self.client.get_object(Bucket=bucket, Key=object_name)
        return response["Body"].read()

    def download_file(
        self, bucket: str, object_name: str, destination: Path | str
    ) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, object_name, str(destination))

    def head_object(self, bucket: str, object_name: str) -> S3ObjectMetadata:
        response = self.client.head_object(Bucket=bucket, Key=object_name)
        return S3ObjectMetadata(
            content_type=response.get("ContentType"),
            metadata=response.get("Metadata", {}),
        )

    def delete_objects(self, bucket: str, object_names: list[str]) -> None:
        for offset in range(0, len(object_names), 1000):
            batch = object_names[offset : offset + 1000]
            response = self.client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
            if errors := response.get("Errors"):
                raise RuntimeError(f"Failed to delete some S3 objects: {errors}")


@lru_cache
def get_object_storage() -> S3ObjectStorage:
    config = Config(
        region_name=settings.S3_REGION,
        connect_timeout=settings.S3_CONNECT_TIMEOUT,
        read_timeout=settings.S3_READ_TIMEOUT,
        retries={"max_attempts": settings.S3_MAX_RETRIES, "mode": "standard"},
        s3={
            "addressing_style": ("path" if settings.S3_FORCE_PATH_STYLE else "virtual")
        },
    )
    client = boto3.client(
        "s3",
        endpoint_url=str(settings.S3_ENDPOINT_URL),
        region_name=settings.S3_REGION,
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        aws_session_token=settings.S3_SESSION_TOKEN,
        use_ssl=settings.S3_USE_SSL,
        verify=settings.S3_VERIFY_TLS,
        config=config,
    )
    return S3ObjectStorage(client, settings.S3_CREATE_BUCKETS)


def ensure_bucket_exists(bucket_name: str | None = None) -> None:
    target_bucket = bucket_name or settings.S3_SHELF_BUCKET
    if target_bucket is None:
        raise RuntimeError("S3_SHELF_BUCKET is required")
    get_object_storage().ensure_bucket_exists(target_bucket)


def ensure_default_buckets_exist() -> None:
    for bucket_name in (
        settings.S3_SHELF_BUCKET,
        settings.S3_SCALE_BUCKET,
        settings.S3_EXTERNAL_BUCKET,
        settings.S3_TRAINING_BUCKET,
        settings.S3_LABEL_STUDIO_EXPORT_BUCKET,
    ):
        if bucket_name is None:
            raise RuntimeError("All ML S3 bucket names are required")
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
    base_url = str(settings.S3_PUBLIC_BASE_URL or settings.S3_ENDPOINT_URL).rstrip("/")
    target_bucket = bucket_name or settings.S3_SHELF_BUCKET
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
    target_bucket = bucket_name or settings.S3_SHELF_BUCKET
    if target_bucket is None:
        raise RuntimeError("S3 shelf bucket is required")
    ensure_bucket_exists(target_bucket)
    snapshot_filename = build_snapshot_filename(
        capture_index=capture_index,
        filename=filename,
        content_type=content_type,
    )
    object_name = build_object_name(session_id, snapshot_filename)
    get_object_storage().put_bytes(
        target_bucket,
        object_name,
        data,
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
    get_object_storage().put_bytes(
        bucket_name,
        object_name,
        data,
        content_type=content_type,
        metadata=metadata,
    )
    return object_name
