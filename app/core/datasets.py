"""YOLO dataset management in MinIO (list, delete)."""

from __future__ import annotations

import json

from app.core.config import settings

DATASETS_PREFIX = "datasets/releases"


def list_datasets() -> list[dict]:
    """List all exported datasets from the training-data bucket."""
    from app.core.object_storage import get_minio_client

    bucket = settings.ML_MINIO_TRAINING_BUCKET_NAME
    client = get_minio_client()

    if not client.bucket_exists(bucket):
        return []

    datasets = []
    for obj in client.list_objects(
        bucket, prefix=f"{DATASETS_PREFIX}/", recursive=True
    ):
        if not obj.object_name.endswith("/manifest.json"):
            continue
        data = client.get_object(bucket, obj.object_name)
        manifest = json.loads(data.read().decode("utf-8"))
        data.close()

        parts = obj.object_name.split("/")
        manifest["project_slug"] = parts[2] if len(parts) > 3 else None
        manifest["release_prefix"] = "/".join(parts[:-1])
        manifest["bucket"] = bucket
        datasets.append(manifest)

    datasets.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return datasets


def delete_dataset(project_slug: str, release_name: str) -> dict:
    """Delete a dataset release from the training-data bucket."""
    from minio.deleteobjects import DeleteObject

    from app.core.object_storage import get_minio_client

    bucket = settings.ML_MINIO_TRAINING_BUCKET_NAME
    client = get_minio_client()
    prefix = f"{DATASETS_PREFIX}/{project_slug}/{release_name}/"

    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        raise ValueError(f"Dataset '{project_slug}/{release_name}' not found")

    delete_list = [DeleteObject(obj.object_name) for obj in objects]
    errors = list(client.remove_objects(bucket, delete_list))
    if errors:
        raise RuntimeError(f"Failed to delete some objects: {errors}")

    return {
        "project_slug": project_slug,
        "release_name": release_name,
        "deleted_files": len(delete_list),
    }
