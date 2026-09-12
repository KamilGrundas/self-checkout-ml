from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import ValidationError
from rq import get_current_job

from app.core.autolabel import (
    AutolabelConfiguration,
    CatalogCandidate,
    is_supported_image,
    process_image,
    read_sidecar,
    sidecar_object_name,
    source_fingerprint,
    write_failed_sidecar,
)
from app.core.config import settings
from app.core.labeled_images import label_metadata, labeled_object_name
from app.core.object_storage import get_object_storage


def _save_meta(meta: dict[str, Any]) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta = meta
    job.save_meta()


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, (ValueError, RuntimeError, ValidationError)):
        return str(exc)[:512]
    return "Unexpected autolabel processing error"


def run_autolabel_batch(body: dict[str, Any]) -> dict[str, Any]:
    job = get_current_job()
    meta = dict(job.meta) if job is not None else {}
    items = list(meta.get("items") or [])
    completed_object_names = list(meta.get("completed_object_names") or [])
    detailed_items = len(items) == len(body["object_names"])
    configuration = AutolabelConfiguration.model_validate(body["configuration"])
    catalog = [CatalogCandidate.model_validate(item) for item in body["catalog"]]
    batch_id = str(body["batch_id"])
    meta["status"] = "processing"
    _save_meta(meta)

    object_names = body["object_names"]
    if detailed_items:
        for item in items:
            item["status"] = "processing"
        meta["items"] = items
        _save_meta(meta)

    def process(object_name: str):
        return process_image(
            object_name=object_name,
            batch_id=batch_id,
            configuration=configuration,
            catalog=catalog,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(object_names))) as executor:
        futures = {
            executor.submit(process, object_name): (index, object_name)
            for index, object_name in enumerate(object_names)
        }
        for future in as_completed(futures):
            if job is not None:
                job.refresh()
                if job.meta.get("cancel_requested"):
                    continue
            index, object_name = futures[future]
            item = items[index] if detailed_items else None
            try:
                sidecar = future.result()
                if item is not None:
                    item.update(
                        status=sidecar.state,
                        product_id=sidecar.product_id,
                        product_name=sidecar.product_name,
                        error=None,
                    )
                meta[sidecar.state] = int(meta.get(sidecar.state) or 0) + 1
            except Exception as exc:
                error = _safe_error(exc)
                write_failed_sidecar(
                    object_name=object_name,
                    batch_id=batch_id,
                    configuration=configuration,
                    error=error,
                )
                if item is not None:
                    item.update(
                        status="failed",
                        product_id=None,
                        product_name=None,
                        error=error,
                    )
                meta["failed"] = int(meta.get("failed") or 0) + 1
            meta["completed"] = int(meta.get("completed") or 0) + 1
            completed_object_names.append(object_name)
            meta["completed_object_names"] = completed_object_names
            meta["items"] = items
            _save_meta(meta)

    if job is not None:
        job.refresh()
        if job.meta.get("cancel_requested"):
            return dict(job.meta)
    meta["status"] = "completed"
    _save_meta(meta)
    return meta


def run_finalize_all_matched() -> dict[str, int]:
    if not settings.S3_SCALE_BUCKET or not settings.S3_EXTERNAL_BUCKET:
        raise RuntimeError("Image buckets are not configured")
    storage = get_object_storage()
    moved = 0
    for item in storage.list_objects(settings.S3_SCALE_BUCKET):
        if not is_supported_image(item.object_name, item.content_type):
            continue
        metadata = storage.head_object(settings.S3_SCALE_BUCKET, item.object_name)
        sidecar = read_sidecar(item.object_name, source_fingerprint(metadata))
        if not sidecar or sidecar.state != "matched" or not sidecar.product_name:
            continue
        target = labeled_object_name(
            item.object_name.rsplit("/", 1)[-1],
            metadata.content_type or "application/octet-stream",
        )
        storage.copy_object(
            source_bucket=settings.S3_SCALE_BUCKET,
            source_object=item.object_name,
            target_bucket=settings.S3_EXTERNAL_BUCKET,
            target_object=target,
            content_type=metadata.content_type or "application/octet-stream",
            metadata=label_metadata(sidecar.product_id or "", sidecar.product_name),
        )
        storage.delete_objects(
            settings.S3_SCALE_BUCKET,
            [item.object_name, sidecar_object_name(item.object_name)],
        )
        moved += 1
        _save_meta({"status": "processing", "moved": moved})
    result = {"moved": moved}
    _save_meta({"status": "completed", **result})
    return result
