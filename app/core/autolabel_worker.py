from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from rq import get_current_job

from app.core.autolabel import (
    AutolabelConfiguration,
    CatalogCandidate,
    process_image,
    write_failed_sidecar,
)


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
    configuration = AutolabelConfiguration.model_validate(body["configuration"])
    catalog = [CatalogCandidate.model_validate(item) for item in body["catalog"]]
    batch_id = str(body["batch_id"])
    meta["status"] = "processing"
    _save_meta(meta)

    for index, object_name in enumerate(body["object_names"]):
        item = items[index]
        item["status"] = "processing"
        meta["items"] = items
        _save_meta(meta)
        try:
            sidecar = process_image(
                object_name=object_name,
                batch_id=batch_id,
                configuration=configuration,
                catalog=catalog,
            )
            item.update(
                status=sidecar.state,
                product_id=sidecar.product_id,
                product_name=sidecar.product_name,
                error=None,
            )
        except Exception as exc:
            error = _safe_error(exc)
            write_failed_sidecar(
                object_name=object_name,
                batch_id=batch_id,
                configuration=configuration,
                error=error,
            )
            item.update(
                status="failed",
                product_id=None,
                product_name=None,
                error=error,
            )
        meta["completed"] = index + 1
        meta["matched"] = sum(i["status"] == "matched" for i in items)
        meta["unmatched"] = sum(i["status"] == "unmatched" for i in items)
        meta["failed"] = sum(i["status"] == "failed" for i in items)
        meta["items"] = items
        _save_meta(meta)

    meta["status"] = "completed"
    _save_meta(meta)
    return meta
