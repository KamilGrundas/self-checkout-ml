from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from botocore.exceptions import ClientError
from fastapi import APIRouter, Header, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, ValidationError
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.api.deps import SuperuserDep, SuperuserToken
from app.core.autolabel import (
    MAX_BATCH_IMAGES,
    MAX_OBJECT_NAME_LENGTH,
    AutolabelConfiguration,
    AutolabelSidecar,
    build_prompt,
    call_inference,
    cancel_provider_inference,
    configure_provider_cancellation,
    decode_cursor,
    encode_cursor,
    is_supported_image,
    load_catalog,
    load_configuration,
    manual_label,
    read_sidecar,
    require_source_image,
    sidecar_object_name,
    source_fingerprint,
)
from app.core.autolabel_queue import (
    IDEMPOTENCY_TTL_SECONDS,
    JOB_RETENTION_SECONDS,
    JOB_TIMEOUT,
    get_autolabel_queue,
    get_redis,
    initial_job_meta,
    request_digest,
)
from app.core.autolabel_worker import run_autolabel_batch, run_finalize_all_matched
from app.core.config import settings
from app.core.labeled_images import (
    MAX_SELECTED_IMAGES,
    label_metadata,
    labeled_object_name,
)
from app.core.object_storage import get_object_storage

router = APIRouter(
    prefix="/autolabel/scale",
    tags=["scale-autolabel"],
    dependencies=[SuperuserDep],
)

ItemStatus = Literal[
    "unlabeled",
    "queued",
    "processing",
    "matched",
    "unmatched",
    "failed",
    "cancelled",
]


class AutolabelResultPublic(BaseModel):
    state: Literal["matched", "unmatched", "failed"]
    product_id: str | None
    product_name: str | None
    timestamp: datetime
    batch_id: str
    error: str | None = None


class ScaleImagePublic(BaseModel):
    object_name: str
    image_url: str
    size: int
    etag: str | None
    session_id: str | None
    capture_index: int | None
    captured_at: datetime | None
    is_empty: bool
    is_imported: bool
    existing_product_id: str | None
    existing_product_name: str | None
    autolabel: AutolabelResultPublic | None
    status: ItemStatus


class ScaleImagesPage(BaseModel):
    data: list[ScaleImagePublic]
    next_cursor: str | None


class AutolabelBatchRequest(BaseModel):
    object_names: list[str] = Field(default_factory=list, max_length=MAX_BATCH_IMAGES)
    selection: Literal["explicit", "all_non_empty"] = "explicit"
    retry_only: bool = False


class AutolabelBatchItem(BaseModel):
    object_name: str
    status: ItemStatus
    product_id: str | None = None
    product_name: str | None = None
    error: str | None = None


class ActiveAutolabelItems(dict[str, AutolabelBatchItem]):
    """Expose detailed items, or synthesize bulk processing states on lookup."""

    def __init__(
        self,
        items: list[AutolabelBatchItem],
        *,
        object_names: set[str] | None = None,
        completed_object_names: set[str] | None = None,
        status: ItemStatus | None = None,
    ) -> None:
        super().__init__((item.object_name, item) for item in items)
        self.object_names = object_names or set()
        self.completed_object_names = completed_object_names or set()
        self.status = status

    def get(
        self, object_name: str, default: AutolabelBatchItem | None = None
    ) -> AutolabelBatchItem | None:
        item = super().get(object_name)
        if item is not None:
            return item
        if (
            self.status is not None
            and object_name in self.object_names
            and object_name not in self.completed_object_names
        ):
            return AutolabelBatchItem(object_name=object_name, status=self.status)
        return default


class AutolabelBatchPublic(BaseModel):
    batch_id: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    total: int
    completed: int
    matched: int
    unmatched: int
    failed: int
    cancelled: int
    provider_cancellation: Literal[
        "available", "not_supported", "unavailable", "requested", "not_requested"
    ]
    items: list[AutolabelBatchItem]


class AutolabelTestRequest(BaseModel):
    object_name: str = Field(min_length=1, max_length=MAX_OBJECT_NAME_LENGTH)


class AutolabelTestResult(BaseModel):
    state: Literal["matched", "unmatched"]
    candidate_key: str | None
    product_id: str | None
    product_name: str | None
    response_sha256: str


class ManualLabelRequest(BaseModel):
    object_name: str = Field(min_length=1, max_length=MAX_OBJECT_NAME_LENGTH)
    product_id: str = Field(min_length=1, max_length=128)


class ImageDeleteRequest(BaseModel):
    object_names: list[str] = Field(min_length=1, max_length=MAX_BATCH_IMAGES)


class PendingImageSelection(BaseModel):
    object_names: list[str] = Field(
        default_factory=list, max_length=MAX_SELECTED_IMAGES
    )
    selection: Literal["explicit", "all_matched"] = "explicit"


class SelectionCountRequest(BaseModel):
    selection: Literal["all_non_empty", "all_matched"]


class SelectionCountPublic(BaseModel):
    count: int


class LabelCountPublic(BaseModel):
    product_id: str
    product_name: str
    count: int


class LabelCountsPublic(BaseModel):
    total: int
    labels: list[LabelCountPublic]


class FinalizeJobPublic(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    moved: int = 0
    error: str | None = None


def _subject(access_token: str) -> str:
    from app.api.deps import backend_identity

    return str(backend_identity(access_token)["id"])


def _redis_text(value: bytes | str | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode()
    return value


def _serialize_job(job: Job) -> AutolabelBatchPublic:
    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}
    if rq_status in {"failed", "stopped"}:
        public_status = "failed"
    else:
        public_status = str(meta.get("status") or "queued")
    if rq_status == "canceled" or meta.get("cancel_requested"):
        public_status = "cancelled"
    if public_status not in {
        "queued",
        "processing",
        "completed",
        "failed",
        "cancelled",
    }:
        public_status = "queued"
    items = [
        AutolabelBatchItem.model_validate(item) for item in (meta.get("items") or [])
    ]
    if public_status in {"failed", "cancelled"}:
        for item in items:
            if item.status in {"queued", "processing"}:
                item.status = public_status
                item.error = (
                    "Autolabeling was cancelled"
                    if public_status == "cancelled"
                    else "Autolabel worker stopped before completing the item"
                )
    return AutolabelBatchPublic(
        batch_id=job.id,
        status=public_status,  # type: ignore[arg-type]
        total=int(meta.get("total") or len(items)),
        completed=int(meta.get("completed") or 0),
        matched=int(meta.get("matched") or 0),
        unmatched=int(meta.get("unmatched") or 0),
        failed=int(meta.get("failed") or 0),
        cancelled=int(meta.get("cancelled") or 0),
        provider_cancellation=str(meta.get("provider_cancellation") or "not_supported"),  # type: ignore[arg-type]
        items=items,
    )


def _latest_batch(access_token: str) -> AutolabelBatchPublic | None:
    try:
        job_id = _redis_text(
            get_redis().get(f"autolabel:latest:{_subject(access_token)}")
        )
        if not job_id:
            return None
        job = Job.fetch(job_id, connection=get_autolabel_queue().connection)
        return _serialize_job(job)
    except (NoSuchJobError, RedisError):
        return None


def _active_items(access_token: str) -> dict[str, AutolabelBatchItem]:
    batch = _latest_batch(access_token)
    if not batch or batch.status not in {"queued", "processing", "cancelled"}:
        return {}
    if batch.items:
        return ActiveAutolabelItems(batch.items)
    try:
        job = Job.fetch(batch.batch_id, connection=get_autolabel_queue().connection)
        body = job.args[0]
        object_names = {str(name) for name in body["object_names"]}
        completed_object_names = {
            str(name) for name in (job.meta or {}).get("completed_object_names") or []
        }
    except (IndexError, KeyError, TypeError, NoSuchJobError, RedisError):
        return {}
    return ActiveAutolabelItems(
        [],
        object_names=object_names,
        completed_object_names=completed_object_names,
        status=batch.status,
    )


def _parse_session_fields(object_name: str) -> tuple[str | None, int | None]:
    matched = re.fullmatch(
        r"sessions/([^/]+)/captures/(\d+)-(?:empty|product)(?:\.[^/]+)?",
        object_name,
    )
    if not matched:
        return None, None
    return matched.group(1), int(matched.group(2))


def _result_public(sidecar: AutolabelSidecar | None) -> AutolabelResultPublic | None:
    if sidecar is None:
        return None
    return AutolabelResultPublic(
        state=sidecar.state,
        product_id=sidecar.product_id,
        product_name=sidecar.product_name,
        timestamp=sidecar.timestamp,
        batch_id=sidecar.batch_id,
        error=sidecar.error,
    )


def _is_empty_image(object_name: str) -> bool:
    _, capture_index = _parse_session_fields(object_name)
    return capture_index == 0 or "0000-empty" in object_name.rsplit("/", 1)[-1]


def _all_non_empty_image_names() -> list[str]:
    if not settings.S3_SCALE_BUCKET:
        raise ValueError("Scale image bucket is not configured")
    return [
        item.object_name
        for item in get_object_storage().list_objects(settings.S3_SCALE_BUCKET)
        if is_supported_image(item.object_name, item.content_type)
        and not _is_empty_image(item.object_name)
    ]


def _matched_image_count() -> int:
    if not settings.S3_SCALE_BUCKET:
        raise ValueError("Scale image bucket is not configured")
    storage = get_object_storage()
    count = 0
    for item in storage.list_objects(settings.S3_SCALE_BUCKET):
        if not is_supported_image(item.object_name, item.content_type):
            continue
        metadata = storage.head_object(settings.S3_SCALE_BUCKET, item.object_name)
        sidecar = read_sidecar(item.object_name, source_fingerprint(metadata))
        if sidecar and sidecar.state == "matched":
            count += 1
    return count


def _scale_label_counts() -> LabelCountsPublic:
    if not settings.S3_SCALE_BUCKET:
        raise ValueError("Scale image bucket is not configured")
    storage = get_object_storage()
    total = 0
    labels: dict[str, tuple[str, int]] = {}
    for item in storage.list_objects(settings.S3_SCALE_BUCKET):
        metadata = storage.head_object(settings.S3_SCALE_BUCKET, item.object_name)
        if not is_supported_image(item.object_name, metadata.content_type):
            continue
        total += 1
        sidecar = read_sidecar(item.object_name, source_fingerprint(metadata))
        if not sidecar or sidecar.state != "matched" or not sidecar.product_id:
            continue
        product_name, count = labels.get(
            sidecar.product_id, (sidecar.product_name or sidecar.product_id, 0)
        )
        labels[sidecar.product_id] = (product_name, count + 1)
    return LabelCountsPublic(
        total=total,
        labels=sorted(
            (
                LabelCountPublic(
                    product_id=product_id,
                    product_name=product_name,
                    count=count,
                )
                for product_id, (product_name, count) in labels.items()
            ),
            key=lambda item: item.product_name.casefold(),
        ),
    )


@router.post("/images/selection-count", response_model=SelectionCountPublic)
def selection_count(body: SelectionCountRequest) -> SelectionCountPublic:
    try:
        count = (
            len(_all_non_empty_image_names())
            if body.selection == "all_non_empty"
            else _matched_image_count()
        )
        return SelectionCountPublic(count=count)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/images/label-counts", response_model=LabelCountsPublic)
def scale_label_counts() -> LabelCountsPublic:
    try:
        return _scale_label_counts()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/images", response_model=ScaleImagesPage)
def list_scale_images(
    access_token: SuperuserToken,
    cursor: str | None = None,
    page_size: int = Query(default=100, ge=1, le=100),
    label_product_id: str | None = Query(default=None, max_length=128),
) -> ScaleImagesPage:
    if not settings.S3_SCALE_BUCKET:
        raise HTTPException(
            status_code=503, detail="Scale image bucket is not configured"
        )
    try:
        continuation = decode_cursor(cursor) if cursor else None
        storage = get_object_storage()
    except (ValueError, ClientError) as exc:
        raise HTTPException(status_code=422, detail="Invalid image cursor") from exc

    active_items = _active_items(access_token)
    images: list[ScaleImagePublic] = []
    next_token: str | None = continuation
    try:
        while len(images) < page_size:
            objects, next_token = storage.list_objects_page(
                settings.S3_SCALE_BUCKET,
                max_keys=page_size - len(images),
                continuation_token=next_token,
            )
            for item in objects:
                metadata = storage.head_object(
                    settings.S3_SCALE_BUCKET, item.object_name
                )
                if not is_supported_image(item.object_name, metadata.content_type):
                    continue
                fingerprint = source_fingerprint(metadata)
                sidecar = read_sidecar(item.object_name, fingerprint)
                if label_product_id and (
                    sidecar is None or sidecar.product_id != label_product_id
                ):
                    continue
                existing_product_id, existing_product_name = manual_label(metadata)
                session_id, capture_index = _parse_session_fields(item.object_name)
                active = active_items.get(item.object_name)
                item_status: ItemStatus
                if active:
                    item_status = active.status
                elif sidecar:
                    item_status = sidecar.state
                else:
                    item_status = "unlabeled"
                images.append(
                    ScaleImagePublic(
                        object_name=item.object_name,
                        image_url="autolabel/scale/images/content",
                        size=metadata.size,
                        etag=metadata.etag,
                        session_id=session_id,
                        capture_index=capture_index,
                        captured_at=item.last_modified,
                        is_empty=_is_empty_image(item.object_name),
                        is_imported=item.object_name.startswith("raw/scale/"),
                        existing_product_id=existing_product_id,
                        existing_product_name=existing_product_name,
                        autolabel=_result_public(sidecar),
                        status=item_status,
                    )
                )
                if len(images) == page_size:
                    break
            if not next_token:
                break
    except ClientError as exc:
        raise HTTPException(
            status_code=502, detail="Could not load scale images"
        ) from exc
    return ScaleImagesPage(
        data=images,
        next_cursor=encode_cursor(next_token) if next_token else None,
    )


@router.get("/images/content")
def get_scale_image_content(
    access_token: SuperuserToken,
    object_name: str = Query(min_length=1, max_length=MAX_OBJECT_NAME_LENGTH),
) -> Response:
    del access_token
    try:
        metadata = require_source_image(object_name)
        image_bytes = get_object_storage().get_bytes(
            settings.S3_SCALE_BUCKET,
            object_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClientError as exc:
        raise HTTPException(
            status_code=502, detail="Could not load scale image"
        ) from exc
    return Response(
        content=image_bytes,
        media_type=metadata.content_type or "application/octet-stream",
        headers={
            "Cache-Control": "private, max-age=60",
            **({"ETag": metadata.etag} if metadata.etag else {}),
        },
    )


@router.post(
    "/batches",
    response_model=AutolabelBatchPublic,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_batch(
    body: AutolabelBatchRequest,
    access_token: SuperuserToken,
    idempotency_key: str = Header(
        min_length=8,
        max_length=128,
        alias="Idempotency-Key",
    ),
) -> AutolabelBatchPublic:
    if body.selection == "all_non_empty":
        if body.object_names or body.retry_only:
            raise HTTPException(
                status_code=422,
                detail="Bulk autolabel selection cannot include explicit images or retry mode",
            )
        try:
            object_names = _all_non_empty_image_names()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    else:
        object_names = list(dict.fromkeys(body.object_names))
    if not object_names:
        raise HTTPException(status_code=422, detail="No images matched the selection")
    if body.selection == "explicit" and len(object_names) != len(body.object_names):
        body = body.model_copy(update={"object_names": object_names})
    batch_id = str(uuid4())
    try:
        configuration = load_configuration(access_token)
        catalog = load_catalog(access_token)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    configuration, provider_cancellation = configure_provider_cancellation(
        configuration, batch_id
    )
    for object_name in object_names if body.selection == "explicit" else []:
        try:
            metadata = require_source_image(object_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sidecar = read_sidecar(object_name, source_fingerprint(metadata))
        if body.retry_only and (sidecar is None or sidecar.state == "matched"):
            raise HTTPException(
                status_code=409,
                detail="Retry accepts only failed or unmatched images",
            )

    subject = _subject(access_token)
    digest = request_digest(object_names)
    redis = get_redis()
    idempotency_redis_key = f"autolabel:idempotency:{subject}:{idempotency_key}"
    reservation = json.dumps({"digest": digest, "batch_id": batch_id})
    try:
        reserved = redis.set(
            idempotency_redis_key,
            reservation,
            ex=IDEMPOTENCY_TTL_SECONDS,
            nx=True,
        )
        if not reserved:
            existing_raw = _redis_text(redis.get(idempotency_redis_key))
            existing = json.loads(existing_raw or "{}")
            if existing.get("digest") != digest:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency-Key was already used for another request",
                )
            job = Job.fetch(
                str(existing["batch_id"]),
                connection=get_autolabel_queue().connection,
            )
            return _serialize_job(job)

        job = get_autolabel_queue().enqueue(
            run_autolabel_batch,
            {
                "batch_id": batch_id,
                "object_names": object_names,
                "configuration": configuration.model_dump(mode="json"),
                "catalog": [item.model_dump(mode="json") for item in catalog],
            },
            job_id=batch_id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=JOB_RETENTION_SECONDS,
            failure_ttl=JOB_RETENTION_SECONDS,
            meta=initial_job_meta(
                object_names,
                include_items=body.selection == "explicit",
                provider_cancellation=provider_cancellation,
            ),
            description=f"Autolabel {len(object_names)} scale images",
        )
        redis.set(
            f"autolabel:latest:{subject}",
            batch_id,
            ex=JOB_RETENTION_SECONDS,
        )
        return _serialize_job(job)
    except HTTPException:
        raise
    except (RedisError, NoSuchJobError, KeyError, ValueError) as exc:
        redis.delete(idempotency_redis_key)
        raise HTTPException(
            status_code=503,
            detail="Autolabel queue is unavailable",
        ) from exc


@router.get("/batches/latest", response_model=AutolabelBatchPublic)
def latest_batch(access_token: SuperuserToken) -> AutolabelBatchPublic:
    batch = _latest_batch(access_token)
    if batch is None:
        raise HTTPException(status_code=404, detail="No recent autolabel batch")
    return batch


@router.get("/batches/{batch_id}", response_model=AutolabelBatchPublic)
def batch_status(
    batch_id: str,
    access_token: SuperuserToken,
) -> AutolabelBatchPublic:
    latest = _latest_batch(access_token)
    if latest is None or latest.batch_id != batch_id:
        raise HTTPException(status_code=404, detail="Autolabel batch not found")
    return latest


@router.post("/batches/{batch_id}/cancel", response_model=AutolabelBatchPublic)
def cancel_batch(batch_id: str, access_token: SuperuserToken) -> AutolabelBatchPublic:
    latest = _latest_batch(access_token)
    if latest is None or latest.batch_id != batch_id:
        raise HTTPException(status_code=404, detail="Autolabel batch not found")
    if latest.status not in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Autolabel batch is not running")
    try:
        job = Job.fetch(batch_id, connection=get_autolabel_queue().connection)
        meta = dict(job.meta or {})
        try:
            body = job.args[0]
            configuration = AutolabelConfiguration.model_validate(body["configuration"])
            provider_cancellation = cancel_provider_inference(configuration)
        except (IndexError, KeyError, TypeError, ValidationError):
            provider_cancellation = "unavailable"
        cancelled = 0
        for item in meta.get("items") or []:
            if item.get("status") in {"queued", "processing"}:
                item.update(
                    status="cancelled",
                    product_id=None,
                    product_name=None,
                    error="Autolabeling was cancelled",
                )
                cancelled += 1
        meta.update(
            status="cancelled",
            cancel_requested=True,
            completed=int(meta.get("total") or len(meta.get("items") or [])),
            cancelled=cancelled,
            provider_cancellation=provider_cancellation,
            items=meta.get("items") or [],
        )
        job.cancel()
        job.meta = meta
        job.save_meta()
        return _serialize_job(job)
    except (NoSuchJobError, RedisError) as exc:
        raise HTTPException(
            status_code=503, detail="Autolabel queue is unavailable"
        ) from exc


@router.post("/test", response_model=AutolabelTestResult)
def test_endpoint(
    body: AutolabelTestRequest,
    access_token: SuperuserToken,
) -> AutolabelTestResult:
    if not settings.S3_SCALE_BUCKET:
        raise HTTPException(
            status_code=503, detail="Scale image bucket is not configured"
        )
    try:
        configuration = load_configuration(access_token)
        catalog = load_catalog(access_token)
        metadata = require_source_image(body.object_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    image_bytes = get_object_storage().get_bytes(
        settings.S3_SCALE_BUCKET, body.object_name
    )
    try:
        result = call_inference(
            configuration=configuration,
            object_name=body.object_name,
            content_type=metadata.content_type or "application/octet-stream",
            image_bytes=image_bytes,
            prompt=build_prompt(catalog),
            allowed_keys={item.key for item in catalog},
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    selected = next(
        (item for item in catalog if item.key == result.candidate_key), None
    )
    return AutolabelTestResult(
        state=result.state,
        candidate_key=result.candidate_key,
        product_id=selected.product_id if selected else None,
        product_name=selected.name if selected else None,
        response_sha256=result.response_sha256,
    )


@router.patch("/images/label", response_model=ScaleImagePublic)
def update_image_label(
    body: ManualLabelRequest,
    access_token: SuperuserToken,
) -> ScaleImagePublic:
    try:
        catalog = load_catalog(access_token)
        product = next(
            (
                candidate
                for candidate in catalog
                if candidate.product_id == body.product_id
            ),
            None,
        )
        if product is None:
            raise ValueError("Selected product does not exist")
        metadata = require_source_image(body.object_name)
        existing_product_id, existing_product_name = manual_label(metadata)
        session_id, capture_index = _parse_session_fields(body.object_name)
        sidecar = AutolabelSidecar(
            bucket=settings.S3_SCALE_BUCKET,
            object_name=body.object_name,
            source_size=metadata.size,
            source_fingerprint=source_fingerprint(metadata),
            product_id=product.product_id,
            product_name=product.name,
            state="matched",
            timestamp=datetime.now(UTC),
            batch_id=f"manual-{uuid4()}",
            endpoint_url="manual",
            max_tokens=1,
        )
        from app.core.autolabel import write_sidecar

        write_sidecar(sidecar)
        return ScaleImagePublic(
            object_name=body.object_name,
            image_url="autolabel/scale/images/content",
            size=metadata.size,
            etag=metadata.etag,
            session_id=session_id,
            capture_index=capture_index,
            captured_at=None,
            is_empty=_is_empty_image(body.object_name),
            is_imported=body.object_name.startswith("raw/scale/"),
            existing_product_id=existing_product_id,
            existing_product_name=existing_product_name,
            autolabel=_result_public(sidecar),
            status="matched",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/images")
def delete_scale_images(
    body: ImageDeleteRequest, access_token: SuperuserToken
) -> dict[str, int]:
    del access_token
    storage = get_object_storage()
    selected = list(dict.fromkeys(body.object_names))
    try:
        for object_name in selected:
            require_source_image(object_name)
        storage.delete_objects(
            settings.S3_SCALE_BUCKET,
            [
                name
                for object_name in selected
                for name in (object_name, sidecar_object_name(object_name))
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClientError as exc:
        raise HTTPException(
            status_code=502, detail="Could not delete scale images"
        ) from exc
    return {"deleted": len(selected)}


def _selected_pending_rows(
    object_names: list[str],
) -> list[tuple[str, str, str, AutolabelSidecar]]:
    rows: list[tuple[str, str, str, AutolabelSidecar]] = []
    for object_name in list(dict.fromkeys(object_names)):
        metadata = require_source_image(object_name)
        sidecar = read_sidecar(object_name, source_fingerprint(metadata))
        if not sidecar or sidecar.state != "matched" or not sidecar.product_name:
            raise ValueError(f"Image has no completed Label: {object_name}")
        rows.append(
            (settings.S3_SCALE_BUCKET, object_name, sidecar.product_name, sidecar)
        )
    return rows


@router.post("/images/finalize")
def finalize_pending_images(
    body: PendingImageSelection, access_token: SuperuserToken
) -> dict:
    del access_token
    storage = get_object_storage()
    try:
        if body.selection == "all_matched":
            if body.object_names:
                raise ValueError(
                    "Matched selection cannot include explicit image names"
                )
            job = get_autolabel_queue().enqueue(
                run_finalize_all_matched,
                job_timeout=JOB_TIMEOUT,
                result_ttl=JOB_RETENTION_SECONDS,
                failure_ttl=JOB_RETENTION_SECONDS,
                meta={"status": "queued", "moved": 0},
                description="Move all matched images to the labeled collection",
            )
            return {"queued": True, "job_id": job.id}
        else:
            rows = _selected_pending_rows(body.object_names)
        if not rows:
            raise ValueError("No matched images selected")
        moved: list[str] = []
        moved_count = 0
        for bucket, object_name, _, sidecar in rows:
            metadata = storage.head_object(bucket, object_name)
            target = labeled_object_name(
                object_name.rsplit("/", 1)[-1],
                metadata.content_type or "application/octet-stream",
            )
            storage.copy_object(
                source_bucket=bucket,
                source_object=object_name,
                target_bucket=settings.S3_EXTERNAL_BUCKET,
                target_object=target,
                content_type=metadata.content_type or "application/octet-stream",
                metadata=label_metadata(
                    sidecar.product_id or "", sidecar.product_name or ""
                ),
            )
            storage.delete_objects(
                bucket,
                [object_name, sidecar_object_name(object_name)],
            )
            moved_count += 1
            if body.selection == "explicit":
                moved.append(target)
        return {"moved": moved_count, "object_names": moved}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/images/finalize/{job_id}", response_model=FinalizeJobPublic)
def finalize_job_status(job_id: str) -> FinalizeJobPublic:
    try:
        job = Job.fetch(job_id, connection=get_autolabel_queue().connection)
        rq_status = job.get_status(refresh=True)
        meta = job.meta or {}
        if rq_status in {"failed", "stopped", "canceled"}:
            public_status = "failed"
        elif rq_status == "finished":
            public_status = "completed"
        elif rq_status in {"started", "busy"}:
            public_status = "processing"
        else:
            public_status = "queued"
        return FinalizeJobPublic(
            job_id=job.id,
            status=public_status,
            moved=int(meta.get("moved") or 0),
            error=str(meta.get("error")) if meta.get("error") else None,
        )
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Move job not found") from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Move queue is unavailable"
        ) from exc
