from datetime import datetime
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep, SuperuserToken
from app.core.config import settings
from app.core.datasets import delete_dataset, list_datasets
from app.core.autolabel_queue import (
    JOB_RETENTION_SECONDS,
    JOB_TIMEOUT,
    get_autolabel_queue,
)
from app.core.autolabel import (
    decode_cursor,
    encode_cursor,
    is_supported_image,
    load_catalog,
)
from app.core.labeled_images import (
    MAX_SELECTED_IMAGES,
    export_csv_dataset,
    label_metadata,
    labeled_object_name,
    read_label,
)
from app.core.image_duplicates import delete_duplicate_images, find_duplicate_images
from app.core.object_storage import (
    get_object_storage,
    public_object_url,
    store_dataset_image,
)
from app.schemas import StoredImagePublic

from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[SuperuserDep])

MAX_UPLOAD_IMAGES = 100
MAX_UPLOAD_IMAGE_BYTES = 20 * 1024 * 1024


class LabeledImagePublic(BaseModel):
    object_name: str
    image_url: str
    size: int
    etag: str | None
    captured_at: datetime | None
    product_id: str
    product_name: str


class LabeledImagesPage(BaseModel):
    data: list[LabeledImagePublic]
    next_cursor: str | None


class LabelCountPublic(BaseModel):
    product_id: str
    product_name: str
    count: int


class LabelCountsPublic(BaseModel):
    total: int
    labels: list[LabelCountPublic]


class ImageSelection(BaseModel):
    object_names: list[str] = Field(min_length=1, max_length=MAX_SELECTED_IMAGES)
    release_name: str = Field(default="labeled-images", min_length=1, max_length=80)


class ImageLabelUpdate(BaseModel):
    object_name: str = Field(min_length=1, max_length=1024)
    product_id: str = Field(min_length=1, max_length=128)


class ImageDeleteRequest(BaseModel):
    object_names: list[str] = Field(min_length=1, max_length=MAX_SELECTED_IMAGES)


class DuplicateScanPublic(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    processed: int = 0
    total: int = 0
    progress: float = 0
    duplicate_count: int = 0
    similarity_threshold: float = 0.99
    error: str | None = None


def _duplicate_scan_public(job: Job) -> DuplicateScanPublic:
    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}
    if rq_status in {"failed", "stopped", "canceled"}:
        scan_status = "failed"
    elif rq_status in {"started", "busy"}:
        scan_status = str(meta.get("status") or "processing")
    elif rq_status == "finished":
        scan_status = "completed"
    else:
        scan_status = "queued"
    return DuplicateScanPublic(
        job_id=job.id,
        status=scan_status,  # type: ignore[arg-type]
        processed=int(meta.get("processed") or 0),
        total=int(meta.get("total") or 0),
        progress=float(meta.get("progress") or 0),
        duplicate_count=int(meta.get("duplicate_count") or 0),
        similarity_threshold=float(meta.get("similarity_threshold") or 0.99),
        error=str(meta.get("error")) if meta.get("error") else None,
    )


def _catalog_product(access_token: str, product_id: str):
    try:
        product = next(
            (
                item
                for item in load_catalog(access_token)
                if item.product_id == product_id
            ),
            None,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if product is None:
        raise HTTPException(status_code=422, detail="Selected product does not exist")
    return product


async def _upload_raw_image(
    *,
    bucket_name: str,
    prefix: str,
    file: UploadFile,
) -> StoredImagePublic:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file")
    if len(data) > MAX_UPLOAD_IMAGE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Image is too large: {file.filename}"
        )

    object_name = store_dataset_image(
        bucket_name=bucket_name,
        prefix=prefix,
        filename=file.filename,
        content_type=file.content_type,
        data=data,
    )
    return StoredImagePublic(
        bucket_name=bucket_name,
        object_name=object_name,
        image_url=public_object_url(object_name, bucket_name),
        content_type=file.content_type,
        size=len(data),
    )


@router.post("/shelf-images", response_model=list[StoredImagePublic])
async def upload_shelf_images(
    files: list[UploadFile] = File(...),
) -> list[StoredImagePublic]:
    return [
        await _upload_raw_image(
            bucket_name=settings.S3_SHELF_BUCKET,
            prefix="raw/shelf",
            file=file,
        )
        for file in files
    ]


@router.post("/scale-images", response_model=list[StoredImagePublic])
async def upload_scale_images(
    files: list[UploadFile] = File(...),
) -> list[StoredImagePublic]:
    if len(files) > MAX_UPLOAD_IMAGES:
        raise HTTPException(
            status_code=413,
            detail=f"A maximum of {MAX_UPLOAD_IMAGES} images can be imported at once",
        )
    return [
        await _upload_raw_image(
            bucket_name=settings.S3_SCALE_BUCKET,
            prefix="raw/scale",
            file=file,
        )
        for file in files
    ]


@router.post("/external-images", response_model=list[StoredImagePublic])
async def upload_external_images(
    files: list[UploadFile] = File(...),
) -> list[StoredImagePublic]:
    return [
        await _upload_raw_image(
            bucket_name=settings.S3_EXTERNAL_BUCKET,
            prefix="raw/uploaded",
            file=file,
        )
        for file in files
    ]


@router.get("/images", response_model=LabeledImagesPage)
async def list_labeled_images(
    cursor: str | None = None,
    page_size: int = Query(default=100, ge=1, le=100),
    label_product_id: str | None = Query(default=None, max_length=128),
) -> LabeledImagesPage:
    storage = get_object_storage()
    try:
        next_token = decode_cursor(cursor) if cursor else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid image cursor") from exc
    images: list[LabeledImagePublic] = []
    while len(images) < page_size:
        objects, next_token = storage.list_objects_page(
            settings.S3_EXTERNAL_BUCKET,
            prefix="labeled/",
            max_keys=page_size - len(images),
            continuation_token=next_token,
        )
        for item in objects:
            metadata = storage.head_object(
                settings.S3_EXTERNAL_BUCKET, item.object_name
            )
            if not is_supported_image(item.object_name, metadata.content_type):
                continue
            product_id, product_name = read_label(metadata)
            if not product_id or not product_name:
                continue
            if label_product_id and product_id != label_product_id:
                continue
            images.append(
                LabeledImagePublic(
                    object_name=item.object_name,
                    image_url="datasets/images/content",
                    size=item.size,
                    etag=item.etag,
                    captured_at=item.last_modified,
                    product_id=product_id,
                    product_name=product_name,
                )
            )
        if not next_token:
            break
    return LabeledImagesPage(
        data=images,
        next_cursor=encode_cursor(next_token) if next_token else None,
    )


@router.get("/images/label-counts", response_model=LabelCountsPublic)
async def labeled_image_counts() -> LabelCountsPublic:
    storage = get_object_storage()
    total = 0
    labels: dict[str, tuple[str, int]] = {}
    for item in storage.list_objects(settings.S3_EXTERNAL_BUCKET, prefix="labeled/"):
        metadata = storage.head_object(settings.S3_EXTERNAL_BUCKET, item.object_name)
        if not is_supported_image(item.object_name, metadata.content_type):
            continue
        product_id, product_name = read_label(metadata)
        if not product_id or not product_name:
            continue
        total += 1
        current_name, count = labels.get(product_id, (product_name, 0))
        labels[product_id] = (current_name, count + 1)
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


@router.get("/images/content")
async def get_labeled_image_content(
    object_name: str = Query(min_length=1, max_length=1024),
) -> Response:
    if not object_name.startswith("labeled/"):
        raise HTTPException(status_code=422, detail="Invalid labeled image object name")
    storage = get_object_storage()
    try:
        metadata = storage.head_object(settings.S3_EXTERNAL_BUCKET, object_name)
        body = storage.get_bytes(settings.S3_EXTERNAL_BUCKET, object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Labeled image not found") from exc
    return Response(
        content=body,
        media_type=metadata.content_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.post("/images/import", response_model=list[LabeledImagePublic])
async def import_labeled_images(
    access_token: SuperuserToken,
    product_id: str = Form(min_length=1, max_length=128),
    files: list[UploadFile] = File(...),
) -> list[LabeledImagePublic]:
    if not 1 <= len(files) <= 100:
        raise HTTPException(status_code=422, detail="Upload between 1 and 100 images")
    storage = get_object_storage()
    product = _catalog_product(access_token, product_id)
    result: list[LabeledImagePublic] = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {file.filename}"
            )
        body = await file.read()
        if not body:
            raise HTTPException(
                status_code=400, detail=f"Empty image file: {file.filename}"
            )
        if len(body) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail=f"Image is too large: {file.filename}"
            )
        object_name = labeled_object_name(file.filename, file.content_type)
        storage.put_bytes(
            settings.S3_EXTERNAL_BUCKET,
            object_name,
            body,
            content_type=file.content_type,
            metadata=label_metadata(product.product_id, product.name),
        )
        metadata = storage.head_object(settings.S3_EXTERNAL_BUCKET, object_name)
        result.append(
            LabeledImagePublic(
                object_name=object_name,
                image_url="datasets/images/content",
                size=len(body),
                etag=metadata.etag,
                captured_at=datetime.now().astimezone(),
                product_id=product.product_id,
                product_name=product.name,
            )
        )
    return result


@router.patch("/images/label", response_model=LabeledImagePublic)
async def update_labeled_image(
    body: ImageLabelUpdate, access_token: SuperuserToken
) -> LabeledImagePublic:
    if not body.object_name.startswith("labeled/"):
        raise HTTPException(status_code=422, detail="Invalid labeled image object name")
    storage = get_object_storage()
    product = _catalog_product(access_token, body.product_id)
    try:
        metadata = storage.head_object(settings.S3_EXTERNAL_BUCKET, body.object_name)
        storage.copy_object(
            source_bucket=settings.S3_EXTERNAL_BUCKET,
            source_object=body.object_name,
            target_bucket=settings.S3_EXTERNAL_BUCKET,
            target_object=body.object_name,
            content_type=metadata.content_type or "application/octet-stream",
            metadata=label_metadata(product.product_id, product.name),
        )
        refreshed = storage.head_object(settings.S3_EXTERNAL_BUCKET, body.object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Labeled image not found") from exc
    return LabeledImagePublic(
        object_name=body.object_name,
        image_url="datasets/images/content",
        size=refreshed.size,
        etag=refreshed.etag,
        captured_at=None,
        product_id=product.product_id,
        product_name=product.name,
    )


@router.delete("/images")
async def delete_labeled_images(body: ImageDeleteRequest) -> dict[str, int]:
    storage = get_object_storage()
    selected = list(dict.fromkeys(body.object_names))
    try:
        for object_name in selected:
            if not object_name.startswith("labeled/"):
                raise ValueError("Invalid labeled image object name")
            storage.head_object(settings.S3_EXTERNAL_BUCKET, object_name)
        storage.delete_objects(settings.S3_EXTERNAL_BUCKET, selected)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Labeled image not found") from exc
    return {"deleted": len(selected)}


@router.post("/images/export")
async def export_labeled_images(body: ImageSelection) -> dict:
    storage = get_object_storage()
    selected = list(dict.fromkeys(body.object_names))
    rows: list[tuple[str, str, str]] = []
    try:
        for object_name in selected:
            if not object_name.startswith("labeled/"):
                raise ValueError("Invalid labeled image object name")
            metadata = storage.head_object(settings.S3_EXTERNAL_BUCKET, object_name)
            _, product_name = read_label(metadata)
            if not product_name:
                raise ValueError(f"Image has no label: {object_name}")
            rows.append((settings.S3_EXTERNAL_BUCKET, object_name, product_name))
        return await run_in_threadpool(
            export_csv_dataset, release_name=body.release_name, images=rows
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/images/duplicates", response_model=DuplicateScanPublic)
async def start_duplicate_scan() -> DuplicateScanPublic:
    try:
        job = get_autolabel_queue().enqueue(
            find_duplicate_images,
            job_id=str(uuid4()),
            job_timeout=JOB_TIMEOUT,
            result_ttl=JOB_RETENTION_SECONDS,
            failure_ttl=JOB_RETENTION_SECONDS,
            meta={"status": "queued", "processed": 0, "total": 0, "progress": 0},
            description="Detect near-duplicate labeled images",
        )
        return _duplicate_scan_public(job)
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Duplicate scan queue is unavailable"
        ) from exc


@router.get("/images/duplicates/{job_id}", response_model=DuplicateScanPublic)
async def duplicate_scan_status(job_id: str) -> DuplicateScanPublic:
    try:
        job = Job.fetch(job_id, connection=get_autolabel_queue().connection)
        return _duplicate_scan_public(job)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Duplicate scan not found") from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Duplicate scan queue is unavailable"
        ) from exc


@router.delete("/images/duplicates/{job_id}")
async def remove_detected_duplicates(job_id: str) -> dict[str, int]:
    try:
        job = Job.fetch(job_id, connection=get_autolabel_queue().connection)
        if job.get_status(refresh=True) != "finished":
            raise HTTPException(
                status_code=409, detail="Duplicate scan is not completed"
            )
        result = job.return_value() or {}
        return delete_duplicate_images(list(result.get("duplicate_object_names") or []))
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Duplicate scan not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RedisError as exc:
        raise HTTPException(
            status_code=503, detail="Duplicate scan queue is unavailable"
        ) from exc


@router.get("/training")
async def get_training_datasets() -> list[dict]:
    """List all exported YOLO datasets in the training-data bucket."""
    return await run_in_threadpool(list_datasets)


@router.delete("/training/{project_slug}/{release_name}")
async def remove_training_dataset(project_slug: str, release_name: str) -> dict:
    """Delete a dataset release from the training-data bucket."""
    try:
        return await run_in_threadpool(delete_dataset, project_slug, release_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/upload-ui", response_class=HTMLResponse)
async def upload_ui():
    content = """
    <body>
        <h2>Shelf Images</h2>
        <form action="/api/v1/datasets/shelf-images" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" value="Upload Shelf Images">
        </form>

        <h2>Scale Images</h2>
        <form action="/api/v1/datasets/scale-images" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" value="Upload Scale Images">
        </form>

        <h2>External Images</h2>
        <form action="/api/v1/datasets/external-images" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" value="Upload External Images">
        </form>
    </body>
    """
    return HTMLResponse(content=content)
