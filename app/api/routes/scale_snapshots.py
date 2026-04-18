from urllib.parse import unquote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.object_storage import (
    build_object_name,
    ensure_bucket_exists,
    get_minio_client,
    public_object_url,
    store_session_snapshot,
)
from app.schemas import SessionSnapshotListPublic, SessionSnapshotPublic

router = APIRouter(prefix="/checkout-sessions", tags=["scale-snapshots"])


@router.post("/{session_id}/scale-snapshots", response_model=SessionSnapshotPublic)
async def upload_scale_snapshot(
    session_id: str,
    capture_index: int = Form(..., ge=0),
    product_id: str | None = Form(default=None),
    product_name: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> SessionSnapshotPublic:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file")

    filename, object_name = store_session_snapshot(
        session_id=session_id,
        capture_index=capture_index,
        product_id=product_id,
        product_name=product_name,
        filename=file.filename,
        content_type=file.content_type,
        data=data,
        bucket_name=settings.ML_MINIO_SCALE_BUCKET_NAME,
    )

    return SessionSnapshotPublic(
        session_id=session_id,
        capture_index=capture_index,
        product_id=product_id,
        product_name=product_name,
        filename=filename,
        object_name=object_name,
        image_url=public_object_url(object_name, settings.ML_MINIO_SCALE_BUCKET_NAME),
        content_type=file.content_type,
        size=len(data),
    )


@router.get("/{session_id}/scale-snapshots", response_model=SessionSnapshotListPublic)
def list_scale_snapshots(session_id: str) -> SessionSnapshotListPublic:
    ensure_bucket_exists(settings.ML_MINIO_SCALE_BUCKET_NAME)
    client = get_minio_client()
    prefix = build_object_name(session_id, "")
    snapshots: list[SessionSnapshotPublic] = []

    for obj in client.list_objects(
        settings.ML_MINIO_SCALE_BUCKET_NAME, prefix=prefix, recursive=True
    ):
        filename = obj.object_name.rsplit("/", 1)[-1]
        capture_prefix = filename.split("-", 1)[0]
        if not capture_prefix.isdigit():
            continue

        capture_index = int(capture_prefix)
        content_type = obj.content_type or "application/octet-stream"
        stat = client.stat_object(settings.ML_MINIO_SCALE_BUCKET_NAME, obj.object_name)
        metadata = stat.metadata or {}
        product_id = unquote(metadata.get("x-amz-meta-product-id") or "") or None
        product_name = unquote(metadata.get("x-amz-meta-product-name") or "") or None

        snapshots.append(
            SessionSnapshotPublic(
                session_id=session_id,
                capture_index=capture_index,
                product_id=product_id,
                product_name=product_name,
                filename=filename,
                object_name=obj.object_name,
                image_url=public_object_url(
                    obj.object_name, settings.ML_MINIO_SCALE_BUCKET_NAME
                ),
                content_type=content_type,
                size=obj.size,
            )
        )

    snapshots.sort(key=lambda item: item.capture_index)
    return SessionSnapshotListPublic(data=snapshots)
