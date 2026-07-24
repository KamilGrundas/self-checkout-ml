from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep
from app.core.config import settings
from app.core.datasets import delete_dataset, list_datasets
from app.core.object_storage import public_object_url, store_dataset_image
from app.schemas import StoredImagePublic

from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[SuperuserDep])


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
