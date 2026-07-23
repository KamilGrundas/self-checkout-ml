from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep
from app.core.inference import classifier_model_store, shelf_model_store
from app.schemas import (
    ModelActivatedPublic,
    ModelVersionPublic,
    PredictionPublic,
    SetModelRequest,
)

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/classify", response_model=PredictionPublic)
async def classify_image(file: UploadFile = File(...)) -> PredictionPublic:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    scores, run_id = await run_in_threadpool(
        classifier_model_store.predict,
        image_bytes,
    )
    return PredictionPublic(scores=scores, run_id=run_id)


@router.post("/detect", response_model=PredictionPublic, dependencies=[SuperuserDep])
async def detect_image(file: UploadFile = File(...)) -> PredictionPublic:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    scores, run_id = await run_in_threadpool(
        shelf_model_store.predict,
        image_bytes,
    )
    return PredictionPublic(scores=scores, run_id=run_id)


@router.get(
    "/classify-models",
    response_model=list[ModelVersionPublic],
    dependencies=[SuperuserDep],
)
async def list_classify_models() -> list[ModelVersionPublic]:
    versions = await run_in_threadpool(classifier_model_store.list_versions)
    return [ModelVersionPublic(**v) for v in versions]


@router.get(
    "/detect-models",
    response_model=list[ModelVersionPublic],
    dependencies=[SuperuserDep],
)
async def list_detect_models() -> list[ModelVersionPublic]:
    versions = await run_in_threadpool(shelf_model_store.list_versions)
    return [ModelVersionPublic(**v) for v in versions]


@router.post(
    "/set-classify-model",
    response_model=ModelActivatedPublic,
    dependencies=[SuperuserDep],
)
async def set_classify_model(body: SetModelRequest) -> ModelActivatedPublic:
    result = await run_in_threadpool(classifier_model_store.set_version, body.version)
    return ModelActivatedPublic(**result)


@router.post(
    "/set-detect-model",
    response_model=ModelActivatedPublic,
    dependencies=[SuperuserDep],
)
async def set_detect_model(body: SetModelRequest) -> ModelActivatedPublic:
    result = await run_in_threadpool(shelf_model_store.set_version, body.version)
    return ModelActivatedPublic(**result)
