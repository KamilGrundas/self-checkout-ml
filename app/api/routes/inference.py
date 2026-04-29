from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep
from app.core.inference import classifier_model_store, shelf_model_store
from app.schemas import ModelRefreshPublic, PredictionPublic

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


@router.post(
    "/refresh-classify-model",
    response_model=ModelRefreshPublic,
    dependencies=[SuperuserDep],
)
def refresh_classify_model() -> ModelRefreshPublic:
    refresh_result = classifier_model_store.refresh_latest_model()
    return ModelRefreshPublic(**refresh_result)


@router.post(
    "/refresh-detect-model",
    response_model=ModelRefreshPublic,
    dependencies=[SuperuserDep],
)
def refresh_detect_model() -> ModelRefreshPublic:
    refresh_result = shelf_model_store.refresh_latest_model()
    return ModelRefreshPublic(**refresh_result)
