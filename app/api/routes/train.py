from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep
from app.core.training import check_mlflow, train_classifier

router = APIRouter(prefix="/train", tags=["train"], dependencies=[SuperuserDep])


class TrainRequest(BaseModel):
    yolo_datasets: list[str] = Field(
        default=[], description="MinIO prefixes of YOLO datasets (crops)"
    )
    csv_datasets: list[str] = Field(
        default=[], description="MinIO prefixes of CSV datasets (whole images)"
    )
    image_size: int = Field(default=160, description="Input image size (square)")
    epochs: int = Field(default=12, description="Max training epochs")
    batch_size: int = Field(default=16, description="Training batch size")
    validation_ratio: float = Field(
        default=0.2, description="Fraction of data for validation"
    )


@router.post("/classifier")
async def classifier(body: TrainRequest) -> dict:
    """Train a classifier on specified YOLO datasets from MinIO.

    Returns 503 if MLflow is not reachable.
    """
    try:
        await run_in_threadpool(check_mlflow)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        result = await run_in_threadpool(
            train_classifier,
            body.yolo_datasets,
            body.csv_datasets or None,
            image_size=body.image_size,
            epochs=body.epochs,
            batch_size=body.batch_size,
            validation_ratio=body.validation_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result
