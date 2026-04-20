import httpx
from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.core.label_studio import export_dataset, sync_label_studio

router = APIRouter(prefix="/label-studio", tags=["label-studio"])


@router.post("/sync")
async def sync() -> dict:
    """Sync MinIO images with Label Studio projects.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        result = await run_in_threadpool(sync_label_studio)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/export")
async def export(
    project_title: str = Query(..., description="Label Studio project title"),
    release_name: str | None = Query(
        default=None, description="Release name (defaults to timestamp)"
    ),
) -> dict:
    """Export reviewed annotations as YOLO dataset and upload to MinIO.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        result = await run_in_threadpool(export_dataset, project_title, release_name)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
