import httpx
from fastapi import APIRouter, Header, HTTPException, Query
from typing_extensions import Annotated
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserDep
from app.core.label_studio import export_dataset, list_projects, sync_label_studio

router = APIRouter(
    prefix="/label-studio", tags=["label-studio"], dependencies=[SuperuserDep]
)

LabelStudioApiKey = Annotated[
    str,
    Header(
        alias="X-Label-Studio-Api-Key",
        min_length=1,
        description="Label Studio personal access token",
    ),
]


@router.get("/projects")
async def get_projects(api_key: LabelStudioApiKey) -> list[dict]:
    try:
        return await run_in_threadpool(list_projects, api_key)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sync")
async def sync(api_key: LabelStudioApiKey) -> dict:
    """Sync S3-compatible object storage images with Label Studio projects.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        result = await run_in_threadpool(sync_label_studio, api_key)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/export")
async def export(
    api_key: LabelStudioApiKey,
    project_title: str = Query(..., description="Label Studio project title"),
    release_name: str | None = Query(
        default=None, description="Release name (defaults to timestamp)"
    ),
) -> dict:
    """Export reviewed annotations as YOLO dataset and upload to S3-compatible object storage.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        result = await run_in_threadpool(
            export_dataset, project_title, api_key, release_name
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
