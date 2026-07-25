import httpx
from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.api.deps import SuperuserToken
from app.core.label_studio import (
    export_dataset,
    get_user_label_studio_api_key,
    list_projects,
    sync_label_studio,
)

router = APIRouter(prefix="/label-studio", tags=["label-studio"])


@router.get("/projects")
async def get_projects(access_token: SuperuserToken) -> list[dict]:
    try:
        api_key = await run_in_threadpool(get_user_label_studio_api_key, access_token)
        return await run_in_threadpool(list_projects, api_key)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sync")
async def sync(access_token: SuperuserToken) -> dict:
    """Sync S3-compatible object storage images with Label Studio projects.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        api_key = await run_in_threadpool(get_user_label_studio_api_key, access_token)
        result = await run_in_threadpool(sync_label_studio, api_key)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/export")
async def export(
    access_token: SuperuserToken,
    project_title: str = Query(..., description="Label Studio project title"),
    release_name: str | None = Query(
        default=None, description="Release name (defaults to timestamp)"
    ),
) -> dict:
    """Export reviewed annotations as YOLO dataset and upload to S3-compatible object storage.

    Returns 503 if Label Studio is unreachable.
    """
    try:
        api_key = await run_in_threadpool(get_user_label_studio_api_key, access_token)
        result = await run_in_threadpool(
            export_dataset, project_title, api_key, release_name
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Label Studio is not reachable")
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
