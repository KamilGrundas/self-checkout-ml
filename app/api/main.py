from fastapi import APIRouter

from app.api.routes import (
    datasets,
    inference,
    label_studio,
    scale_snapshots,
    shelf_snapshots,
    train,
    utils,
)

api_router = APIRouter()
api_router.include_router(utils.router)
api_router.include_router(shelf_snapshots.router)
api_router.include_router(scale_snapshots.router)
api_router.include_router(datasets.router)
api_router.include_router(inference.router)
api_router.include_router(label_studio.router)
api_router.include_router(train.router)
