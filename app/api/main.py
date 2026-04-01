from fastapi import APIRouter

from app.api.routes import session_snapshots, utils

api_router = APIRouter()
api_router.include_router(utils.router)
api_router.include_router(session_snapshots.router)
