from fastapi import APIRouter

router = APIRouter(prefix="/utils", tags=["utils"])


@router.get("/health-check/", response_model=bool)
def health_check() -> bool:
    return True
