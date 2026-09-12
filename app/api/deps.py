from typing import Annotated, Any

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl="/api/v1/login/access-token", auto_error=False
)
TokenDep = Annotated[str | None, Depends(reusable_oauth2)]


def backend_identity(token: str) -> dict[str, Any]:
    try:
        with httpx.Client(
            timeout=10, trust_env=False, follow_redirects=False
        ) as client:
            response = client.post(
                f"{settings.BACKEND_URL.rstrip('/')}/api/v1/login/test-token",
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code in (400, 401, 403, 404):
            raise HTTPException(401, "Invalid or expired token")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not data.get("id"):
            raise ValueError("Invalid identity response")
        return data
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Authentication service unavailable")


def get_current_superuser(token: TokenDep, request: Request = None) -> str:
    raw_key = request.headers.get("X-API-Key") if request else None
    if raw_key and token:
        raise HTTPException(400, "Use one authentication method per request")
    token = raw_key or token
    if not token:
        raise HTTPException(401, "Authentication required")
    if not backend_identity(token).get("is_superuser"):
        raise HTTPException(403, "Not enough privileges")
    return token


def require_invoke(request: Request) -> None:
    api_key = request.headers.get("X-API-Key")
    authorization = request.headers.get("Authorization", "")
    if api_key and authorization:
        raise HTTPException(400, "Use one authentication method per request")
    if api_key:
        try:
            with httpx.Client(
                timeout=10, trust_env=False, follow_redirects=False
            ) as client:
                response = client.post(
                    f"{settings.BACKEND_URL.rstrip('/')}/api/v1/login/api-key/check",
                    params={"scope": "ml:invoke"},
                    headers={"X-API-Key": api_key},
                )
            if response.status_code in (401, 403):
                raise HTTPException(response.status_code, "API access denied")
            response.raise_for_status()
            return
        except httpx.HTTPError:
            raise HTTPException(503, "Authentication service unavailable")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Authentication required")
    get_current_superuser(token)


def require_checkout_snapshot_invoke(session_id: str, request: Request) -> None:
    api_key = request.headers.get("X-API-Key")
    authorization = request.headers.get("Authorization", "")
    if api_key and authorization:
        raise HTTPException(400, "Use one authentication method per request")
    if not api_key:
        raise HTTPException(401, "Checkout counter API key required")
    try:
        with httpx.Client(
            timeout=10, trust_env=False, follow_redirects=False
        ) as client:
            response = client.post(
                f"{settings.BACKEND_URL.rstrip('/')}/api/v1/login/checkout-key/check",
                params={"scope": "ml:invoke", "checkout_session_id": session_id},
                headers={"X-API-Key": api_key},
            )
        if response.status_code in (400, 401, 403, 404):
            raise HTTPException(
                response.status_code, "Checkout session API access denied"
            )
        response.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(503, "Authentication service unavailable")


SuperuserDep = Depends(get_current_superuser)
SuperuserToken = Annotated[str, Depends(get_current_superuser)]
InvokeDep = Depends(require_invoke)
CheckoutSnapshotInvokeDep = Depends(require_checkout_snapshot_invoke)
