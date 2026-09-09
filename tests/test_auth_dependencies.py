import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api import deps


def mock_backend(monkeypatch, status, body):
    real_client = httpx.Client
    monkeypatch.setattr(
        deps.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=body)
            ),
            **kwargs,
        ),
    )


@pytest.mark.parametrize("status", [401, 403])
def test_backend_rejection_cannot_be_bypassed(monkeypatch, status):
    mock_backend(monkeypatch, status, {"detail": "denied"})
    with pytest.raises(HTTPException) as error:
        deps.get_current_superuser("untrusted-token")
    assert error.value.status_code == 401


def test_backend_role_is_authoritative(monkeypatch):
    mock_backend(monkeypatch, 200, {"id": "user-id", "is_superuser": False})
    with pytest.raises(HTTPException) as error:
        deps.get_current_superuser("token-with-possibly-stale-admin-claim")
    assert error.value.status_code == 403


def test_key_without_invoke_scope_is_rejected(monkeypatch):
    mock_backend(monkeypatch, 403, {"detail": "missing scope"})
    request = Request({"type": "http", "headers": [(b"x-api-key", b"sck_test")]})
    with pytest.raises(HTTPException) as error:
        deps.require_invoke(request)
    assert error.value.status_code == 403


def test_unavailable_backend_fails_closed(monkeypatch):
    mock_backend(monkeypatch, 503, {})
    with pytest.raises(HTTPException) as error:
        deps.get_current_superuser("untrusted-token")
    assert error.value.status_code == 503


@pytest.mark.parametrize("is_admin,status", [(False, 403), (True, None)])
def test_human_invoke_requires_admin(monkeypatch, is_admin, status):
    mock_backend(monkeypatch, 200, {"id": "user-id", "is_superuser": is_admin})
    request = Request(
        {"type": "http", "headers": [(b"authorization", b"Bearer test-token")]}
    )
    if status:
        with pytest.raises(HTTPException) as error:
            deps.require_invoke(request)
        assert error.value.status_code == status
    else:
        deps.require_invoke(request)


def test_all_ml_routes_reject_catalog_reader(monkeypatch):
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.api.main import api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    import re

    mock_backend(monkeypatch, 200, {"id": "user-id", "is_superuser": False})
    checked = 0
    with TestClient(app) as client:
        for template, operations in app.openapi()["paths"].items():
            if template.endswith("/health-check/"):
                continue
            path = re.sub(r"\{[^}]+\}", "test-id", template)
            for method in operations:
                response = client.request(
                    method,
                    path,
                    headers={"Authorization": "Bearer test-token"},
                    json={},
                )
                checked += 1
                assert response.status_code == 403, (method, path, response.status_code)

    assert checked >= 30


@pytest.mark.parametrize("admin", [False, True])
def test_role_api_key_uses_backend_identity(monkeypatch, admin):
    real_client = httpx.Client

    def handle(request):
        assert request.headers["Authorization"] == "Bearer sck_test_role"
        return httpx.Response(200, json={"id": "owner", "is_superuser": admin})

    monkeypatch.setattr(
        deps.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    request = Request({"type": "http", "headers": [(b"x-api-key", b"sck_test_role")]})
    if admin:
        assert deps.get_current_superuser(None, request) == "sck_test_role"
    else:
        with pytest.raises(HTTPException) as error:
            deps.get_current_superuser(None, request)
        assert error.value.status_code == 403
    with pytest.raises(HTTPException) as error:
        deps.get_current_superuser("another-token", request)
    assert error.value.status_code == 400
