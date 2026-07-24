from typing import Any

import pytest
from fastapi import FastAPI

from app.api.routes.label_studio import router
from app.core import label_studio


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, list[dict[str, Any]]]:
        return {"results": [{"id": 7, "title": "scale-products"}]}


class FakeClient:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def get(self, path: str, **_kwargs: Any) -> FakeResponse:
        assert path == "/api/projects/"
        return FakeResponse()


def test_list_projects_uses_request_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[FakeClient] = []

    def fake_client(headers: dict[str, str]) -> FakeClient:
        client = FakeClient(headers)
        clients.append(client)
        return client

    monkeypatch.setattr(label_studio, "_client", fake_client)

    projects = label_studio.list_projects("request-token")

    assert projects == [{"id": 7, "title": "scale-products"}]
    assert clients[0].headers == {"Authorization": "Token request-token"}
    assert clients[1].headers == {"Authorization": "Token request-token"}


def test_blank_request_api_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        label_studio.list_projects("  ")


def test_label_studio_routes_require_api_key_header() -> None:
    app = FastAPI()
    app.include_router(router)
    schema = app.openapi()

    operations = (
        schema["paths"]["/label-studio/projects"]["get"],
        schema["paths"]["/label-studio/sync"]["post"],
        schema["paths"]["/label-studio/export"]["post"],
    )
    for operation in operations:
        api_key_parameter = next(
            parameter
            for parameter in operation["parameters"]
            if parameter["name"] == "X-Label-Studio-Api-Key"
        )
        assert api_key_parameter["in"] == "header"
        assert api_key_parameter["required"] is True
