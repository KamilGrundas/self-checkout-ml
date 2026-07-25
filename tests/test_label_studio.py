from typing import Any

import pytest
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


def test_user_api_key_is_loaded_from_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BackendResponse:
        status_code = 200
        is_success = True

        def json(self) -> dict[str, str]:
            return {"api_key": "saved-user-token"}

    class BackendClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "BackendClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def get(self, url: str, **kwargs: Any) -> BackendResponse:
            assert url.endswith("/api/v1/users/me/label-studio/api-key")
            assert kwargs["headers"] == {"Authorization": "Bearer admin-token"}
            return BackendResponse()

    monkeypatch.setattr(label_studio.httpx, "Client", BackendClient)

    assert (
        label_studio.get_user_label_studio_api_key("admin-token") == "saved-user-token"
    )


def test_classify_export_downloads_label_studio_managed_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image_content = b"local-label-studio-image"

    class ImageResponse:
        content = image_content

        def raise_for_status(self) -> None:
            return None

    class ImageClient:
        def __enter__(self) -> "ImageClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def get(self, path: str, **kwargs: Any) -> ImageResponse:
            assert path == "/data/upload/4/product.jpg"
            assert kwargs == {"follow_redirects": True}
            return ImageResponse()

    monkeypatch.setattr(
        label_studio,
        "_client",
        lambda headers: ImageClient(),
    )
    tasks = [
        {
            "data": {"image": "/data/upload/4/product.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "choices",
                            "value": {"choices": ["apple"]},
                        }
                    ]
                }
            ],
        }
    ]

    rows = label_studio._parse_classify_tasks(
        tasks,
        tmp_path,
        {"Authorization": "Token saved-user-token"},
    )

    assert rows == [("product.jpg", "apple")]
    assert (tmp_path / "product.jpg").read_bytes() == image_content
