from io import BytesIO
from pathlib import Path
from typing import Any

from app.core import object_storage
from app.core.object_storage import S3ObjectStorage


class FakePaginator:
    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs == {"Bucket": "source", "Prefix": "images/"}
        return [{"Contents": [{"Key": "images/a.jpg", "Size": 4}]}]


class FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def head_bucket(self, **kwargs: Any) -> None:
        self.calls.append(("head_bucket", kwargs))

    def get_paginator(self, operation: str) -> FakePaginator:
        assert operation == "list_objects_v2"
        return FakePaginator()

    def put_object(self, **kwargs: Any) -> None:
        self.calls.append(("put_object", kwargs))

    def get_object(self, **kwargs: Any) -> dict[str, BytesIO]:
        self.calls.append(("get_object", kwargs))
        return {"Body": BytesIO(b"same-domain-data")}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_object", kwargs))
        return {"ContentType": "image/jpeg", "Metadata": {"product-id": "p1"}}

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete_objects", kwargs))
        return {}

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        self.calls.append(("download_file", (bucket, key, destination)))
        Path(destination).write_bytes(b"same-domain-data")

    def upload_file(
        self,
        source: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, str],
    ) -> None:
        self.calls.append(("upload_file", (source, bucket, key, ExtraArgs)))


def test_generic_s3_crud_preserves_domain_data_and_metadata(tmp_path: Path) -> None:
    client = FakeS3Client()
    storage = S3ObjectStorage(client, create_buckets=False)  # type: ignore[arg-type]

    objects = list(storage.list_objects("source", "images/"))
    storage.put_bytes(
        "target",
        objects[0].object_name,
        b"same-domain-data",
        content_type="image/jpeg",
        metadata={"product-id": "p1"},
    )
    assert storage.get_bytes("target", "images/a.jpg") == b"same-domain-data"
    metadata = storage.head_object("target", "images/a.jpg")
    destination = tmp_path / "a.jpg"
    storage.download_file("target", "images/a.jpg", destination)
    storage.upload_file(
        "target",
        "images/uploaded.jpg",
        destination,
        content_type="image/jpeg",
    )
    storage.delete_objects("target", ["images/a.jpg"])

    assert objects[0].size == 4
    assert metadata.content_type == "image/jpeg"
    assert metadata.metadata == {"product-id": "p1"}
    assert destination.read_bytes() == b"same-domain-data"
    assert any(call[0] == "upload_file" for call in client.calls)
    assert client.calls[-1][0] == "delete_objects"


def test_custom_endpoint_path_style_and_tls_are_configuration(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> FakeS3Client:
        captured["service"] = service
        captured.update(kwargs)
        return FakeS3Client()

    monkeypatch.setattr(object_storage.boto3, "client", fake_client)
    monkeypatch.setattr(
        object_storage.settings, "S3_ENDPOINT_URL", "https://custom.test"
    )
    monkeypatch.setattr(object_storage.settings, "S3_USE_SSL", True)
    monkeypatch.setattr(object_storage.settings, "S3_VERIFY_TLS", False)
    monkeypatch.setattr(object_storage.settings, "S3_FORCE_PATH_STYLE", True)
    object_storage.get_object_storage.cache_clear()

    object_storage.get_object_storage()

    assert captured["service"] == "s3"
    assert captured["endpoint_url"] == "https://custom.test"
    assert captured["use_ssl"] is True
    assert captured["verify"] is False
    assert captured["config"].s3["addressing_style"] == "path"
    object_storage.get_object_storage.cache_clear()
