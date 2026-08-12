from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core import autolabel, autolabel_worker
from app.core.autolabel import (
    AutolabelConfiguration,
    AutolabelSidecar,
    CatalogCandidate,
    InferenceParseError,
    InferenceRequestError,
)
from app.core.object_storage import S3Object, S3ObjectMetadata

ORIGINAL_HTTPX_CLIENT = httpx.Client


def configuration(**updates: Any) -> AutolabelConfiguration:
    values = {
        "endpoint_url": "http://vlm.test/v1/files/inference",
        "max_tokens": 512,
        "connect_timeout_seconds": 5,
        "read_timeout_seconds": 120,
        "configured": True,
    }
    values.update(updates)
    return AutolabelConfiguration(**values)


def catalog() -> list[CatalogCandidate]:
    return [
        CatalogCandidate(
            key="P0001",
            product_id="11111111-1111-1111-1111-111111111111",
            name="Jabłko",
            category="Owoce",
        ),
        CatalogCandidate(
            key="P0002",
            product_id="22222222-2222-2222-2222-222222222222",
            name="Banan",
            category="Owoce",
        ),
    ]


@pytest.mark.parametrize(
    ("body", "state", "candidate_key"),
    [
        (b'{"candidate_key":"P0001"}', "matched", "P0001"),
        (b'{"candidate_key":null}', "unmatched", None),
        (b'{"candidate_key":"P9999"}', "unmatched", None),
    ],
)
def test_parser_accepts_direct_json_and_strict_candidate_keys(
    body: bytes,
    state: str,
    candidate_key: str | None,
) -> None:
    result = autolabel.parse_inference_response(body, {"P0001", "P0002"})

    assert result.state == state
    assert result.candidate_key == candidate_key
    assert len(result.response_sha256) == 64
    assert result.response_preview.startswith('{"candidate_key":')


def test_parser_accepts_observed_inference_envelope_with_fenced_json() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "inference_success_envelope.json"
    ).read_bytes()

    result = autolabel.parse_inference_response(fixture, {"P0001"})

    assert result.state == "unmatched"
    assert result.candidate_key is None


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b'{"response":"Jablko"}',
        b'{"response":"{\\"candidate_key\\":\\"P0001\\",\\"name\\":\\"Jablko\\"}"}',
        b'{"response":"{\\"candidate_key\\":\\"P0001\\"}","extra":true}',
        b'{"candidate_key":"P0001","name":"Jablko"}',
        b'{"candidate_key":"P0001"} trailing text',
    ],
)
def test_parser_rejects_invalid_json_descriptions_and_extra_keys(body: bytes) -> None:
    with pytest.raises(InferenceParseError):
        autolabel.parse_inference_response(body, {"P0001"})


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> None:
    transport = httpx.MockTransport(handler)

    def safe_factory(**kwargs):
        kwargs["transport"] = transport
        return ORIGINAL_HTTPX_CLIENT(**kwargs)

    monkeypatch.setattr(autolabel.httpx, "Client", safe_factory)


def test_inference_sends_exact_multipart_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        captured["body"] = body
        captured["content_type"] = request.headers["content-type"]
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={"candidate_key": "P0001"},
        )

    _install_transport(monkeypatch, handler)
    result = autolabel.call_inference(
        configuration=configuration(),
        object_name="sessions/session/captures/0001-product.jpg",
        content_type="image/jpeg",
        image_bytes=b"real-image-bytes",
        prompt="test prompt",
        allowed_keys={"P0001"},
    )

    body = captured["body"]
    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="prompt"' in body
    assert b"test prompt" in body
    assert b'name="max_tokens"' in body
    assert b"512" in body
    assert b'name="image"; filename="0001-product.jpg"' in body
    assert b"Content-Type: image/jpeg" in body
    assert b"real-image-bytes" in body
    assert result.candidate_key == "P0001"


@pytest.mark.parametrize("status_code", [400, 500])
def test_inference_rejects_http_errors(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            status_code,
            headers={"Content-Type": "application/json"},
            json={"error": "safe"},
        ),
    )
    with pytest.raises(InferenceRequestError, match=f"HTTP {status_code}"):
        autolabel.call_inference(
            configuration=configuration(),
            object_name="image.jpg",
            content_type="image/jpeg",
            image_bytes=b"image",
            prompt="prompt",
            allowed_keys={"P0001"},
        )


def test_inference_does_not_retry_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    _install_transport(monkeypatch, handler)
    with pytest.raises(InferenceRequestError, match="read timeout"):
        autolabel.call_inference(
            configuration=configuration(),
            object_name="image.jpg",
            content_type="image/jpeg",
            image_bytes=b"image",
            prompt="prompt",
            allowed_keys={"P0001"},
        )
    assert calls == 1


def test_inference_retries_connection_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    _install_transport(monkeypatch, handler)
    monkeypatch.setattr(autolabel.time, "sleep", lambda _: None)
    with pytest.raises(InferenceRequestError, match="connect"):
        autolabel.call_inference(
            configuration=configuration(),
            object_name="image.jpg",
            content_type="image/jpeg",
            image_bytes=b"image",
            prompt="prompt",
            allowed_keys={"P0001"},
        )
    assert calls == 2


def test_inference_rejects_oversized_and_non_json_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b"x" * (autolabel.MAX_RESPONSE_BYTES + 1),
        ),
    )
    with pytest.raises(InferenceRequestError, match="too large"):
        autolabel.call_inference(
            configuration=configuration(),
            object_name="image.jpg",
            content_type="image/jpeg",
            image_bytes=b"image",
            prompt="prompt",
            allowed_keys={"P0001"},
        )

    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            content=b"{}",
        ),
    )
    with pytest.raises(InferenceRequestError, match="non-JSON"):
        autolabel.call_inference(
            configuration=configuration(),
            object_name="image.jpg",
            content_type="image/jpeg",
            image_bytes=b"image",
            prompt="prompt",
            allowed_keys={"P0001"},
        )


class MemoryStorage:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], bytes] = {}

    def put_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        **kwargs,
    ) -> None:
        self.data[(bucket, object_name)] = data

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        if (bucket, object_name) not in self.data:
            from botocore.exceptions import ClientError

            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        return self.data[(bucket, object_name)]


def test_sidecar_round_trip_and_fingerprint_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MemoryStorage()
    monkeypatch.setattr(autolabel, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel.settings, "S3_SCALE_BUCKET", "scale")
    sidecar = AutolabelSidecar(
        bucket="scale",
        object_name="raw/scale/image.jpg",
        source_size=5,
        source_fingerprint="etag:one:size:5",
        product_id="product-id",
        product_name="Jabłko",
        state="matched",
        timestamp="2026-07-27T12:00:00Z",
        batch_id="batch",
        endpoint_url="http://vlm.test/inference",
        max_tokens=512,
        response_sha256="a" * 64,
        response_preview='{"candidate_key":"P0001"}',
    )

    autolabel.write_sidecar(sidecar)

    assert autolabel.read_sidecar(sidecar.object_name, "etag:one:size:5") == sidecar
    assert autolabel.read_sidecar(sidecar.object_name, "etag:two:size:5") is None


class FakeJob:
    def __init__(self) -> None:
        self.meta = {
            "status": "queued",
            "items": [
                {"object_name": "one.jpg", "status": "queued"},
                {"object_name": "two.jpg", "status": "queued"},
            ],
        }
        self.saved = 0

    def save_meta(self) -> None:
        self.saved += 1


def test_worker_continues_after_single_image_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = FakeJob()
    monkeypatch.setattr(autolabel_worker, "get_current_job", lambda: job)
    monkeypatch.setattr(
        autolabel_worker,
        "write_failed_sidecar",
        lambda **kwargs: None,
    )
    processed: list[str] = []

    def process(**kwargs):
        object_name = kwargs["object_name"]
        processed.append(object_name)
        if object_name == "one.jpg":
            raise RuntimeError("first failed")
        return AutolabelSidecar(
            bucket="scale",
            object_name=object_name,
            source_size=1,
            source_fingerprint="etag:x:size:1",
            product_id=None,
            product_name=None,
            state="unmatched",
            timestamp="2026-07-27T12:00:00Z",
            batch_id="batch",
            endpoint_url="http://vlm.test/inference",
            max_tokens=512,
        )

    monkeypatch.setattr(autolabel_worker, "process_image", process)
    result = autolabel_worker.run_autolabel_batch(
        {
            "batch_id": "batch",
            "object_names": ["one.jpg", "two.jpg"],
            "configuration": configuration().model_dump(),
            "catalog": [item.model_dump() for item in catalog()],
        }
    )

    assert processed == ["one.jpg", "two.jpg"]
    assert result["status"] == "completed"
    assert result["failed"] == 1
    assert result["unmatched"] == 1
    assert job.saved >= 5

    bulk_job = FakeJob()
    bulk_job.meta.update(
        items=[], total=2, completed=0, matched=0, unmatched=0, failed=0
    )
    processed.clear()
    monkeypatch.setattr(autolabel_worker, "get_current_job", lambda: bulk_job)

    bulk_result = autolabel_worker.run_autolabel_batch(
        {
            "batch_id": "bulk-batch",
            "object_names": ["one.jpg", "two.jpg"],
            "configuration": configuration().model_dump(),
            "catalog": [item.model_dump() for item in catalog()],
        }
    )

    assert bulk_result["items"] == []
    assert bulk_result["completed"] == 2
    assert bulk_result["failed"] == 1
    assert bulk_result["unmatched"] == 1


def test_finalize_worker_moves_only_matched_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinalizeStorage:
        def __init__(self) -> None:
            self.copied: list[str] = []
            self.deleted: list[str] = []

        def list_objects(self, bucket: str):
            yield S3Object("raw/scale/matched.jpg", 10, etag="matched")
            yield S3Object("raw/scale/unmatched.jpg", 10, etag="unmatched")
            yield S3Object("_autolabel/scale/v1/result.json", 10)

        def head_object(self, bucket: str, object_name: str):
            return S3ObjectMetadata("image/jpeg", {}, 10, object_name)

        def copy_object(self, **kwargs) -> None:
            self.copied.append(kwargs["source_object"])

        def delete_objects(self, bucket: str, object_names: list[str]) -> None:
            self.deleted.extend(object_names)

    storage = FinalizeStorage()
    matched = AutolabelSidecar(
        bucket="scale",
        object_name="raw/scale/matched.jpg",
        source_size=10,
        source_fingerprint="raw/scale/matched.jpg",
        product_id="product-id",
        product_name="Jabłko",
        state="matched",
        timestamp="2026-08-12T10:00:00Z",
        batch_id="batch",
        endpoint_url="manual",
        max_tokens=1,
    )
    monkeypatch.setattr(autolabel_worker, "get_object_storage", lambda: storage)
    monkeypatch.setattr(autolabel_worker.settings, "S3_SCALE_BUCKET", "scale")
    monkeypatch.setattr(autolabel_worker.settings, "S3_EXTERNAL_BUCKET", "external")
    monkeypatch.setattr(
        autolabel_worker,
        "read_sidecar",
        lambda object_name, fingerprint: (
            matched if object_name == "raw/scale/matched.jpg" else None
        ),
    )
    monkeypatch.setattr(autolabel_worker, "get_current_job", lambda: None)

    result = autolabel_worker.run_finalize_all_matched()

    assert result == {"moved": 1}
    assert storage.copied == ["raw/scale/matched.jpg"]
    assert "raw/scale/matched.jpg" in storage.deleted
