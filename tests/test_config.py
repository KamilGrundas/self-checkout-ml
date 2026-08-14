import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_production_rejects_missing_s3_endpoint() -> None:
    with pytest.raises(ValidationError, match="S3_ENDPOINT_URL is required"):
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            S3_ENDPOINT_URL=None,
            S3_CREATE_BUCKETS=False,
        )


def test_production_accepts_external_s3() -> None:
    settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        S3_ENDPOINT_URL="https://objects.example.invalid",
        S3_USE_SSL=True,
        S3_SHELF_BUCKET="shelf",
        S3_SCALE_BUCKET="scale",
        S3_EXTERNAL_BUCKET="uploads",
        S3_TRAINING_BUCKET="training",
        S3_CREATE_BUCKETS=False,
        TRAINING_QUEUE_URL="redis://queue.example.invalid:6379/0",
    )

    assert settings.S3_CREATE_BUCKETS is False
