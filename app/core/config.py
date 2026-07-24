from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    AnyUrl,
    BeforeValidator,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [item.strip() for item in v.split(",") if item.strip()]
    if isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Self-Checkout ML"
    SECRET_KEY: str = ""
    ENVIRONMENT: str = "local"
    FRONTEND_HOST: str = "http://localhost:5173"
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []
    S3_ENDPOINT_URL: AnyHttpUrl | None = None
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_SESSION_TOKEN: str | None = None
    S3_USE_SSL: bool = False
    S3_FORCE_PATH_STYLE: bool = True
    S3_VERIFY_TLS: bool = True
    S3_CONNECT_TIMEOUT: int = 5
    S3_READ_TIMEOUT: int = 30
    S3_MAX_RETRIES: int = 3
    S3_CREATE_BUCKETS: bool = False
    S3_PUBLIC_BASE_URL: AnyHttpUrl | None = None
    S3_SHELF_BUCKET: str | None = None
    S3_SCALE_BUCKET: str | None = None
    S3_EXTERNAL_BUCKET: str | None = None
    S3_TRAINING_BUCKET: str | None = None
    S3_LABEL_STUDIO_EXPORT_BUCKET: str | None = None
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str = "self-checkout-classifier"
    MLFLOW_REGISTERED_MODEL_NAME: str = "self-checkout-classifier"
    MLFLOW_SHELF_EXPERIMENT_NAME: str = "self-checkout-shelf-classifier"
    MLFLOW_SHELF_MODEL_NAME: str = "self-checkout-shelf-classifier"
    MODEL_CACHE_DIR: str = ".cache/model_store"
    BACKEND_URL: str = "http://127.0.0.1:8000"
    LABEL_STUDIO_URL: str = "http://127.0.0.1:8080"
    LABEL_STUDIO_API_KEY: str = ""
    LABEL_STUDIO_SCALE_PROJECT_TITLE: str = "scale-products"
    LABEL_STUDIO_SHELF_PROJECT_TITLE: str = "shelf-products"
    LABEL_STUDIO_EXTERNAL_PROJECT_TITLE: str = "external-products"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    @model_validator(mode="after")
    def _validate_external_services(self) -> Self:
        is_local = self.ENVIRONMENT == "local"
        defaults = {
            "S3_SHELF_BUCKET": "session-images",
            "S3_SCALE_BUCKET": "scale-images",
            "S3_EXTERNAL_BUCKET": "uploaded-images",
            "S3_TRAINING_BUCKET": "training-data",
            "S3_LABEL_STUDIO_EXPORT_BUCKET": "labelstudio-exports",
        }
        if self.S3_ENDPOINT_URL is None:
            if not is_local:
                raise ValueError(
                    "S3_ENDPOINT_URL is required outside local development"
                )
            self.S3_ENDPOINT_URL = AnyHttpUrl("http://localhost:8082")
        for field, default in defaults.items():
            if getattr(self, field) is None:
                if not is_local:
                    raise ValueError(f"{field} is required outside local development")
                setattr(self, field, default)
        if bool(self.S3_ACCESS_KEY_ID) != bool(self.S3_SECRET_ACCESS_KEY):
            raise ValueError(
                "S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set together"
            )
        if self.S3_SESSION_TOKEN and not self.S3_ACCESS_KEY_ID:
            raise ValueError("S3_SESSION_TOKEN requires S3 access credentials")
        if self.S3_CREATE_BUCKETS and not is_local:
            raise ValueError("S3_CREATE_BUCKETS is allowed only in local development")
        if self.S3_ENDPOINT_URL.scheme == "https" and not self.S3_USE_SSL:
            raise ValueError("S3_USE_SSL must be true for an https S3_ENDPOINT_URL")
        if self.S3_ENDPOINT_URL.scheme == "http" and self.S3_USE_SSL:
            raise ValueError("S3_USE_SSL must be false for an http S3_ENDPOINT_URL")
        if self.MLFLOW_TRACKING_URI is None and is_local:
            self.MLFLOW_TRACKING_URI = "http://127.0.0.1:5002"
        return self


settings = Settings()  # type: ignore
