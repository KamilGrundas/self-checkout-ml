from typing import Annotated, Any

from pydantic import AnyUrl, BeforeValidator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ENVIRONMENT: str = "local"
    FRONTEND_HOST: str = "http://localhost:5173"
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_PUBLIC_URL: str = "http://localhost:9000"
    MINIO_USE_SSL: bool = False
    ML_MINIO_SHELF_BUCKET_NAME: str = "session-images"
    ML_MINIO_SCALE_BUCKET_NAME: str = "scale-images"
    ML_MINIO_EXTERNAL_BUCKET_NAME: str = "uploaded-images"
    ML_MINIO_TRAINING_BUCKET_NAME: str = "training-data"
    ML_MINIO_LABELSTUDIO_EXPORT_BUCKET_NAME: str = "labelstudio-exports"
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5002"
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


settings = Settings()  # type: ignore
