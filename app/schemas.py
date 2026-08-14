from pydantic import BaseModel, Field


class SessionSnapshotPublic(BaseModel):
    session_id: str
    capture_index: int
    product_id: str | None
    product_name: str | None
    filename: str
    object_name: str
    image_url: str
    content_type: str
    size: int


class SessionSnapshotListPublic(BaseModel):
    data: list[SessionSnapshotPublic]


class PredictionPublic(BaseModel):
    scores: dict[str, float]
    model_id: str


class ModelVersionPublic(BaseModel):
    name: str
    version: int
    model_id: str
    status: str
    description: str | None = None
    created_at: str | None = None
    is_active: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)


class SetModelRequest(BaseModel):
    version: int


class ModelActivatedPublic(BaseModel):
    model_name: str
    model_version: int
    model_id: str
    cache_key: str


class StoredImagePublic(BaseModel):
    bucket_name: str
    object_name: str
    image_url: str
    content_type: str
    size: int
