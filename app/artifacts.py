import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class RemoteObjectSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["remote_object"] = "remote_object"
    provider: Literal["gcs"] = "gcs"
    bucket: str
    object_key: str


ArtifactSource = Annotated[Union[RemoteObjectSource], Field(discriminator="type")]


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: Literal["file"] = "file"
    source: ArtifactSource
    metadata: ArtifactMetadata = Field(default_factory=ArtifactMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
