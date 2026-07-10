import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import AliasPath, JsonValue
from sqlalchemy import Enum as SAEnum
from sqlmodel import JSON, TIMESTAMP, Column, Field, Integer, Relationship, SQLModel

from app.artifacts import FileArtifact


# =============================================================================
# ==                              Thread Models                              ==
# =============================================================================
class ThreadPayload(SQLModel):
    title: str


class ThreadCreate(ThreadPayload):
    user_id: uuid.UUID = Field(index=True)


class Thread(ThreadCreate, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=TIMESTAMP(timezone=True),
        index=True,
    )
    deleted: bool = Field(default=False)

    messages: list["Message"] = Relationship(back_populates="thread")


# ==============================================================================
# ==                              Message Models                              ==
# ==============================================================================
class MessageRole(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class MessageStatus(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    INTERRUPTED = "INTERRUPTED"
    MODEL_CALL_LIMIT = "MODEL_CALL_LIMIT"


class MessageCreate(SQLModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: uuid.UUID = Field(foreign_key="thread.id", index=True)
    user_message_id: uuid.UUID | None = Field(default=None, foreign_key="message.id")
    model_uri: str
    role: MessageRole = Field(
        sa_column=Column(SAEnum(MessageRole), nullable=False),
    )
    content: str
    events: JsonValue | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )
    structured_response: JsonValue | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )
    status: MessageStatus = Field(
        sa_column=Column(SAEnum(MessageStatus), nullable=False),
        default=MessageStatus.SUCCESS,
    )


class Message(MessageCreate, table=True):
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=TIMESTAMP(timezone=True),
        index=True,
    )

    thread: Thread = Relationship(back_populates="messages")
    feedback: "Feedback" = Relationship(back_populates="message")
    artifacts: list["Artifact"] = Relationship(back_populates="message")


class MessagePublic(MessageCreate):
    created_at: datetime
    artifacts: list["ArtifactPublic"] = []


# ==============================================================================
# ==                             Artifact Models                              ==
# ==============================================================================
class Artifact(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    message_id: uuid.UUID = Field(foreign_key="message.id", index=True)
    thread_id: uuid.UUID = Field(foreign_key="thread.id", index=True)
    type: str
    data: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=TIMESTAMP(timezone=True),
        index=True,
    )

    message: Message = Relationship(back_populates="artifacts")

    @classmethod
    def from_tool_artifact(
        cls,
        artifact: dict[str, Any],
        *,
        message_id: str | uuid.UUID,
        thread_id: str | uuid.UUID,
    ) -> "Artifact":
        """Build an Artifact row from a tool's `content_and_artifact` payload.

        The tool emits the transport shape produced by `app.artifacts.FileArtifact`
        (id + `source` + `metadata`). The relational fields become columns; the
        type-specific body (`source` + `metadata`) is stored as-is in `data`.
        """
        # model_validate (not direct construction) so the ISO string / str ids from
        # the tool payload are coerced to datetime / UUID, since table models skip
        # validation on __init__.
        return cls.model_validate(
            {
                "id": artifact["id"],
                "message_id": message_id,
                "thread_id": thread_id,
                "type": artifact["type"],
                "data": {
                    "source": artifact["source"],
                    "metadata": artifact["metadata"],
                },
                "created_at": artifact["created_at"],
            }
        )

    def to_file_artifact(self) -> FileArtifact:
        """Reconstruct the domain FileArtifact from this row."""
        # Assumes type == "file"; when other types exist, dispatch on self.type.
        return FileArtifact.model_validate(
            {
                "id": self.id,
                "type": self.type,
                "created_at": self.created_at,
                **self.data,
            }
        )


class ArtifactPublic(SQLModel):
    # NOTE: these fields are FileArtifact-specific (the only artifact type today).
    # Unlike the polymorphic Artifact table, this response model is intentionally
    # file-shaped for now. When a second type lands, generalize this — e.g. a generic
    # public body or a discriminated union per type (a code/contract change, no DB migration).
    id: uuid.UUID
    type: str
    created_at: datetime
    filename: str | None = Field(
        default=None, validation_alias=AliasPath("data", "metadata", "filename")
    )
    mime_type: str | None = Field(
        default=None, validation_alias=AliasPath("data", "metadata", "mime_type")
    )
    size_bytes: int | None = Field(
        default=None, validation_alias=AliasPath("data", "metadata", "size_bytes")
    )


# ===============================================================================
# ==                              Feedback Models                              ==
# ===============================================================================
class FeedbackRating(int, Enum):
    POSITIVE = 1
    NEGATIVE = 0


class FeedbackSyncStatus(str, Enum):
    FAILED = "FAILED"
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"


class FeedbackPayload(SQLModel):
    rating: FeedbackRating = Field(sa_column=Column(Integer, nullable=False))
    comments: str | None = Field(default=None)


class FeedbackCreate(FeedbackPayload):
    message_id: uuid.UUID = Field(foreign_key="message.id", unique=True, index=True)


class FeedbackPublic(FeedbackCreate):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class Feedback(FeedbackCreate, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=TIMESTAMP(timezone=True),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_type=TIMESTAMP(timezone=True),
    )
    sync_status: FeedbackSyncStatus = Field(
        sa_column=Column(
            SAEnum(FeedbackSyncStatus),
            nullable=False,
        ),
        default=FeedbackSyncStatus.PENDING,
    )
    synced_at: datetime | None = Field(
        default=None,
        sa_type=TIMESTAMP(timezone=True),
    )

    message: Message = Relationship(back_populates="feedback")
