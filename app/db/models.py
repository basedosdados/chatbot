import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import JsonValue, computed_field
from sqlalchemy import Enum as SAEnum
from sqlmodel import JSON, TIMESTAMP, Column, Field, Integer, Relationship, SQLModel


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
    query_handles: list["QueryHandle"] = Relationship(back_populates="message")


class MessagePublic(MessageCreate):
    created_at: datetime
    # Eager-loaded to derive `downloads`; excluded from serialization because each
    # handle carries the internal `destination_table`, which must not reach the client.
    query_handles: list["QueryHandle"] = Field(default_factory=list, exclude=True)

    @computed_field
    @property
    def downloads(self) -> list[dict[str, Any]]:
        """The downloads offered for this message."""
        from app.exports import query_result_download

        return [
            query_result_download(handle.query_ref, handle.slug)
            for handle in self.query_handles
        ]


# ==============================================================================
# ==                           Query Handle Models                            ==
# ==============================================================================
class QueryHandle(SQLModel, table=True):
    __tablename__ = "query_handles"

    # Field order defines the composite PK column order (message_id, query_ref) — keep
    # message_id first to match the migration; reordering these fields changes the PK.
    message_id: uuid.UUID = Field(foreign_key="message.id", primary_key=True)

    # The `str` column leaves room to shorten it to a model-reproducible
    # token should exports ever key off the answer again.
    query_ref: str = Field(primary_key=True)

    # The model-generated slug for the query.
    slug: str
    destination_table: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=TIMESTAMP(timezone=True),
        index=True,
    )

    message: Message = Relationship(back_populates="query_handles")


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
