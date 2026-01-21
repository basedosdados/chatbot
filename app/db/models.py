import uuid
from datetime import datetime
from enum import Enum

from pydantic import JsonValue
from sqlalchemy import Enum as SAEnum
from sqlmodel import JSON, Column, Field, Integer, Relationship, SQLModel

from app.settings import settings


# Stub table for the Django-managed 'account' table in the website schema.
class Account(SQLModel, table=True):
    __tablename__ = "account"
    __table_args__ = {"schema": settings.PG_SCHEMA_WEBSITE}

    id: int = Field(primary_key=True)


# =============================================================================
# ==                              Thread Models                              ==
# =============================================================================
class ThreadPayload(SQLModel):
    title: str


class ThreadCreate(ThreadPayload):
    user_id: int = Field(
        foreign_key=f"{settings.PG_SCHEMA_WEBSITE}.account.id", index=True
    )


class Thread(ThreadCreate, table=True):
    __table_args__ = {"schema": settings.PG_SCHEMA_CHATBOT}

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now, index=True)
    deleted: bool = Field(default=False)

    messages: list["Message"] = Relationship(back_populates="thread")


# ==============================================================================
# ==                              Message Models                              ==
# ==============================================================================
class MessageRole(str, Enum):
    ASSISTANT = "ASSISTANT"
    USER = "USER"


class MessageStatus(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class MessageCreate(SQLModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: uuid.UUID = Field(foreign_key="chatbot.thread.id", index=True)
    user_message_id: uuid.UUID | None = Field(
        default=None, foreign_key="chatbot.message.id"
    )
    model_uri: str
    role: MessageRole = Field(
        sa_column=Column(
            SAEnum(MessageRole, schema=settings.PG_SCHEMA_CHATBOT), nullable=False
        ),
    )
    content: str
    artifacts: JsonValue | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )
    events: JsonValue | None = Field(
        default=None, sa_column=Column(JSON(none_as_null=True))
    )
    status: MessageStatus = Field(
        sa_column=Column(
            SAEnum(MessageStatus, schema=settings.PG_SCHEMA_CHATBOT), nullable=False
        ),
        default=MessageStatus.SUCCESS,
    )


class Message(MessageCreate, table=True):
    __table_args__ = {"schema": settings.PG_SCHEMA_CHATBOT}

    created_at: datetime = Field(default_factory=datetime.now, index=True)

    thread: Thread = Relationship(back_populates="messages")
    feedback: "Feedback" = Relationship(back_populates="message")


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
    rating: FeedbackRating = Field(sa_column=Column(Integer))
    comments: str | None = Field(default=None)


class FeedbackCreate(FeedbackPayload):
    message_id: uuid.UUID = Field(
        foreign_key="chatbot.message.id", unique=True, index=True
    )


class FeedbackPublic(FeedbackCreate):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class Feedback(FeedbackCreate, table=True):
    __table_args__ = {"schema": settings.PG_SCHEMA_CHATBOT}

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = Field(default=None)
    sync_status: FeedbackSyncStatus = Field(
        sa_column=Column(
            SAEnum(FeedbackSyncStatus, schema=settings.PG_SCHEMA_CHATBOT),
            nullable=False,
        ),
        default=FeedbackSyncStatus.PENDING,
    )
    synced_at: datetime | None = Field(default=None)

    message: Message = Relationship(back_populates="feedback")
