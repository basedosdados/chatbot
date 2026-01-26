"""Initial migration.

Revision ID: 091db36ec6e3
Revises:
Create Date: 2026-01-26 10:52:41.249851
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.settings import settings

# revision identifiers, used by Alembic.
revision: str = "091db36ec6e3"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create thread table
    op.create_table(
        "thread",
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.create_index(
        op.f("ix_chatbot_thread_created_at"),
        "thread",
        ["created_at"],
        unique=False,
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.create_index(
        op.f("ix_chatbot_thread_user_id"),
        "thread",
        ["user_id"],
        unique=False,
        schema=settings.DB_SCHEMA_CHATBOT,
    )

    # Create message table
    op.create_table(
        "message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("user_message_id", sa.Uuid(), nullable=True),
        sa.Column("model_uri", sa.String(), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "ASSISTANT",
                "USER",
                name="messagerole",
                schema=settings.DB_SCHEMA_CHATBOT,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("artifacts", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("events", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "ERROR",
                "SUCCESS",
                name="messagestatus",
                schema=settings.DB_SCHEMA_CHATBOT,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            [f"{settings.DB_SCHEMA_CHATBOT}.thread.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_message_id"],
            [f"{settings.DB_SCHEMA_CHATBOT}.message.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.create_index(
        op.f("ix_chatbot_message_created_at"),
        "message",
        ["created_at"],
        unique=False,
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.create_index(
        op.f("ix_chatbot_message_thread_id"),
        "message",
        ["thread_id"],
        unique=False,
        schema=settings.DB_SCHEMA_CHATBOT,
    )

    # Create feedback table
    op.create_table(
        "feedback",
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("comments", sa.String(), nullable=True),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column(
            "sync_status",
            sa.Enum(
                "FAILED",
                "PENDING",
                "SUCCESS",
                name="feedbacksyncstatus",
                schema=settings.DB_SCHEMA_CHATBOT,
            ),
            nullable=False,
        ),
        sa.Column("synced_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"],
            [f"{settings.DB_SCHEMA_CHATBOT}.message.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.create_index(
        op.f("ix_chatbot_feedback_message_id"),
        "feedback",
        ["message_id"],
        unique=True,
        schema=settings.DB_SCHEMA_CHATBOT,
    )


def downgrade() -> None:
    # Drop feedback table
    op.drop_index(
        op.f("ix_chatbot_feedback_message_id"),
        table_name="feedback",
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.drop_table("feedback", schema=settings.DB_SCHEMA_CHATBOT)

    # Drop message table
    op.drop_index(
        op.f("ix_chatbot_message_thread_id"),
        table_name="message",
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.drop_index(
        op.f("ix_chatbot_message_created_at"),
        table_name="message",
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.drop_table("message", schema=settings.DB_SCHEMA_CHATBOT)

    # Drop thread table
    op.drop_index(
        op.f("ix_chatbot_thread_user_id"),
        table_name="thread",
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.drop_index(
        op.f("ix_chatbot_thread_created_at"),
        table_name="thread",
        schema=settings.DB_SCHEMA_CHATBOT,
    )
    op.drop_table("thread", schema=settings.DB_SCHEMA_CHATBOT)
