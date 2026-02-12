"""Initial migration.

Revision ID: 091db36ec6e3
Revises:
Create Date: 2026-01-26 10:52:41.249851
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "091db36ec6e3"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Create thread table
    op.create_table(
        "thread",
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_thread_created_at"), "thread", ["created_at"], unique=False
    )
    op.create_index(op.f("ix_thread_user_id"), "thread", ["user_id"], unique=False)

    # Create message table
    op.create_table(
        "message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("user_message_id", sa.Uuid(), nullable=True),
        sa.Column("model_uri", sa.String(), nullable=False),
        sa.Column(
            "role", sa.Enum("ASSISTANT", "USER", name="messagerole"), nullable=False
        ),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("artifacts", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("events", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "status", sa.Enum("ERROR", "SUCCESS", name="messagestatus"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["thread.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_message_id"],
            ["message.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_message_created_at"), "message", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_message_thread_id"), "message", ["thread_id"], unique=False
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
            sa.Enum("FAILED", "PENDING", "SUCCESS", name="feedbacksyncstatus"),
            nullable=False,
        ),
        sa.Column("synced_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_feedback_message_id"), "feedback", ["message_id"], unique=True
    )


def downgrade() -> None:
    """Downgrade schema."""

    # Drop feedback table
    op.drop_index(op.f("ix_feedback_message_id"), table_name="feedback")
    op.drop_table("feedback")

    # Drop message table
    op.drop_index(op.f("ix_message_thread_id"), table_name="message")
    op.drop_index(op.f("ix_message_created_at"), table_name="message")
    op.drop_table("message")

    # Drop thread table
    op.drop_index(op.f("ix_thread_user_id"), table_name="thread")
    op.drop_index(op.f("ix_thread_created_at"), table_name="thread")
    op.drop_table("thread")
