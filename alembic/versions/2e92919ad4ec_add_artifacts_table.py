"""Add artifacts table

Revision ID: 2e92919ad4ec
Revises: 21d5a7602704
Create Date: 2026-07-10 10:29:24.466540
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2e92919ad4ec"
down_revision: Union[str, Sequence[str], None] = "21d5a7602704"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "artifact",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["thread.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_artifact_created_at"), "artifact", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_artifact_message_id"), "artifact", ["message_id"], unique=False
    )
    op.create_index(
        op.f("ix_artifact_thread_id"), "artifact", ["thread_id"], unique=False
    )
    # Safe to drop outright only because the artifacts feature never shipped to
    # production, so there is no message.artifacts data to preserve. Otherwise
    # this migration would first move existing artifacts from the JSON column
    # into the new `artifact` table.
    op.drop_column("message", "artifacts")


def downgrade() -> None:
    """Downgrade schema."""
    # Re-created empty: since nothing was moved out of the column on upgrade (no
    # production data), nothing is moved back. A downgrade carrying real data would
    # first repopulate message.artifacts from the `artifact` table before dropping it.
    op.add_column(
        "message",
        sa.Column("artifacts", sa.JSON(none_as_null=True), nullable=True),
    )
    op.drop_index(op.f("ix_artifact_thread_id"), table_name="artifact")
    op.drop_index(op.f("ix_artifact_message_id"), table_name="artifact")
    op.drop_index(op.f("ix_artifact_created_at"), table_name="artifact")
    op.drop_table("artifact")
