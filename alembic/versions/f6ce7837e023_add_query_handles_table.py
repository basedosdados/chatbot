"""Add query handles table

Revision ID: f6ce7837e023
Revises: 21d5a7602704
Create Date: 2026-07-14 11:02:57.309209
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6ce7837e023"
down_revision: Union[str, Sequence[str], None] = "21d5a7602704"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "query_handles",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("query_ref", sa.String(), nullable=False),
        sa.Column("destination_table", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
        ),
        sa.PrimaryKeyConstraint("message_id", "query_ref"),
    )
    op.create_index(
        op.f("ix_query_handles_created_at"),
        "query_handles",
        ["created_at"],
        unique=False,
    )
    # Safe to drop outright only because the artifacts feature never shipped to
    # production, so there is no message.artifacts data to preserve. Otherwise
    # this migration would first move existing artifacts from the JSON column.
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
    op.drop_index(op.f("ix_query_handles_created_at"), table_name="query_handles")
    op.drop_table("query_handles")
