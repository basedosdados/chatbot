"""Add structured_response column to messages table.

Revision ID: 21d5a7602704
Revises: 19e2c92563e2
Create Date: 2026-06-22 13:20:35.583925
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "21d5a7602704"
down_revision: Union[str, Sequence[str], None] = "19e2c92563e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "message",
        sa.Column("structured_response", sa.JSON(none_as_null=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("message", "structured_response")
