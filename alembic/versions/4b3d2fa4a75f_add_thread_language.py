"""Add language column to threads table.

Revision ID: 4b3d2fa4a75f
Revises: f6ce7837e023
Create Date: 2026-08-04 19:50:02.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b3d2fa4a75f"
down_revision: Union[str, Sequence[str], None] = "f6ce7837e023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default backfills existing threads with the Portuguese default;
    # new rows get their value from the application (ThreadPayload.language).
    op.add_column(
        "thread",
        sa.Column("language", sa.String(), nullable=False, server_default="pt"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("thread", "language")
