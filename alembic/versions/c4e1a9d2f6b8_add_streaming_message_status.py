"""Add STREAMING message status.

Revision ID: c4e1a9d2f6b8
Revises: 91426eac7604
Create Date: 2026-08-18 15:02:34.101000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e1a9d2f6b8"
down_revision: Union[str, Sequence[str], None] = "91426eac7604"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add STREAMING to the messagestatus enum.

    Note: `ALTER TYPE ... ADD VALUE` cannot run inside a transaction block
    on PostgreSQL pre-v12, so we use an autocommit block.
    """
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE messagestatus ADD VALUE IF NOT EXISTS 'STREAMING'")


def downgrade() -> None:
    """Downgrade is intentionally unsupported.

    Postgres cannot drop enum values, and rebuilding the type would
    require remapping any STREAMING rows to another status.
    """
    raise NotImplementedError(
        "Downgrade not supported: removing enum values would silently rewrite rows."
    )
