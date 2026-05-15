"""Add MODEL_CALL_LIMIT message status.

Revision ID: 19e2c92563e2
Revises: 1c6556bb74f2
Create Date: 2026-05-14 16:02:45.217643
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "19e2c92563e2"
down_revision: Union[str, Sequence[str], None] = "1c6556bb74f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add MODEL_CALL_LIMIT to the messagestatus enum.

    Note: `ALTER TYPE ... ADD VALUE` cannot run inside a transaction block
    on PostgreSQL pre-v12, so we use an autocommit block.
    """
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE messagestatus ADD VALUE IF NOT EXISTS 'MODEL_CALL_LIMIT'"
        )


def downgrade() -> None:
    """Downgrade is intentionally unsupported.

    Postgres cannot drop enum values, and rebuilding the type would require
    remapping existing MODEL_CALL_LIMIT rows to another status — losing the
    diagnostic signal this migration was added to capture.
    """
    raise NotImplementedError(
        "Downgrade not supported: removing MODEL_CALL_LIMIT would silently rewrite rows."
    )
