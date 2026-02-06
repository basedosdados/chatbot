"""Make feedback ratings non-nullable.

Revision ID: f593c35b86e7
Revises: c0135b4524b9
Create Date: 2026-02-05 17:05:51.285336
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f593c35b86e7"
down_revision: Union[str, Sequence[str], None] = "c0135b4524b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("feedback", "rating", existing_type=sa.INTEGER(), nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("feedback", "rating", existing_type=sa.INTEGER(), nullable=True)
