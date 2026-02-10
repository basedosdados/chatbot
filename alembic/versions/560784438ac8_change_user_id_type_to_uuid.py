"""Change user_id type to UUID.

Revision ID: 560784438ac8
Revises: f593c35b86e7
Create Date: 2026-02-10 10:27:20.545378
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "560784438ac8"
down_revision: Union[str, Sequence[str], None] = "f593c35b86e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "thread",
        "user_id",
        existing_type=sa.INTEGER(),
        type_=sa.Uuid(),
        existing_nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "thread",
        "user_id",
        existing_type=sa.Uuid(),
        type_=sa.INTEGER(),
        existing_nullable=False,
    )
