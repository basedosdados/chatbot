"""Make query_ref the sole primary key of query_handles.

Revision ID: 91426eac7604
Revises: 4b3d2fa4a75f
Create Date: 2026-08-18 15:02:34.101000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "91426eac7604"
down_revision: Union[str, Sequence[str], None] = "4b3d2fa4a75f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("query_handles_pkey", "query_handles", type_="primary")
    op.create_primary_key("query_handles_pkey", "query_handles", ["query_ref"])
    op.create_index(
        op.f("ix_query_handles_message_id"),
        "query_handles",
        ["message_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_query_handles_message_id"), table_name="query_handles")
    op.drop_constraint("query_handles_pkey", "query_handles", type_="primary")
    op.create_primary_key(
        "query_handles_pkey", "query_handles", ["message_id", "query_ref"]
    )
