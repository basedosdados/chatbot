"""Make query_ref the sole primary key of query_handles

Revision ID: 9a3c7b2e1f04
Revises: 4b3d2fa4a75f
Create Date: 2026-08-14 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9a3c7b2e1f04"
down_revision: Union[str, Sequence[str], None] = "4b3d2fa4a75f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # query_ref is a globally-unique `qr_<uuid4hex>`, so it stands alone as the PK.
    # Safe to repoint without deduping: no two handles can share a query_ref. The
    # composite (message_id, query_ref) key is retired now that query_ref is never
    # shortened to a model-reproducible (non-unique) token.
    op.drop_constraint("query_handles_pkey", "query_handles", type_="primary")
    op.create_primary_key("query_handles_pkey", "query_handles", ["query_ref"])
    # message_id is no longer the leading PK column, so index it for the FK join and
    # the per-message download lookup.
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
