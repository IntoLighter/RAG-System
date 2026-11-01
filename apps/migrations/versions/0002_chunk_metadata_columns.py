"""
Add document metadata columns to chunks.

PGVectorStore stores these metadata keys in their own columns instead of
the JSON column, so chunks can be filtered and deleted by ``document_id``.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

embedding_table = "embedding"


def upgrade() -> None:
    # Nullable, since chunks indexed before this revision have no metadata
    op.add_column(embedding_table, sa.Column("document_id", sa.Text()))
    op.add_column(embedding_table, sa.Column("filename", sa.Text()))
    op.add_column(embedding_table, sa.Column("page", sa.Integer()))
    op.create_index(
        f"idx_{embedding_table}_document_id",
        embedding_table,
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index(f"idx_{embedding_table}_document_id", embedding_table)
    op.drop_column(embedding_table, "page")
    op.drop_column(embedding_table, "filename")
    op.drop_column(embedding_table, "document_id")
