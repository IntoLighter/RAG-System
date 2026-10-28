"""
Create langchain-postgres tables.

Mirrors the schema that langchain-postgres builds itself via
``PGEngine.ainit_vectorstore_table``, ``PGVectorStore.aapply_vector_index``
with ``HNSWIndex()`` and ``PostgresChatMessageHistory.acreate_tables``.

Revision ID: 0001
Revises:
Create Date: 2026-10-26 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A revision is a snapshot of the schema, so these values are fixed here
# instead of being read from settings. Table names must match
# POSTGRES__EMBEDDING_TABLE_NAME and POSTGRES__CHAT_TABLE_NAME, and the
# vector size must match the embedding model; changing any of them needs
# a new revision. The index suffix is DEFAULT_INDEX_NAME_SUFFIX of
# langchain-postgres 0.0.19.
embedding_table = "embedding"
chat_table = "chat"
vector_size = 1024
index_suffix = "langchainvectorindex"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        embedding_table,
        sa.Column("langchain_id", sa.Uuid(), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "embedding",
            Vector(vector_size),
            nullable=False,
        ),
        sa.Column("langchain_metadata", sa.JSON()),
    )
    # PGVectorStore looks the index up by this name in ``ais_valid_index``.
    op.create_index(
        embedding_table + index_suffix,
        embedding_table,
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        chat_table,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("message", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(f"idx_{chat_table}_session_id", chat_table, ["session_id"])


def downgrade() -> None:
    op.drop_table(chat_table)
    op.drop_table(embedding_table)
