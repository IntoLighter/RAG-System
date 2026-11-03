"""
Replace the chat history table with langgraph checkpoint tables.

Mirrors the schema that ``AsyncPostgresSaver.setup`` of
langgraph-checkpoint-postgres 3.2.0 builds after applying all of its
``MIGRATIONS``, and records them in ``checkpoint_migrations``, so the API
can check on startup that the schema matches the installed version.
The thread_id indexes are created without CONCURRENTLY, since the tables
are empty and alembic runs inside a transaction.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

chat_table = "chat"
applied_checkpoint_migrations = 10


def upgrade() -> None:
    migrations = op.create_table(
        "checkpoint_migrations",
        sa.Column("v", sa.Integer(), primary_key=True, autoincrement=False),
    )

    op.create_table(
        "checkpoints",
        sa.Column("thread_id", sa.Text(), primary_key=True),
        sa.Column(
            "checkpoint_ns",
            sa.Text(),
            primary_key=True,
            server_default="",
        ),
        sa.Column("checkpoint_id", sa.Text(), primary_key=True),
        sa.Column("parent_checkpoint_id", sa.Text()),
        sa.Column("type", sa.Text()),
        sa.Column("checkpoint", postgresql.JSONB(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )

    op.create_table(
        "checkpoint_blobs",
        sa.Column("thread_id", sa.Text(), primary_key=True),
        sa.Column(
            "checkpoint_ns",
            sa.Text(),
            primary_key=True,
            server_default="",
        ),
        sa.Column("channel", sa.Text(), primary_key=True),
        sa.Column("version", sa.Text(), primary_key=True),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("blob", postgresql.BYTEA()),
    )

    op.create_table(
        "checkpoint_writes",
        sa.Column("thread_id", sa.Text(), primary_key=True),
        sa.Column(
            "checkpoint_ns",
            sa.Text(),
            primary_key=True,
            server_default="",
        ),
        sa.Column("checkpoint_id", sa.Text(), primary_key=True),
        sa.Column("task_id", sa.Text(), primary_key=True),
        sa.Column("idx", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("type", sa.Text()),
        sa.Column("blob", postgresql.BYTEA(), nullable=False),
        sa.Column("task_path", sa.Text(), nullable=False, server_default=""),
    )

    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        op.create_index(f"{table}_thread_id_idx", table, ["thread_id"])

    op.bulk_insert(
        migrations,
        [{"v": v} for v in range(applied_checkpoint_migrations)],
    )

    op.drop_table(chat_table)


def downgrade() -> None:
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

    op.drop_table("checkpoint_writes")
    op.drop_table("checkpoint_blobs")
    op.drop_table("checkpoints")
    op.drop_table("checkpoint_migrations")
