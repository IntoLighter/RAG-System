"""
Add the chat session table for listing conversations.

Messages stay in the langgraph checkpoint tables; this table keeps what
a list needs without loading every checkpoint. Existing threads are
backfilled with the time of their first and last checkpoint and no title.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-14 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

session_table = "chat_session"


def upgrade() -> None:
    op.create_table(
        session_table,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("title", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        f"idx_{session_table}_updated_at",
        session_table,
        [sa.text("updated_at DESC")],
    )

    op.execute(
        """
        INSERT INTO chat_session (id, created_at, updated_at)
        SELECT
            thread_id::uuid,
            min((checkpoint->>'ts')::timestamptz),
            max((checkpoint->>'ts')::timestamptz)
        FROM checkpoints
        WHERE checkpoint_ns = ''
        GROUP BY thread_id
        """,
    )


def downgrade() -> None:
    op.drop_index(f"idx_{session_table}_updated_at", session_table)
    op.drop_table(session_table)
