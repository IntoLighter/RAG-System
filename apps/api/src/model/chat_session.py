from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class ChatSession(SQLModel, table=True):
    """Listing metadata of a conversation; messages live in langgraph checkpoints."""

    __tablename__ = "chat_session"

    id: UUID = Field(primary_key=True)
    title: str | None = None
    created_at: datetime = Field(default_factory=_now, sa_type=DateTime(timezone=True))
    updated_at: datetime = Field(default_factory=_now, sa_type=DateTime(timezone=True))
