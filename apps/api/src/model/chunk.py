from uuid import UUID

from sqlmodel import Field, SQLModel

from dependency.settings import settings


class Chunk(SQLModel, table=True):
    """
    Read-only view of the PGVectorStore table.

    The table is created by migrations and written by PGVectorStore,
    so the embedding and JSON metadata columns are left out.
    """

    __tablename__ = settings.postgres.embedding_table_name

    langchain_id: UUID = Field(primary_key=True)
    content: str
    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
