from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    message: str = Field(examples=["Что такое квест?"])
    session_id: UUID | None = Field(
        default=None,
        description="Continue an existing conversation; omit to start a new one.",
        examples=[None],
    )


class RetrievedContext(BaseModel):
    query: str
    chunks: list[str]


class QueryResponse(BaseModel):
    session_id: UUID
    answer: str
    retrieved: list[RetrievedContext]
