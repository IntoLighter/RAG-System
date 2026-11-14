from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageRequest(BaseModel):
    message: str = Field(examples=["Что такое квест?"])


class RetrievedChunk(BaseModel):
    content: str
    filename: str | None
    page: int | None
    score: float


class RetrievedContext(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    chunks: list[RetrievedChunk]


class MessageResponse(BaseModel):
    session_id: UUID
    answer: str
    retrieved: list[RetrievedContext]


class SessionInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(validation_alias="id")
    title: str | None = Field(description="First question, unless renamed")
    created_at: datetime
    updated_at: datetime


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    retrieved: list[RetrievedContext] | None = Field(
        default=None,
        description="Passages the answer was based on, with include_sources",
    )


class SessionHistory(BaseModel):
    session_id: UUID
    messages: list[SessionMessage]
