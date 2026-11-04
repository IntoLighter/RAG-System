from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionHistory(BaseModel):
    session_id: UUID
    messages: list[SessionMessage]
