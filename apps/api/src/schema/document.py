from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str = Field(description="SHA-256 of the uploaded file")
    filename: str | None
    chunks: int
