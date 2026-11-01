import asyncio
import bisect
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Annotated

import pymupdf
import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel import col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession
from tokenizers import Tokenizer

from dependency.db import engine, get_session
from dependency.settings import settings
from dependency.vector_store import get_vector_store
from model.chunk import Chunk
from schema.document import DocumentInfo
from utils import elapsed_ms

logger = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])

tokenizer = Tokenizer.from_pretrained(settings.embedding.name)


def _token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.rag.chunk_size,
    chunk_overlap=settings.rag.chunk_overlap,
    separators=settings.rag.separators,
    keep_separator="end",
    length_function=_token_length,
)


def _extract_pages(content: bytes) -> list[str]:
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return [page.get_text() for page in doc]


@dataclass
class PageChunk:
    text: str
    page: int


def _split_pages(pages: list[str]) -> list[PageChunk]:
    """Split the whole book into chunks, each with the page it starts on."""
    cleaned_pages = [pre_clean(page) for page in pages]

    """
    The page of a chunk is found by its start offset in the joined text.
    The splitter's add_start_index can't be used for that: it looks for
    each next chunk from (previous start + previous length in chars -
    chunk_overlap), but with a token length_function chunk_overlap is in
    tokens, so the search starts after the real start and returns -1.
    Chunks come in text order, so searching from the previous chunk's
    start + 1 always finds the right occurrence, even of repeated text.
    """
    page_starts = []
    offset = 0
    for page in cleaned_pages:
        page_starts.append(offset)
        offset += len(page) + len(settings.rag.page_separator)

    full_text = settings.rag.page_separator.join(cleaned_pages)

    splits = []
    start = -1
    for chunk in text_splitter.split_text(full_text):
        start = full_text.find(chunk, start + 1)
        page = bisect.bisect_right(page_starts, start)
        splits.append(PageChunk(text=post_clean(chunk), page=page))

    return splits


def pre_clean(text: str) -> str:
    text = re.sub(r"[ \t\xa0\f\v]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?…:;\n])\n(?![\n#*\-»«\"'(\[{])", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def post_clean(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)


async def _document_exists(document_id: str) -> bool:
    statement = select(Chunk.langchain_id).where(Chunk.document_id == document_id)
    async with AsyncSession(engine) as session:
        result = await session.exec(statement.limit(1))
        return result.first() is not None


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    vector_store: Annotated[PGVectorStore, Depends(get_vector_store)],
) -> DocumentInfo:
    """To upload russian or english books."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type")

    started_at = time.perf_counter()
    content = await file.read()
    document_id = hashlib.sha256(content).hexdigest()
    structlog.contextvars.bind_contextvars(document_id=document_id)

    if await _document_exists(document_id):
        logger.warning("document_already_indexed", filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is already uploaded, document_id: {document_id}",
        )

    pages = await asyncio.to_thread(_extract_pages, content)
    extract_elapsed_ms = elapsed_ms(started_at)

    if not any(page.strip() for page in pages):
        logger.warning(
            "pdf_has_no_text_layer",
            filename=file.filename,
            pages=len(pages),
            elapsed_ms=extract_elapsed_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF has no text layer (probably a scanned document)",
        )

    started_at = time.perf_counter()
    splits = await asyncio.to_thread(_split_pages, pages)
    split_elapsed_ms = elapsed_ms(started_at)

    started_at = time.perf_counter()
    await vector_store.aadd_texts(
        [split.text for split in splits],
        metadatas=[
            {
                "document_id": document_id,
                "filename": file.filename,
                "page": split.page,
            }
            for split in splits
        ],
    )
    embed_elapsed_ms = elapsed_ms(started_at)

    logger.info(
        "document_indexed",
        filename=file.filename,
        size_bytes=len(content),
        pages=len(pages),
        chunks=len(splits),
        extract_elapsed_ms=extract_elapsed_ms,
        split_elapsed_ms=split_elapsed_ms,
        embed_elapsed_ms=embed_elapsed_ms,
    )

    return DocumentInfo(
        document_id=document_id,
        filename=file.filename,
        chunks=len(splits),
    )


@router.get("")
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[DocumentInfo]:
    statement = (
        select(Chunk.document_id, Chunk.filename, func.count())
        .where(col(Chunk.document_id).is_not(None))
        .group_by(col(Chunk.document_id), col(Chunk.filename))
        .order_by(col(Chunk.filename))
    )
    result = await session.exec(statement)
    return [
        DocumentInfo(document_id=document_id, filename=filename, chunks=chunks)
        for document_id, filename, chunks in result.all()
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    vector_store: Annotated[PGVectorStore, Depends(get_vector_store)],
) -> Response:
    if not await _document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    await vector_store.adelete(filter={"document_id": document_id})
    logger.info("document_deleted", document_id=document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
