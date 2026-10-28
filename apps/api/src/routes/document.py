import asyncio
import re
import time

import pymupdf
import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from dependency.settings import settings
from dependency.vector_store import vector_store
from utils import elapsed_ms

logger = structlog.get_logger()

router = APIRouter(prefix="/documents", tags=["documents"])

tokenizer = AutoTokenizer.from_pretrained(settings.embedding.name)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=settings.rag.chunk_size,
    chunk_overlap=settings.rag.chunk_overlap,
    separators=settings.rag.separators,
)


def _extract_text(content: bytes) -> tuple[str, int]:
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc), doc.page_count


def _split_text(full_text: str) -> list[str]:
    cleaned = pre_clean(full_text)
    return [post_clean(chunk) for chunk in text_splitter.split_text(cleaned)]


def pre_clean(text: str) -> str:
    text = re.sub(r"[ \t\xa0\f\v]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?…:;\n])\n(?![\n#*\-»«\"'(\[{])", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def post_clean(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile,
) -> JSONResponse:
    """To upload russian or english books."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type")

    started_at = time.perf_counter()
    content = await file.read()

    full_text, pages = await asyncio.to_thread(_extract_text, content)
    extract_elapsed_ms = elapsed_ms(started_at)

    if not full_text.strip():
        logger.warning(
            "pdf_has_no_text_layer",
            filename=file.filename,
            pages=pages,
            elapsed_ms=extract_elapsed_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF has no text layer (probably a scanned document)",
        )

    started_at = time.perf_counter()
    splits = await asyncio.to_thread(_split_text, full_text)
    split_elapsed_ms = elapsed_ms(started_at)

    started_at = time.perf_counter()
    await vector_store.aadd_texts(splits)
    embed_elapsed_ms = elapsed_ms(started_at)

    logger.info(
        "document_indexed",
        filename=file.filename,
        pages=pages,
        chars=len(full_text),
        chunks=len(splits),
        extract_elapsed_ms=extract_elapsed_ms,
        split_elapsed_ms=split_elapsed_ms,
        embed_elapsed_ms=embed_elapsed_ms,
    )

    return JSONResponse(
        content={"status": "success", "message": "Document successfully uploaded"},
    )
