from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dependency.db import close_db_pool, init_db_pool
from dependency.logging import configure_logging
from dependency.settings import settings
from dependency.vector_store import pg_engine
from middleware.logging import LoggingMiddleware
from routes import document, query

configure_logging(
    level=settings.app.log_level,
    json_logs=settings.app.log_json,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_db_pool()
    try:
        yield
    finally:
        await close_db_pool()
        await pg_engine.close()


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)
app.include_router(query.router)
app.include_router(document.router)
