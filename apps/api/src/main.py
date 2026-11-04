from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dependency.agent import init_agent
from dependency.db import close_db_pool, init_db_pool
from dependency.logging import configure_logging
from dependency.settings import settings
from dependency.vector_store import init_vector_store
from middleware.logging import LoggingMiddleware
from routes import document, query, session

configure_logging(
    level=settings.app.log_level,
    json_logs=settings.app.log_json,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    await init_db_pool()
    await init_vector_store()
    await init_agent()
    try:
        yield
    finally:
        await close_db_pool()


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)
app.include_router(query.router)
app.include_router(document.router)
app.include_router(session.router)
