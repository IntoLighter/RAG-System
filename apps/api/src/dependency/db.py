from collections.abc import AsyncIterator

import psycopg_pool
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from dependency.settings import settings

engine = create_async_engine(
    "postgresql+psycopg://",
    connect_args=conninfo_to_dict(settings.postgres.conninfo),
    pool_pre_ping=True,
)

# PostgresChatMessageHistory accepts only a psycopg connection
connection_pool = psycopg_pool.AsyncConnectionPool(
    conninfo=settings.postgres.conninfo,
    open=False,
    min_size=1,
    max_size=20,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
        yield session


async def init_db_pool() -> None:
    await connection_pool.open()


async def close_db_pool() -> None:
    await connection_pool.close()
    await engine.dispose()
