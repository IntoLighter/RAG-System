from collections.abc import AsyncIterator

import psycopg_pool
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from dependency.settings import settings

engine = create_async_engine(
    "postgresql+psycopg://",
    connect_args=conninfo_to_dict(settings.postgres.conninfo),
    pool_pre_ping=True,
)

connection_pool = psycopg_pool.AsyncConnectionPool(
    conninfo=settings.postgres.conninfo,
    open=False,
    min_size=1,
    max_size=20,
    connection_class=AsyncConnection[DictRow],
    kwargs={"row_factory": dict_row},
)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
        yield session


async def init_db_pool() -> None:
    await connection_pool.open()


async def close_db_pool() -> None:
    await connection_pool.close()
    await engine.dispose()
