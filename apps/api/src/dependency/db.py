from collections.abc import AsyncIterator

import psycopg
import psycopg_pool

from dependency.settings import settings

connection_pool = psycopg_pool.AsyncConnectionPool(
    conninfo=settings.postgres.conninfo,
    open=False,
    min_size=1,
    max_size=20,
)


async def get_db_connection() -> AsyncIterator[psycopg.AsyncConnection]:
    async with connection_pool.connection() as conn:
        yield conn


async def init_db_pool() -> None:
    await connection_pool.open()


async def close_db_pool() -> None:
    await connection_pool.close()
